"""Run inference to compute scores and/or save segmentation masks."""
import argparse
import os
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import yaml
from PIL import Image

from util.dist_helper import setup_distributed
from model.semseg.deeplabv3plus import DeepLabV3Plus
from dataset.semi import SemiDataset
from util.utils import AverageMeter, intersectionAndUnion


def _color_map_crag():
    """Two-class palette: 0=background, 1=foreground, 255=ignore."""
    cmap = np.zeros((256, 3), dtype=np.uint8)
    cmap[0] = [0, 0, 0]
    cmap[1] = [255, 255, 255]
    cmap[255] = [255, 0, 0]
    return cmap


def run_inference(model, img, mode, cfg, device):
    """Run model and return predictions [B, H, W] as long tensor."""
    b, _, h, w = img.shape
    if mode == 'sliding_window':
        grid = cfg['crop_size']
        H, W = h, w
        pad_h = ((H + grid - 1) // grid) * grid
        pad_w = ((W + grid - 1) // grid) * grid
        if pad_h > H or pad_w > W:
            img_pad = torch.zeros(b, img.size(1), pad_h, pad_w, device=device, dtype=img.dtype)
            img_pad[:, :, :H, :W] = img
        else:
            img_pad = img
        final = torch.zeros(b, cfg['nclass'], pad_h, pad_w, device=device)
        stride = int(grid * 2 / 3)
        rows = list(range(0, pad_h - grid + 1, stride))
        cols = list(range(0, pad_w - grid + 1, stride))
        if rows[-1] != pad_h - grid:
            rows.append(pad_h - grid)
        if cols[-1] != pad_w - grid:
            cols.append(pad_w - grid)
        for row in rows:
            for col in cols:
                patch = img_pad[:, :, row : row + grid, col : col + grid]
                res = model(patch)
                final[:, :, row : row + grid, col : col + grid] += res['out'].softmax(dim=1)
        final = final[:, :, :H, :W]
        pred = final.argmax(dim=1)
    else:
        res = model(img)
        pred = res['out'].argmax(dim=1)
    return pred


def main():
    parser = argparse.ArgumentParser(description='Generate segmentation masks for visualization')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'])
    parser.add_argument('--out-dir', type=str, default='visual', help='Directory to save masks')
    parser.add_argument('--mode', type=str, default='both', choices=['score', 'mask', 'both'],
                        help='score=metrics only; mask=only masks; both=metrics and masks')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--port', default=None, type=int)
    args = parser.parse_args()

    setup_distributed(port=args.port)
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    token = os.environ.get('HUGGINGFACE_HUB_TOKEN') or os.environ.get('HF_TOKEN') or cfg.get('hf_token')
    if token:
        os.environ['HF_TOKEN'] = os.environ['HUGGINGFACE_HUB_TOKEN'] = str(token).strip()

    model = DeepLabV3Plus(cfg)
    raw = torch.load(args.checkpoint_path, map_location='cpu', weights_only=False)
    state = raw.get('state_dict', raw) if isinstance(raw, dict) else raw
    model.load_state_dict(state, strict=True)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    id_path = cfg.get('val_id_path') if args.split == 'val' else cfg.get('test_id_path')
    evalset = SemiDataset(cfg['dataset'], cfg['data_root'], args.split, id_path=id_path)
    evalsampler = torch.utils.data.distributed.DistributedSampler(evalset, shuffle=False)
    loader = DataLoader(evalset, batch_size=1, pin_memory=True, num_workers=4, drop_last=False, sampler=evalsampler)

    eval_mode = 'sliding_window' if cfg.get('backbone') in ('uni', 'conch') else 'original'
    device = next(model.parameters()).device
    cmap = _color_map_crag()
    save_mask = args.mode in ('mask', 'both')
    do_score = args.mode in ('score', 'both')
    dist_on = dist.is_available() and dist.is_initialized()

    if local_rank == 0 and save_mask:
        os.makedirs(args.out_dir, exist_ok=True)

    if do_score:
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()

    model.eval()
    with torch.no_grad():
        for img, mask, ids, img_ori in loader:
            img = img.to(device)
            pred = run_inference(model, img, eval_mode, cfg, device)
            mask_np = mask.numpy()

            if do_score:
                intersection, union, target = intersectionAndUnion(
                    pred.cpu().numpy(), mask_np, cfg['nclass'], 255)
                reduced_i = torch.from_numpy(intersection).to(device)
                reduced_u = torch.from_numpy(union).to(device)
                reduced_t = torch.from_numpy(target).to(device)
                if dist_on:
                    dist.all_reduce(reduced_i)
                    dist.all_reduce(reduced_u)
                    dist.all_reduce(reduced_t)
                intersection_meter.update(reduced_i.detach().cpu().numpy())
                union_meter.update(reduced_u.detach().cpu().numpy())
                target_meter.update(reduced_t.detach().cpu().numpy())

            if save_mask:
                for i in range(pred.shape[0]):
                    if local_rank != 0:
                        continue
                    gt_rel_path = ids[i].split(' ')[1]
                    gt_name = os.path.basename(gt_rel_path)
                    out_path = os.path.join(args.out_dir, gt_name)
                    pred_np = pred[i].cpu().numpy().astype(np.uint8)
                    pred_pil = Image.fromarray(pred_np, mode='P')
                    pred_pil.putpalette(cmap.tobytes())
                    pred_pil.save(out_path)
                    print('Saved:', out_path)

    if do_score and local_rank == 0:
        eps = 1e-10
        inter = intersection_meter.sum
        uni = union_meter.sum
        iou_class = inter / (uni + eps)
        dice_class = (2.0 * inter) / (uni + inter + eps)
        mIOU = float(np.mean(iou_class) * 100.0)
        mDice = float(np.mean(dice_class) * 100.0)
        print('Split:', args.split)
        print('mIOU (mJaccard):', mIOU)
        print('mDice:', mDice)
        if cfg.get('nclass', 0) == 2:
            print('Jaccard_fg:', float(iou_class[1] * 100.0))
            print('Dice_fg:', float(dice_class[1] * 100.0))
        print('IoU per class:', iou_class * 100.0)


if __name__ == '__main__':
    main()
