import argparse
import os
import numpy as np
import torch
import torch.distributed as dist
from util.dist_helper import setup_distributed
from model.semseg.deeplabv3plus import DeepLabV3Plus

from torch.utils.data import DataLoader
import yaml
from dataset.semi import SemiDataset
from util.utils import AverageMeter, intersectionAndUnion


def evaluate(model, loader, mode, cfg):
    return_dict = {}
    model.eval()
    assert mode in ['original', 'center_crop', 'sliding_window']

    device = next(model.parameters()).device
    dist_on = dist.is_available() and dist.is_initialized()

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    with torch.no_grad():
        for img, mask, ids, img_ori in loader:
            img = img.to(device)
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
                        patch = img_pad[:, :, row:row + grid, col:col + grid]
                        res = model(patch)
                        pred = res['out']
                        final[:, :, row:row + grid, col:col + grid] += pred.softmax(dim=1)

                final = final[:, :, :H, :W]
                pred = final.argmax(dim=1)

            else:
                if mode == 'center_crop':
                    hh, ww = img.shape[-2:]
                    start_h, start_w = (hh - cfg['crop_size']) // 2, (ww - cfg['crop_size']) // 2
                    img = img[:, :, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]
                    mask = mask[:, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]

                res = model(img)
                pred = res['out'].argmax(dim=1)

            intersection, union, target = intersectionAndUnion(
                pred.cpu().numpy(),
                mask.numpy(),
                cfg['nclass'],
                255
            )

            reduced_intersection = torch.from_numpy(intersection).to(device)
            reduced_union = torch.from_numpy(union).to(device)
            reduced_target = torch.from_numpy(target).to(device)

            if dist_on:
                dist.all_reduce(reduced_intersection)
                dist.all_reduce(reduced_union)
                dist.all_reduce(reduced_target)

            intersection_meter.update(reduced_intersection.detach().cpu().numpy())
            union_meter.update(reduced_union.detach().cpu().numpy())
            target_meter.update(reduced_target.detach().cpu().numpy())

    eps = 1e-10
    inter = intersection_meter.sum
    uni = union_meter.sum

    iou_class = inter / (uni + eps)                    
    dice_class = (2.0 * inter) / (uni + inter + eps)   

    # mean over classes
    mIOU = float(np.mean(iou_class) * 100.0)
    mDice = float(np.mean(dice_class) * 100.0)
    mJaccard = mIOU

    return_dict['iou_class'] = iou_class * 100.0
    return_dict['dice_class'] = dice_class * 100.0
    return_dict['mIOU'] = mIOU
    return_dict['mJaccard'] = mJaccard
    return_dict['mDice'] = mDice

    if cfg.get('nclass', 0) == 2:
        return_dict['Jaccard_fg'] = float(iou_class[1] * 100.0)
        return_dict['Dice_fg'] = float(dice_class[1] * 100.0)

    return return_dict



def main():
    parser = argparse.ArgumentParser(description='Semi-Supervised Semantic Segmentation')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'],
                        help='Evaluate on val or test set')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--port', default=None, type=int)
    args = parser.parse_args()
    setup_distributed(port=args.port)
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN") or cfg.get("hf_token")
    if token:
        token = str(token).strip()
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token

    model = DeepLabV3Plus(cfg)
    ckpt = torch.load(args.checkpoint_path, weights_only=True)
    model.load_state_dict(ckpt, strict=False)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=False)

    id_path = cfg.get('val_id_path') if args.split == 'val' else cfg.get('test_id_path')
    evalset = SemiDataset(cfg['dataset'], cfg['data_root'], args.split, id_path=id_path)
    evalsampler = torch.utils.data.distributed.DistributedSampler(evalset)
    evalloader = DataLoader(evalset, batch_size=1, pin_memory=True, num_workers=4,
                            drop_last=False, sampler=evalsampler)

    model.eval()
    eval_mode = 'sliding_window' if cfg.get('backbone') in ('uni', 'conch') else 'original'
    res = evaluate(model, evalloader, eval_mode, cfg)
    mIOU = res['mIOU']
    mDice = res['mDice']
    mJaccard = res['mJaccard']
    iou_class = res['iou_class']
    print('Split:', args.split)
    print('mIOU (mJaccard):', mIOU)
    print('mDice:', mDice)
    if cfg.get('nclass', 0) == 2:
        print('Jaccard_fg:', res.get('Jaccard_fg'))
        print('Dice_fg:', res.get('Dice_fg'))
    print('IoU per class:', iou_class)


if __name__ == '__main__':
    main()
