# util/thresh_helper.py
import torch
import torch.distributed as dist


class ThreshController:

    def __init__(self, nclass, momentum, thresh_init=0.85):
        self.dist_on = dist.is_available() and dist.is_initialized()
        self.gpu_num = dist.get_world_size() if self.dist_on else 1

        self.nclass = nclass
        self.momentum = momentum

        self.thresh_global = torch.tensor(float(thresh_init))
        self.thresh_local = torch.tensor(float(thresh_init))

    def new_global_mask_pooling(self, pred, ignore_mask=None):
        """
        Computes a scalar threshold from predictions.

        pred: [B, C, H, W] logits
        ignore_mask: [B, H, W] (255 ignored), optional
        returns: {'new_global': scalar tensor on pred.device} or None
        """
        device = pred.device
        n, c, h, w = pred.shape

        # ---- Gather across GPUs only if DDP is active ----
        if self.dist_on:
            pred_gather = torch.zeros([n * self.gpu_num, c, h, w], device=device)
            dist.all_gather_into_tensor(pred_gather, pred)
            pred = pred_gather

            if ignore_mask is not None:
                ignore_mask_gather = torch.zeros([n * self.gpu_num, h, w],
                                                device=ignore_mask.device, dtype=torch.long)
                dist.all_gather_into_tensor(ignore_mask_gather, ignore_mask)
                ignore_mask = ignore_mask_gather

        # Local computation (single GPU or after gather)
        mask_pred = torch.argmax(pred, dim=1)           # [B, H, W]
        pred_softmax = pred.softmax(dim=1)              # [B, C, H, W]
        pred_conf = pred_softmax.max(dim=1)[0]          # [B, H, W]

        unique_cls = torch.unique(mask_pred)
        cls_num = int(unique_cls.numel())

        new_global = torch.tensor(0.0, device=device)
        valid_cls = 0

        for cls in unique_cls:
            cls_map = (mask_pred == cls)                # [B, H, W] bool
            if ignore_mask is not None:
                cls_map = cls_map & (ignore_mask != 255)

            if cls_map.sum().item() == 0:
                continue

            # max confidence over pixels predicted as this class
            cls_max_conf = pred_conf[cls_map].max()
            new_global = new_global + cls_max_conf
            valid_cls += 1

        if valid_cls > 0:
            return {"new_global": new_global / float(valid_cls)}
        else:
            return {"new_global": None}

    def thresh_update(self, pred, ignore_mask=None, update_g=False):
        """
        Updates global threshold via EMA:
          T <- m*T + (1-m)*T_new
        """
        thresh = self.new_global_mask_pooling(pred, ignore_mask)

        if update_g and thresh["new_global"] is not None:
            device = pred.device

            tg = self.thresh_global.to(device)
            ng = thresh["new_global"]
            if not torch.is_tensor(ng):
                ng = torch.tensor(float(ng), device=device)
            else:
                ng = ng.to(device)

            tg = self.momentum * tg + (1.0 - self.momentum) * ng
            # store on CPU to avoid persistent GPU state
            self.thresh_global = tg.detach().cpu()

    def get_thresh_global(self, device=None):
        """
        Returns scalar threshold.
        If device provided, returns tensor on that device.
        """
        if device is None:
            return self.thresh_global
        return self.thresh_global.to(device)
