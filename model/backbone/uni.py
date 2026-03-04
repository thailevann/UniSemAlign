import math
import torch
import torch.nn as nn

# ImageNet normalize for UNI (expects similar input distribution)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _tokens_to_feature_map(x: torch.Tensor) -> torch.Tensor:
    """[B, N+1, C] -> [B, C, H, W], remove CLS."""
    x = x[:, 1:, :]
    B, N, C = x.shape
    H = W = int(math.sqrt(N))
    return x.transpose(1, 2).reshape(B, C, H, W)


class UNIWrapper(nn.Module):
    """
    UNI backbone wrapper that returns c1 (low-level) and c4 (high-level) from two selected blocks.
    - stage_low: block index used for c1 (default 2).
    - stage_high: block index used for c4 (default 9).
    - A single forward pass through UNI, taking outputs at those blocks and projecting to low_channels/out_channels.
    """

    def __init__(
        self,
        stage_low: int = 2,
        stage_high: int = 9,
        low_channels: int = 256,
        out_channels: int = 2048,
        pretrained: bool = True,
    ):
        super().__init__()
        try:
            import timm
        except ImportError:
            raise ImportError("UNI wrapper requires timm. Please install with `pip install timm`.")

        self.stage_low = int(stage_low)
        self.stage_high = int(stage_high)
        self.low_channels = low_channels
        self.out_channels = out_channels

        self._uni = timm.create_model(
            "hf-hub:MahmoodLab/uni",
            pretrained=pretrained,
            init_values=1e-5,
            dynamic_img_size=True,
        )

        for p in self._uni.parameters():
            p.requires_grad = False

        self._uni_dim = getattr(self._uni, "embed_dim", None) or getattr(
            self._uni, "num_features", 1024
        )

        self.proj_low = nn.Conv2d(self._uni_dim, low_channels, 1)
        self.proj_high = nn.Conv2d(self._uni_dim, out_channels, 1)
        nn.init.kaiming_normal_(self.proj_low.weight, mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(self.proj_high.weight, mode="fan_out", nonlinearity="relu")
        if self.proj_low.bias is not None:
            nn.init.zeros_(self.proj_low.bias)
        if self.proj_high.bias is not None:
            nn.init.zeros_(self.proj_high.bias)

    def forward(self, x: torch.Tensor, target_size=None):
        """
        x: [B, 3, H, W] in [0,1], normalized to ImageNet for UNI.
        If target_size is set, both c1 and c4 are interpolated to that size.
        Returns (c1, c4) with shapes [B, low_channels, h, w] and [B, out_channels, h, w].
        """
        if x.shape[1] == 3 and x.max() <= 1.0 + 1e-5:
            mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
            std = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
            x = (x - mean) / std

        with torch.no_grad():
            x = self._uni.patch_embed(x)
            x = self._uni._pos_embed(x)
            feat_low = None
            feat_high = None
            for i, blk in enumerate(self._uni.blocks):
                x = blk(x)
                if i == self.stage_low:
                    feat_low = _tokens_to_feature_map(x)  # [B, C_uni, h, w]
                if i == self.stage_high:
                    feat_high = _tokens_to_feature_map(x)

        if feat_low is None or feat_high is None:
            raise RuntimeError(
                f"UNIWrapper: stage_low={self.stage_low}, stage_high={self.stage_high}; "
                "must run enough blocks to obtain both features. Check the number of UNI blocks."
            )

        c1 = self.proj_low(feat_low)
        c4 = self.proj_high(feat_high)

        if target_size is not None:
            c1 = torch.nn.functional.interpolate(
                c1, size=target_size, mode="bilinear", align_corners=False
            )
            c4 = torch.nn.functional.interpolate(
                c4, size=target_size, mode="bilinear", align_corners=False
            )
        return c1, c4

