
import math
import os
import torch
import torch.nn as nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _tokens_to_feature_map(x: torch.Tensor) -> torch.Tensor:
    """[B, N+1, C] or [B, N, C] -> [B, C, H, W], dropping CLS if present."""
    if x.dim() == 3 and x.shape[1] > 1:
        # Drop the first token (CLS) if present
        x = x[:, 1:, :] if x.shape[1] % 2 == 1 else x
    B, N, C = x.shape
    H = W = int(math.sqrt(N))
    if H * W != N:
        return x.transpose(1, 2)  # [B, C, N] fallback, caller may need to handle
    return x.transpose(1, 2).reshape(B, C, H, W)


def _get_conch_via_package(pretrained: bool, hf_auth_token: str = None, checkpoint_path: str = None):
    """
    Load CONCH via the official package (conch.open_clip_custom).
    """
    try:
        from conch.open_clip_custom import create_model_from_pretrained
    except ImportError as e:
        raise ImportError(
            "CONCH backbone requires the official package: pip install git+https://github.com/Mahmoodlab/CONCH.git"
        ) from e

    token = hf_auth_token or os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if checkpoint_path:
        pretrained_path = checkpoint_path
    elif pretrained:
        pretrained_path = "hf_hub:MahmoodLab/conch"
        if not token:
            raise ValueError(
                "CONCH from HuggingFace requires hf_auth_token via config (conch.hf_auth_token or hf_token) "
                "or env HF_TOKEN / HUGGINGFACE_HUB_TOKEN."
            )
    else:
        raise ValueError(
            "CONCH wrapper requires pretrained=True (HF) or conch.checkpoint_path (path to pytorch_model.bin)."
        )

    model, _preprocess = create_model_from_pretrained(
        "conch_ViT-B-16",
        pretrained_path,
        hf_auth_token=token if isinstance(pretrained_path, str) and pretrained_path.startswith("hf_hub") else None,
    )
    # OpenCLIP-style: image encoder is model.visual
    visual = getattr(model, "visual", None) or getattr(model, "image_encoder", None) or model
    embed_dim = getattr(visual, "embed_dim", None) or getattr(visual, "output_dim", None)
    if embed_dim is None:
        for name in ("embed_dim", "output_dim", "num_features", "width"):
            embed_dim = getattr(visual, name, None)
            if embed_dim is not None:
                break
    if embed_dim is None and hasattr(visual, "transformer"):
        embed_dim = getattr(visual.transformer, "width", 768)
    if embed_dim is None:
        embed_dim = 768
    return model, int(embed_dim)


def _contains_module(module, target):
    """Return True if target is the module itself or contained inside it."""
    if module is target:
        return True
    for child in module.children():
        if _contains_module(child, target):
            return True
    return False


def _run_stem(module, x: torch.Tensor, blocks_ref: nn.Module, depth: int = 0):
    """
    Run the stem: all layers before blocks_ref, following the original forward order.
    If the blocks live inside a child module (e.g. transformer.resblocks), recurse into that child.
    """
    if module is blocks_ref or depth > 5:
        return x
    for name, child in module.named_children():
        if child is blocks_ref:
            return x
        if _contains_module(child, blocks_ref):
            x = _run_stem(child, x, blocks_ref, depth + 1)
        else:
            x = child(x)
    return x


def _extract_patch_embed(visual, x: torch.Tensor, blocks_ref: nn.Module):
    """
    Run the stem (patch embed + pos + ln_pre, ...) up to just before the blocks.
    Then add the CLS token and positional embedding if the model expects them but the stem has not added them.
    """
    x = _run_stem(visual, x, blocks_ref)
    obj = getattr(visual, "trunk", visual)
    # Add CLS token only when it is missing (N is the number of patches, e.g. 256 = 16*16)
    if hasattr(obj, "class_embedding") and x.dim() == 3:
        B, N, C = x.shape
        r = int(round(N ** 0.5))
        if r * r == N:  # N is the number of pure patches, without CLS
            cls = obj.class_embedding.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
            x = torch.cat([cls, x], dim=1)
    # Add positional embedding if available (using "is None" because the attribute may be a tensor)
    pos_embed = getattr(obj, "positional_embedding", None)
    if pos_embed is None:
        pos_embed = getattr(obj, "pos_embed", None)
    if pos_embed is None:
        pos_embed = getattr(obj, "_pos_embed", None)
    if pos_embed is not None and not callable(pos_embed) and x.dim() == 3:
        pe = pos_embed
        if pe.dim() == 2:
            pe = pe.unsqueeze(0).expand(x.shape[0], -1, -1)
        if pe.shape[1] == x.shape[1]:
            x = x + pe
        elif pe.shape[1] == x.shape[1] + 1:
            x = x + pe[:, 1:, :]
        elif pe.shape[1] == x.shape[1] - 1:
            x = x + torch.cat([torch.zeros_like(pe[:, :1]), pe], dim=1)
    return x


def _log_visual_structure(visual, prefix="visual"):
    """Log the visual encoder structure for debugging."""
    import logging
    log = logging.getLogger("global")
    for name, child in visual.named_children():
        cls = type(child).__name__
        if isinstance(child, nn.ModuleList):
            log.warning("%s.%s = ModuleList(len=%d)", prefix, name, len(child))
        else:
            log.warning("%s.%s = %s", prefix, name, cls)


def _get_blocks(visual):
    """Return the list of transformer blocks (resblocks or blocks) for common OpenCLIP/CONCH layouts."""
    # OpenCLIP: visual.transformer.resblocks or .blocks
    trans = getattr(visual, "transformer", None)
    if trans is not None:
        blocks = getattr(trans, "resblocks", None) or getattr(trans, "blocks", None)
        if blocks is not None:
            return blocks
    # Directly on the visual module
    blocks = getattr(visual, "resblocks", None) or getattr(visual, "blocks", None)
    if blocks is not None:
        return blocks
    # HuggingFace-style: .encoder.layers
    enc = getattr(visual, "encoder", None)
    if enc is not None:
        blocks = getattr(enc, "layers", None) or getattr(enc, "block", None)
        if blocks is not None:
            return blocks
    # Some models use .trunk to hold the encoder
    trunk = getattr(visual, "trunk", None)
    if trunk is not None:
        blocks = getattr(trunk, "resblocks", None) or getattr(trunk, "blocks", None)
        if blocks is not None:
            return blocks
    # Fallback: find any ModuleList of length 12 (ViT-B) or 6–24 (typical transformer layers)
    for name, child in visual.named_children():
        if isinstance(child, nn.ModuleList) and 6 <= len(child) <= 24:
            return child
    return None


class CONCHWrapper(nn.Module):
    """
    CONCH image encoder wrapper.
    - Prefer loading via conch.open_clip_custom.create_model_from_pretrained.
    - Extract features from selected blocks (block_indices) -> concat -> conv1x1 -> [B, out_channels, H', W'].
    - Requires hf_auth_token (config or env) when pretrained=True.
    """

    def __init__(
        self,
        out_channels: int = 2048,
        block_index: int = None,
        block_indices: list = None,
        pretrained: bool = True,
        hf_auth_token: str = None,
        checkpoint_path: str = None,
    ):
        super().__init__()
        self.out_channels = out_channels
        self._model, self._conch_dim = _get_conch_via_package(
            pretrained, hf_auth_token=hf_auth_token, checkpoint_path=checkpoint_path
        )

        # Freeze all CONCH weights (linear-probe style)
        for p in self._model.parameters():
            p.requires_grad = False

        self.proj = nn.Conv2d(self._conch_dim, out_channels, 1)
        nn.init.kaiming_normal_(self.proj.weight, mode="fan_out", nonlinearity="relu")
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, target_size=None) -> torch.Tensor:
        if x.shape[1] == 3 and x.max() <= 1.0 + 1e-5:
            mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
            std = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
            x = (x - mean) / std

        with torch.no_grad():
            _global, tokens = self._model._encode_image(x, normalize=False)
            if tokens is None:
                raise RuntimeError("CONCH _encode_image did not return patch tokens.")
            feat = _tokens_to_feature_map(tokens)  # [B, C, H', W']

        feat = self.proj(feat)
        if target_size is not None:
            feat = torch.nn.functional.interpolate(
                feat, size=target_size, mode="bilinear", align_corners=False
            )
        return feat

