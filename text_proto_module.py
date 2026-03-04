import math
import os
from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class TextProtoHead(nn.Module):
    """
    Text + prototype head.

    - Takes a pixel feature map [B, D, H, W] (projected to D, default 256).
    - Learns prototypes P ∈ R^{C × D}.
    - Optionally learns text embeddings Z_t ∈ R^{C × D} (CoOp-style).
    - Outputs:
        + S_vp: semantic logits from pixel ↔ prototype.
        + S_vt: semantic logits from pixel ↔ text (if enabled).
    - Also provides auxiliary losses on labeled images:
        + L_pixel_proto_ce
        + L_proto_text_align
    """

    def __init__(self, nclass: int, cfg: Dict):
        super().__init__()
        self.nclass = int(nclass)
        self.cfg = cfg

        self.use_text = True
        self.use_learnable_proto = True

        self.coop_cfg: Dict = cfg.get("coop", {}) or {}
        self.coop_learnable: bool = bool(self.coop_cfg.get("learnable", True))
        _n_ctx: int = int(self.coop_cfg.get("n_ctx", 0) or 0)
        self.n_ctx: int = 0 if not self.coop_learnable else _n_ctx
        self.text_encoder: str = str(cfg.get("text_encoder", "") or "").lower()
        class_names_cfg = cfg.get("class_names", None)
        if class_names_cfg is not None:
            self.class_names = [str(c) for c in class_names_cfg]
        else:
            self.class_names = None

        # Loss weights
        self.lambda_pixel_proto_ce = 1.0
        self.lambda_pixel_text_ce = 1.0
        self.lambda_proto_text_align = 1.0

        # Fusion weights cho pseudo-label unlabeled: L_final = γ*L_deeplab + α*S_vp + β*S_vt
        self.gamma_deeplab_logits = 1.0
        self.alpha_proto_logits = 0.1
        self.beta_text_logits = 0.1

        # Projection dim D (matches conv proj in DeepLabV3Plus, default 256)
        self.embed_dim = 256

        # State for CoOp with the CONCH text encoder
        self.conch_text_model: Optional[nn.Module] = None
        self.conch_tokenize = None
        self.conch_prompt_token_ids: Optional[torch.Tensor] = None  # [C, L]

        # Learnable prototypes P ∈ R^{C × D}
        init_scale = 1.0 / math.sqrt(self.embed_dim)
        proto = torch.randn(self.nclass, self.embed_dim) * init_scale
        if self.use_learnable_proto:
            self.prototypes = nn.Parameter(proto)
        else:
            self.register_buffer("prototypes", proto)

        # CoOp-style context tokens at the token level when using the CONCH text encoder:
        #   ctx_tokens: [C, n_ctx, D_token], one set of n_ctx tokens per class.
        # Without CONCH, the text branch uses a simple learnable Z_t (no ctx_mean feature-level combination).
        self.ctx_tokens: Optional[torch.Tensor] = None

        # Text embedding / projection into the shared space:
        # - With CONCH: text_feat_from_conch -> text_proj -> R^{C × embed_dim}
        # - With random init (no CONCH): directly learnable Z_t ∈ R^{C × embed_dim}
        self.text_proj: Optional[nn.Module] = None
        self.text_emb: Optional[torch.Tensor] = None

        if self.use_text:
            if self.text_encoder == "conch" and self.class_names:
                try:
                    from conch.open_clip_custom import (
                        create_model_from_pretrained,
                        get_tokenizer,
                    )

                    token = (
                        os.environ.get("HUGGINGFACE_HUB_TOKEN")
                        or os.environ.get("HF_TOKEN")
                    )
                    if token is None:
                        raise RuntimeError(
                            "CONCH text encoder requires HF_TOKEN / HUGGINGFACE_HUB_TOKEN."
                        )

                    model, _ = create_model_from_pretrained(
                        "conch_ViT-B-16",
                        "hf_hub:MahmoodLab/conch",
                        hf_auth_token=token,
                    )
                    model.eval()
                    for p in model.parameters():
                        p.requires_grad = False
                    self.conch_text_model = model


                    hf_tokenizer = get_tokenizer()
                    prompts = []
                    for cname in self.class_names:
                        cname = cname.strip()
                        if not cname:
                            continue
                        if self.n_ctx > 0:
                            filler = " ".join(["x"] * self.n_ctx)
                            prompts.append(f"{filler} {cname}")
                        else:
                            prompts.append(cname)
                    if not prompts:
                        raise RuntimeError("No valid prompt for CONCH text encoder.")

                    with torch.no_grad():
                        device = next(model.parameters()).device
                        enc = hf_tokenizer(
                            prompts,
                            max_length=127,
                            add_special_tokens=True,
                            truncation=True,
                            padding="max_length",
                            return_tensors="pt",
                        )
                        input_ids = enc["input_ids"]  # [C, 127]
                        pad_id = hf_tokenizer.pad_token_id
                        text_tokens = F.pad(
                            input_ids, (0, 1), value=pad_id
                        ).to(device)  # [C, 128]

                        self.conch_prompt_token_ids = text_tokens.detach().cpu()

                        base_text_feat = model.encode_text(text_tokens)  # [C, D_text]
                        D_text = base_text_feat.shape[1]

                        if self.n_ctx > 0:
                            text_encoder = getattr(model, "text", model)
                            token_emb = text_encoder.token_embedding(text_tokens)  # [C, L, D_tok]
                            D_tok = token_emb.shape[-1]
                            scale_tok = 1.0 / math.sqrt(D_tok)
                            ctx_tokens = torch.randn(
                                self.nclass,
                                self.n_ctx,
                                D_tok,
                                device=token_emb.device,
                                dtype=token_emb.dtype,
                            ) * scale_tok
                            self.ctx_tokens = nn.Parameter(ctx_tokens.cpu())

                        # Linear projection maps text_feat from D_text -> embed_dim=256
                        self.text_proj = nn.Linear(D_text, self.embed_dim, bias=False)
                        nn.init.normal_(
                            self.text_proj.weight, std=1.0 / math.sqrt(D_text)
                        )

                except Exception as e:
                    raise RuntimeError(
                        "Failed to initialize CONCH CoOp text encoder. "
                        "Check CONCH, open_clip, and HF token configuration."
                    ) from e

            elif self.text_encoder == "conch":
                raise RuntimeError(
                    "text_encoder='conch' requires class_names in the text_proto block."
                )

            if self.text_encoder != "conch":
                if self.text_emb is None:
                    text_emb = torch.randn(self.nclass, self.embed_dim) * init_scale
                    self.text_emb = nn.Parameter(text_emb)

            if not self.coop_learnable:
                if self.text_emb is not None:
                    self.text_emb.requires_grad = False
                if self.text_proj is not None:
                    for p in self.text_proj.parameters():
                        p.requires_grad = False
                if self.ctx_tokens is not None:
                    self.ctx_tokens.requires_grad = False
        else:
            self.text_proj = None
            self.text_emb = None

        self.ce_ignore = nn.CrossEntropyLoss(ignore_index=255)

    def _get_text_features(self) -> Optional[torch.Tensor]:
        """Return text embedding T with dimension D = self.embed_dim."""
        if not self.use_text:
            return None

        if self.text_encoder == "conch":
            if (
                self.conch_text_model is None
                or self.conch_prompt_token_ids is None
                or self.text_proj is None
            ):
                raise RuntimeError(
                    "CONCH CoOp text encoder is not properly initialized."
                )

            model = self.conch_text_model
            text_encoder = getattr(model, "text", model)
            device = next(text_encoder.parameters()).device
            dtype = next(text_encoder.parameters()).dtype

            token_ids = self.conch_prompt_token_ids.to(device)  # [C, L]

            x = text_encoder.token_embedding(token_ids).to(dtype)  # [C, L, D_tok]

            if self.n_ctx > 0:
                if self.ctx_tokens is None:
                    raise RuntimeError("ctx_tokens for CoOp-CONCH are not initialized.")
                ctx = self.ctx_tokens.to(device)
                n_ctx = min(self.n_ctx, x.shape[1] - 1)
                if n_ctx > 0:
                    x[:, 1 : 1 + n_ctx, :] = ctx[:, :n_ctx, :].to(dtype)

            pos_embed = getattr(text_encoder, "positional_embedding", None)
            if pos_embed is None:
                pos_embed = getattr(text_encoder, "text_positional_embedding", None)
            if pos_embed is None:
                raise RuntimeError("CONCH text encoder has no positional_embedding.")
            x = x + pos_embed.to(dtype)

            transformer = getattr(text_encoder, "transformer", None) or getattr(
                text_encoder, "text_transformer", None
            )
            if transformer is None:
                raise RuntimeError("CONCH text encoder has no transformer for text.")
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD

            ln_final = getattr(text_encoder, "ln_final", None) or getattr(
                text_encoder, "text_ln_final", None
            )
            if ln_final is None:
                raise RuntimeError("CONCH text encoder has no ln_final for text.")
            x = ln_final(x).to(dtype)

            eot_indices = token_ids.argmax(dim=-1)
            x = x[torch.arange(x.shape[0], device=device), eot_indices]

            text_proj_clip = getattr(text_encoder, "text_projection", None)
            if text_proj_clip is None:
                raise RuntimeError("CONCH text encoder has no text_projection.")
            x = x @ text_proj_clip

            # Finally, project into embed_dim (256)
            T = self.text_proj(x)  # [C, embed_dim]
            return T

        if self.text_encoder != "conch":
            if self.text_emb is None:
                raise RuntimeError(
                    "text_emb is not initialized for text_encoder != 'conch'."
                )
            return self.text_emb

        raise RuntimeError("Text encoder configuration is invalid in _get_text_features.")

    def _semantic_logits(
        self,
        pixel_feats: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        pixel_feats: [B, D, H, W]
        Returns:
          - S_vp: [B, C, H, W]
          - S_vt: [B, C, H, W] or None
        """
        B, D, H, W = pixel_feats.shape
        assert D == self.embed_dim, f"Expected embed_dim={self.embed_dim}, got {D}"

        # [B, D, H, W] -> [B, N, D]
        v = pixel_feats.view(B, D, H * W).permute(0, 2, 1)
        v = F.normalize(v, dim=-1)

        # Prototype logits
        p = F.normalize(self.prototypes, dim=-1)  # [C, D]
        S_vp = torch.matmul(v, p.t())  # [B, N, C]
        S_vp = S_vp.permute(0, 2, 1).view(B, self.nclass, H, W)

        S_vt = None
        t_raw = self._get_text_features()
        if self.use_text and (t_raw is not None):
            t = F.normalize(t_raw, dim=-1)  # [C, D]
            S_vt = torch.matmul(v, t.t())  # [B, N, C]
            S_vt = S_vt.permute(0, 2, 1).view(B, self.nclass, H, W)

        return S_vp, S_vt

    @torch.no_grad()
    def fuse_unlabeled_logits(
        self,
        pixel_feats: torch.Tensor,
        base_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse logits for the unlabeled branch:
          L_final = γ*L_deeplab + α*S_vp + β*S_vt, then take argmax/softmax for pseudo-labels.

        pixel_feats: [B, D, H, W]
        base_logits: [B, C, H, W] (DeepLab logits)
        """
        if self.alpha_proto_logits == 0.0 and (
            (not self.use_text) or self.beta_text_logits == 0.0
        ):
            return base_logits

        B, C, H, W = base_logits.shape
        pf = pixel_feats
        if pf.shape[-2:] != (H, W):
            pf = F.interpolate(pf, size=(H, W), mode="bilinear", align_corners=True)

        S_vp, S_vt = self._semantic_logits(pf)

        L_final = self.gamma_deeplab_logits * base_logits + self.alpha_proto_logits * S_vp
        if self.use_text and (S_vt is not None) and self.beta_text_logits != 0.0:
            L_final = L_final + self.beta_text_logits * S_vt

        return L_final

    def compute_labeled_losses(
        self,
        pixel_feats: torch.Tensor,
        mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        pixel_feats: [B, D, H, W]
        mask: [B, H, W] with values in {0..C-1} or 255 (ignore)
        """
        B, D, H, W = pixel_feats.shape
        pf = pixel_feats

        if mask.shape[-2:] != (H, W):
            pf = F.interpolate(pf, size=mask.shape[-2:], mode="bilinear", align_corners=True)
            H, W = mask.shape[-2:]

        S_vp, S_vt = self._semantic_logits(pf)  # [B, C, H, W]

        losses: Dict[str, torch.Tensor] = {}

        # 1) Pixel ↔ prototype CE (supervised)
        if self.lambda_pixel_proto_ce > 0.0:
            loss_ce = self.ce_ignore(S_vp, mask.long())
            losses["pixel_proto_ce"] = loss_ce

        if self.lambda_pixel_text_ce > 0.0 and self.use_text and S_vt is not None:
            loss_text_ce = self.ce_ignore(S_vt, mask.long())
            losses["pixel_text_ce"] = loss_text_ce

        if self.lambda_proto_text_align > 0.0 and self.use_text:
            t_raw = self._get_text_features()
            if t_raw is not None:
                p = F.normalize(self.prototypes, dim=-1)
                t = F.normalize(t_raw, dim=-1)
                loss_align = F.mse_loss(p, t)
                losses["proto_text_align"] = loss_align

        return losses
