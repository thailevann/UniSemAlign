import model.backbone.resnet as resnet
from model.backbone.xception import xception
from model.backbone.uni import UNIWrapper
from model.backbone.conch import CONCHWrapper

import torch
from torch import nn
import torch.nn.functional as F
import math


class UNIBackboneCorr(nn.Module):
    """
    Backbone using UNI (UNIWrapper).
    Returns [c1, c4]: c1 is the output of block stage_low (default 2), c4 is the output of block stage_high (default 9).
    """

    def __init__(self, uni_cfg):
        super(UNIBackboneCorr, self).__init__()
        stage_low = int(uni_cfg.get("stage_low", 2))
        stage_high = int(uni_cfg.get("stage_high", 9))
        low_channels = int(uni_cfg.get("low_channels", 256))
        out_channels = int(uni_cfg.get("out_channels", 2048))
        pretrained = uni_cfg.get("pretrained", True)
        freeze_backbone = uni_cfg.get("freeze_backbone", True)

        self.uni = UNIWrapper(
            stage_low=stage_low,
            stage_high=stage_high,
            low_channels=low_channels,
            out_channels=out_channels,
            pretrained=pretrained,
        )
        if not freeze_backbone:
            for p in self.uni.parameters():
                p.requires_grad = True

        self.high_channels = out_channels
        self.low_channels = low_channels

    def base_forward(self, x):
        c1, c4 = self.uni(x)
        return [c1, c4]


class CONCHBackboneCorr(nn.Module):
    """
    Backbone using CONCH (CONCHWrapper).
    Returns [c1, c4] similar to ResNet/UNI; CONCH is patch-based, so sliding_window is recommended for large images.
    """

    def __init__(self, conch_cfg):
        super(CONCHBackboneCorr, self).__init__()
        stages = conch_cfg.get("stages", None)
        stage = conch_cfg.get("stage", 9)
        if stages is not None:
            block_indices = [int(s) for s in stages]
            block_index = None
        else:
            block_indices = None
            block_index = int(stage)

        out_channels = int(conch_cfg.get("out_channels", 2048))
        pretrained = conch_cfg.get("pretrained", True)
        freeze_backbone = conch_cfg.get("freeze_backbone", True)
        hf_auth_token = conch_cfg.get("hf_auth_token") or conch_cfg.get("hf_token")
        checkpoint_path = conch_cfg.get("checkpoint_path")

        self.conch = CONCHWrapper(
            out_channels=out_channels,
            block_index=block_index,
            block_indices=block_indices,
            pretrained=pretrained,
            hf_auth_token=hf_auth_token,
            checkpoint_path=checkpoint_path,
        )
        if not freeze_backbone:
            for p in self.conch.parameters():
                p.requires_grad = True

        self.high_channels = out_channels
        self.low_channels = 256
        self.low_reduce = nn.Sequential(
            nn.Conv2d(self.high_channels, self.low_channels, 1, bias=False),
            nn.BatchNorm2d(self.low_channels),
            nn.ReLU(True),
        )

    def base_forward(self, x):
        feat_high = self.conch(x)
        feat_low = F.interpolate(feat_high, scale_factor=2.0, mode="bilinear", align_corners=True)
        feat_low = self.low_reduce(feat_low)
        return [feat_low, feat_high]


class DeepLabV3Plus(nn.Module):
    def __init__(self, cfg):
        super(DeepLabV3Plus, self).__init__()

        backbone_name = cfg['backbone']
        if 'resnet' in backbone_name:
            self.backbone = resnet.__dict__[backbone_name](
                cfg['pretrain'],
                multi_grid=cfg['multi_grid'],
                replace_stride_with_dilation=cfg['replace_stride_with_dilation']
            )
            low_channels = 256
            high_channels = 2048
        elif backbone_name == 'xception':
            self.backbone = xception(True)
            low_channels = 256
            high_channels = 2048
        elif backbone_name == 'uni':
            uni_cfg = dict(cfg.get('uni', {}) or {})
            self.backbone = UNIBackboneCorr(uni_cfg)
            low_channels = self.backbone.low_channels
            high_channels = self.backbone.high_channels
        elif backbone_name == 'conch':
            conch_cfg = dict(cfg.get('conch', {}) or {})
            conch_cfg.setdefault('hf_token', cfg.get('hf_token'))
            self.backbone = CONCHBackboneCorr(conch_cfg)
            low_channels = self.backbone.low_channels
            high_channels = self.backbone.high_channels
        else:
            raise ValueError('Unsupported backbone: %s' % backbone_name)

        self.head = ASPPModule(high_channels, cfg['dilations'])

        self.reduce = nn.Sequential(nn.Conv2d(low_channels, 48, 1, bias=False),
                                    nn.BatchNorm2d(48),
                                    nn.ReLU(True))

        self.fuse = nn.Sequential(nn.Conv2d(high_channels // 8 + 48, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),
                                  nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True))

        self.classifier = nn.Conv2d(256, cfg['nclass'], 1, bias=True)

        self.proj = nn.Sequential(
            nn.Conv2d(high_channels, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

    def forward(self, x, need_fp=False):
        dict_return = {}
        h, w = x.shape[-2:]

        feats = self.backbone.base_forward(x)
        c1, c4 = feats[0], feats[-1]

        proj_feats = self.proj(c4)

        if need_fp:
            feats_decode = self._decode(torch.cat((c1, nn.Dropout2d(0.5)(c1))), torch.cat((c4, nn.Dropout2d(0.5)(c4))))
            outs = self.classifier(feats_decode)
            outs = F.interpolate(outs, size=(h, w), mode="bilinear", align_corners=True)
            out, out_fp = outs.chunk(2)
            dict_return['out'] = out
            dict_return['out_fp'] = out_fp
            if proj_feats is not None:
                dict_return['pixel_feats'] = proj_feats

            return dict_return

        feats_decode = self._decode(c1, c4)
        out = self.classifier(feats_decode)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)
        dict_return['out'] = out
        if proj_feats is not None:
            dict_return['pixel_feats'] = proj_feats
        return dict_return

    def _decode(self, c1, c4):
        c4 = self.head(c4)
        c4 = F.interpolate(c4, size=c1.shape[-2:], mode="bilinear", align_corners=True)

        c1 = self.reduce(c1)

        feature = torch.cat([c1, c4], dim=1)
        feature = self.fuse(feature)

        return feature


def ASPPConv(in_channels, out_channels, atrous_rate):
    block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
                                    dilation=atrous_rate, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU(True))
    return block


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        h, w = x.shape[-2:]
        pool = self.gap(x)
        return F.interpolate(pool, (h, w), mode="bilinear", align_corners=True)


class ASPPModule(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPPModule, self).__init__()
        out_channels = in_channels // 8
        rate1, rate2, rate3 = atrous_rates

        self.b0 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1)
        self.b2 = ASPPConv(in_channels, out_channels, rate2)
        self.b3 = ASPPConv(in_channels, out_channels, rate3)
        self.b4 = ASPPPooling(in_channels, out_channels)

        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU(True))

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        return self.project(y)
