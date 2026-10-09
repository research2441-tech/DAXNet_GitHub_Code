"""DAX-Net: degradation-aware cross-attention enhancement network (Sect. 3.3, Eqs. 11-16).

Visual encoder  : ImageNet-pretrained SqueezeNet 1.1 (default), ResNet-18, ResNet-50 or MobileNetV2
Auxiliary branch: illumination prior + gradient magnitude -> degradation tokens
Fusion          : bidirectional multi-head cross-attention at stride 16
Decoder         : four up-sampling stages with encoder skips and squeeze-and-excitation gates
Output          : residual reconstruction  I_hat = clip(y + R(y), 0, 1)
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------- encoders
class Encoder(nn.Module):
    """Wraps a torchvision backbone and returns features at strides 2, 4, 8, 16."""

    SPECS = {
        "squeezenet": ([1, 4, 7, 12], [64, 128, 256, 512]),
        "mobilenetv2": ([1, 3, 6, 13], [16, 24, 32, 96]),
        "resnet18": (None, [64, 64, 128, 256]),
        "resnet50": (None, [64, 256, 512, 1024]),
    }

    def __init__(self, name: str = "squeezenet", pretrained: bool = True):
        super().__init__()
        self.name = name
        taps, self.channels = self.SPECS[name]
        if name == "squeezenet":
            w = tvm.SqueezeNet1_1_Weights.IMAGENET1K_V1 if pretrained else None
            self.body = tvm.squeezenet1_1(weights=w).features[:13]
            self.taps = taps
        elif name == "mobilenetv2":
            w = tvm.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
            self.body = tvm.mobilenet_v2(weights=w).features[:14]
            self.taps = taps
        else:
            ctor = tvm.resnet18 if name == "resnet18" else tvm.resnet50
            wcls = tvm.ResNet18_Weights if name == "resnet18" else tvm.ResNet50_Weights
            net = ctor(weights=wcls.IMAGENET1K_V1 if pretrained else None)
            self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)
            self.pool = net.maxpool
            self.layer1, self.layer2, self.layer3 = net.layer1, net.layer2, net.layer3
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        if self.name.startswith("resnet"):
            f1 = self.stem(x)
            f2 = self.layer1(self.pool(f1))
            f3 = self.layer2(f2)
            f4 = self.layer3(f3)
            return [f1, f2, f3, f4]
        feats = []
        for i, layer in enumerate(self.body):
            x = layer(x)
            if i in self.taps:
                feats.append(x)
        return feats


# ---------------------------------------------------------------- auxiliary branch
def degradation_priors(y: torch.Tensor, box: int = 15) -> torch.Tensor:
    """Illumination prior L(p) = box(max_c y_c(p)) and gradient magnitude |grad Y| (Eq. 11)."""
    lum_max = y.max(dim=1, keepdim=True).values
    illum = F.avg_pool2d(lum_max, box, stride=1, padding=box // 2, count_include_pad=False)
    gray = 0.299 * y[:, :1] + 0.587 * y[:, 1:2] + 0.114 * y[:, 2:3]
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=y.dtype, device=y.device).view(1, 1, 3, 3) / 8
    gx = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="replicate"), kx)
    gy = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="replicate"), kx.transpose(2, 3))
    grad = torch.sqrt(gx ** 2 + gy ** 2 + 1e-12)
    return torch.cat([illum, grad], 1)


class AuxBranch(nn.Module):
    def __init__(self, d: int = 128):
        super().__init__()
        chs = [2, 16, 32, 64, d]
        layers = []
        for cin, cout in zip(chs[:-1], chs[1:]):
            layers += [nn.Conv2d(cin, cout, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
        self.net = nn.Sequential(*layers)

    def forward(self, y):
        return self.net(degradation_priors(y))


# ---------------------------------------------------------------- cross-attention (Eqs. 12-14)
class CrossAttention(nn.Module):
    """CA(q^{m'}, k^m, v^m) = [head_1, ..., head_h] W^O,
    head_i = softmax(q_i k_i^T / sqrt(d')) v_i."""

    def __init__(self, d: int = 128, heads: int = 4):
        super().__init__()
        assert d % heads == 0
        self.h, self.dh = heads, d // heads
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)
        self.norm_q = nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d)

    def forward(self, xq, xkv):
        b, t, d = xq.shape
        q = self.wq(self.norm_q(xq)).view(b, t, self.h, self.dh).transpose(1, 2)
        kv = self.norm_kv(xkv)
        k = self.wk(kv).view(b, -1, self.h, self.dh).transpose(1, 2)
        v = self.wv(kv).view(b, -1, self.h, self.dh).transpose(1, 2)
        att = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.dh), dim=-1)
        out = (att @ v).transpose(1, 2).reshape(b, t, d)
        return self.wo(out)


class SE(nn.Module):
    """Squeeze-and-excitation gate s = sigmoid(W2 relu(W1 GAP(X))) (Eq. 15)."""

    def __init__(self, c: int, r: int = 8):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(c, max(c // r, 4)), nn.ReLU(inplace=True),
                                nn.Linear(max(c // r, 4), c), nn.Sigmoid())

    def forward(self, x):
        s = self.fc(x.mean((2, 3)))
        return x * s[:, :, None, None]


class UpBlock(nn.Module):
    def __init__(self, cin, cskip, cout, use_se=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin + cskip, cout, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cout, cout, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True))
        self.se = SE(cout) if use_se else nn.Identity()

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.se(self.conv(torch.cat([x, skip], 1)))


# ---------------------------------------------------------------- full model
class DAXNet(nn.Module):
    def __init__(self, backbone: str = "squeezenet", pretrained: bool = True, d: int = 128,
                 heads: int = 4, use_cross_attention: bool = True, use_aux: bool = True,
                 use_se: bool = True):
        super().__init__()
        self.enc = Encoder(backbone, pretrained)
        c1, c2, c3, c4 = self.enc.channels
        self.use_ca, self.use_aux = use_cross_attention and use_aux, use_aux
        self.proj_vis = nn.Conv2d(c4, d, 1)
        if use_aux:
            self.aux = AuxBranch(d)
            if self.use_ca:
                self.ca_va = CrossAttention(d, heads)  # visual queries degradation tokens
                self.ca_av = CrossAttention(d, heads)  # degradation queries visual tokens
        self.fuse = nn.Conv2d(2 * d if use_aux else d, d, 1)
        self.up3 = UpBlock(d, c3, 128, use_se)
        self.up2 = UpBlock(128, c2, 64, use_se)
        self.up1 = UpBlock(64, c1, 32, use_se)
        self.head = nn.Sequential(nn.Conv2d(32 + 3, 32, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
                                  nn.Conv2d(32, 3, 3, 1, 1))

    def forward(self, y):
        f1, f2, f3, f4 = self.enc(y)
        xv = self.proj_vis(f4)
        b, d, h, w = xv.shape
        if self.use_aux:
            xa = F.interpolate(self.aux(y), size=(h, w), mode="bilinear", align_corners=False)
            tv = xv.flatten(2).transpose(1, 2)
            ta = xa.flatten(2).transpose(1, 2)
            if self.use_ca:
                tv, ta = tv + self.ca_va(tv, ta), ta + self.ca_av(ta, tv)   # Eq. (14)
            z = torch.cat([tv, ta], -1).transpose(1, 2).reshape(b, 2 * d, h, w)
        else:
            z = xv
        z = self.fuse(z)
        z = self.up3(z, f3)
        z = self.up2(z, f2)
        z = self.up1(z, f1)
        z = F.interpolate(z, size=y.shape[-2:], mode="bilinear", align_corners=False)
        r = self.head(torch.cat([z, y], 1))
        return torch.clamp(y + r, 0.0, 1.0)                                  # Eq. (16)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    for bb in ["squeezenet", "mobilenetv2", "resnet18", "resnet50"]:
        m = DAXNet(bb, pretrained=False)
        print(bb, f"{count_params(m) / 1e6:.2f} M")
