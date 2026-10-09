"""Re-implemented comparison networks retrained on the same paired SDD tiles (Table 2).
Restormer and NAFNet are taken from their official repositories (see README)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SRCNN(nn.Module):
    """Three-layer CNN (9-5-5) used as the plain CNN baseline, operating at x1 scale."""

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 64, 9, padding=4)
        self.c2 = nn.Conv2d(64, 32, 5, padding=2)
        self.c3 = nn.Conv2d(32, 3, 5, padding=2)

    def forward(self, y):
        return torch.clamp(y + self.c3(F.relu(self.c2(F.relu(self.c1(y))))), 0, 1)


class DoubleConv(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(True),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(True))


class UNet(nn.Module):
    """Classic five-level U-Net (64-1024 channels, transposed-convolution up-sampling)."""

    def __init__(self, base: int = 64):
        super().__init__()
        ch = [base * 2 ** i for i in range(5)]
        self.downs = nn.ModuleList([DoubleConv(3, ch[0])] + [DoubleConv(ch[i], ch[i + 1]) for i in range(4)])
        self.ups = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, 2) for i in reversed(range(4))])
        self.dec = nn.ModuleList([DoubleConv(ch[i] * 2, ch[i]) for i in reversed(range(4))])
        self.out = nn.Conv2d(ch[0], 3, 1)

    def forward(self, y):
        skips, x = [], y
        for i, d in enumerate(self.downs):
            x = d(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for up, dec in zip(self.ups, self.dec):
            s = skips.pop()
            x = dec(torch.cat([up(x), s], 1))
        return torch.clamp(y + self.out(x), 0, 1)


class RDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv2d(nf + i * gc, gc if i < 4 else nf, 3, 1, 1) for i in range(5)])

    def forward(self, x):
        feats = [x]
        for i, c in enumerate(self.convs):
            out = c(torch.cat(feats, 1))
            if i < 4:
                feats.append(F.leaky_relu(out, 0.2))
        return x + 0.2 * out


class RRDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.b = nn.Sequential(RDB(nf, gc), RDB(nf, gc), RDB(nf, gc))

    def forward(self, x):
        return x + 0.2 * self.b(x)


class RRDBNet(nn.Module):
    """ESRGAN generator (23 RRDBs) adapted to x1 restoration with a 4x pixel-unshuffle input."""

    def __init__(self, nf=64, nb=23, gc=32, unshuffle=4):
        super().__init__()
        self.s = unshuffle
        self.first = nn.Conv2d(3 * unshuffle ** 2, nf, 3, 1, 1)
        self.trunk = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1)
        self.last = nn.Conv2d(nf, 3 * unshuffle ** 2, 3, 1, 1)

    def forward(self, y):
        x = F.pixel_unshuffle(y, self.s)
        f = self.first(x)
        f = f + self.trunk_conv(self.trunk(f))
        return torch.clamp(y + F.pixel_shuffle(self.last(f), self.s), 0, 1)


class DCENet(nn.Module):
    """Zero-DCE curve estimator (7 conv layers, 32 channels, 8 curve iterations)."""

    def __init__(self, nf=32, iters=8):
        super().__init__()
        self.iters = iters
        self.e1 = nn.Conv2d(3, nf, 3, 1, 1)
        self.e2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.e3 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.e4 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.e5 = nn.Conv2d(nf * 2, nf, 3, 1, 1)
        self.e6 = nn.Conv2d(nf * 2, nf, 3, 1, 1)
        self.e7 = nn.Conv2d(nf * 2, 3 * iters, 3, 1, 1)

    def forward(self, y):
        x1 = F.relu(self.e1(y)); x2 = F.relu(self.e2(x1)); x3 = F.relu(self.e3(x2)); x4 = F.relu(self.e4(x3))
        x5 = F.relu(self.e5(torch.cat([x3, x4], 1))); x6 = F.relu(self.e6(torch.cat([x2, x5], 1)))
        r = torch.tanh(self.e7(torch.cat([x1, x6], 1)))
        x = y
        for a in torch.split(r, 3, dim=1):
            x = x + a * (x * x - x)   # LE(x) = x + alpha x (1 - x)
        return torch.clamp(x, 0, 1)
