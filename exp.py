"""Experiment driver for the DAX-Net revision (all numbers in the paper come from here).

python exp.py prepare            # fixed degraded val/test sets
python exp.py doa                # DOA search for (lambda2, lambda3, eta0)
python exp.py train <name>       # train one configuration (see CONFIGS)
python exp.py evaluate <name>    # test-set quality metrics (+ saves outputs for downstream)
python exp.py downstream         # classifier + task scores for saved outputs
python exp.py robustness         # extra-noise study
python exp.py complexity         # params / GMACs at 256x256
"""
from __future__ import annotations

import json, math, os, sys, time, copy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision, timm

torch.set_num_threads(2)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from degradation import gaussian_kernel, motion_kernel, CRF_A, CRF_B
from dax_net import AuxBranch, CrossAttention, UpBlock, SE
from baselines import SRCNN, UNet, RRDBNet, DCENet

TILES = os.path.join(HERE, "tiles")
WTS = os.environ.get("DAX_WEIGHTS", os.path.join(HERE, "weights"))
RES = os.path.join(HERE, "results"); os.makedirs(RES, exist_ok=True)
CK = os.path.join(HERE, "ckpt"); os.makedirs(CK, exist_ok=True)
OUTS = os.path.join(HERE, "outputs"); os.makedirs(OUTS, exist_ok=True)
CROP, BATCH, ITERS = 96, 8, 3000
EVAL_EVERY, NVAL = 250, 100
RARE = [5, 7, 15, 16, 17, 18]          # water, pool, person, dog, car, bicycle


# ============================================================== data
def load(split, kind="img"):
    return np.load(os.path.join(TILES, f"{split}_{kind}.npy"), mmap_mode="r")


def sample_params(rng, lowlight=True):
    return {"k": float(rng.uniform(0.15, 0.6)) if (lowlight and rng.random() < 0.8) else 1.0,
            "sigma_g": float(rng.uniform(0.5, 2.5)) if rng.random() < 0.5 else 0.0,
            "L": int(rng.integers(7, 22)) if rng.random() < 0.5 else 0,
            "theta": float(rng.uniform(-np.pi, np.pi)),
            "sigma_n": float(rng.uniform(0.0, 0.03)) if rng.random() < 0.8 else 0.0}


def _conv(x, k):
    k = torch.from_numpy(k).to(x)[None, None].repeat(3, 1, 1, 1)
    p = k.shape[-1] // 2
    return F.conv2d(F.pad(x, (p, p, p, p), mode="reflect"), k, groups=3)


def degrade_one(x, p, gen):
    """x: (3,H,W) in [0,1]; Eq. (10): y = g(h_m * h_g * x, k) + n, then 8-bit quantisation."""
    y = x[None]
    if p["sigma_g"] > 0:
        y = _conv(y, gaussian_kernel(p["sigma_g"]))
    if p["L"] > 0:
        y = _conv(y, motion_kernel(p["L"], p["theta"]))
    if p["k"] != 1.0:
        gamma = p["k"] ** CRF_A; beta = math.exp(CRF_B * (1 - gamma))
        y = (beta * y.clamp(0, 1) ** gamma).clamp(0, 1)
    if p["sigma_n"] > 0:
        y = y + torch.randn(y.shape, generator=gen) * p["sigma_n"]
    return (torch.round(y.clamp(0, 1) * 255) / 255)[0]


def prepare():
    for split, seed in (("val", 101), ("test", 202)):
        img = load(split); n = len(img)
        out = np.zeros_like(np.asarray(img)); params = []
        for i in range(n):
            rng = np.random.default_rng(seed * 100000 + i); gen = torch.Generator().manual_seed(seed * 100000 + i)
            p = sample_params(rng); params.append(p)
            x = torch.from_numpy(np.asarray(img[i])).permute(2, 0, 1).float() / 255
            out[i] = (degrade_one(x, p, gen).permute(1, 2, 0).numpy() * 255 + 0.5).astype(np.uint8)
        np.save(os.path.join(TILES, f"{split}_deg.npy"), out)
        json.dump(params, open(os.path.join(TILES, f"{split}_deg_params.json"), "w"))
        print(split, n, flush=True)


class Sampler:
    def __init__(self, seed=0, lowlight=True):
        self.img, self.mask = load("train"), load("train", "mask")
        self.rng = np.random.default_rng(seed); self.gen = torch.Generator().manual_seed(seed)
        self.lowlight = lowlight
        cache = os.path.join(TILES, "train_rare_idx.npy")
        if not os.path.exists(cache):
            m = self.mask
            idx = [i for i in range(len(m)) if np.isin(np.asarray(m[i]), RARE).any()]
            np.save(cache, np.array(idx))
        self.rare = np.load(cache)

    def batch(self, bs=BATCH, crop=CROP, degrade=True):
        xs, ys = [], []
        for _ in range(bs):
            if self.rng.random() < 0.8:
                i = int(self.rng.choice(self.rare)); m = np.asarray(self.mask[i])
                yy, xx = np.nonzero(np.isin(m, RARE)); j = self.rng.integers(len(yy))
                y0 = int(np.clip(yy[j] - crop // 2, 0, 256 - crop)); x0 = int(np.clip(xx[j] - crop // 2, 0, 256 - crop))
            else:
                i = int(self.rng.integers(len(self.img))); y0, x0 = self.rng.integers(0, 256 - crop + 1, 2)
            t = np.asarray(self.img[i, y0:y0 + crop, x0:x0 + crop])
            if self.rng.random() < 0.5: t = t[:, ::-1]
            t = np.rot90(t, int(self.rng.integers(4)))
            x = torch.from_numpy(np.ascontiguousarray(t)).permute(2, 0, 1).float() / 255
            xs.append(x)
            ys.append(degrade_one(x, sample_params(self.rng, self.lowlight), self.gen) if degrade else x)
        return torch.stack(ys), torch.stack(xs)


# ============================================================== models
class Encoder(nn.Module):
    TIMM = {"resnet18": "resnet18_a1_0-d63eafa0.pth", "resnet50": "resnet50_a1_0-14fe96d1.pth",
            "mobilenetv2": "mobilenetv2_100_ra-b33bc2c4.pth"}

    def __init__(self, name="squeezenet"):
        super().__init__()
        self.name = name
        if name == "squeezenet":
            net = torchvision.models.squeezenet1_1()
            net.load_state_dict(torch.load(os.path.join(WTS, "squeezenet1_1_imagenet.pth")))
            self.body, self.taps, self.channels = net.features[:13], [1, 4, 7, 12], [64, 128, 256, 512]
        else:
            tname = "mobilenetv2_100" if name == "mobilenetv2" else name
            self.body = timm.create_model(tname, pretrained=True, features_only=True, out_indices=(0, 1, 2, 3),
                                          pretrained_cfg_overlay=dict(file=os.path.join(WTS, self.TIMM[name])))
            self.channels = self.body.feature_info.channels()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        if self.name != "squeezenet":
            return self.body(x)
        feats = []
        for i, layer in enumerate(self.body):
            x = layer(x)
            if i in self.taps: feats.append(x)
        return feats


class DAXNet(nn.Module):
    def __init__(self, backbone="squeezenet", d=128, heads=4, use_ca=True, use_aux=True, use_se=True):
        super().__init__()
        self.enc = Encoder(backbone)
        c1, c2, c3, c4 = self.enc.channels
        self.use_aux, self.use_ca = use_aux, use_ca and use_aux
        self.proj_vis = nn.Conv2d(c4, d, 1)
        if use_aux:
            self.aux = AuxBranch(d)
            if self.use_ca:
                self.ca_va, self.ca_av = CrossAttention(d, heads), CrossAttention(d, heads)
        self.fuse = nn.Conv2d(2 * d if use_aux else d, d, 1)
        self.up3, self.up2, self.up1 = UpBlock(d, c3, 128, use_se), UpBlock(128, c2, 64, use_se), UpBlock(64, c1, 32, use_se)
        self.head = nn.Sequential(nn.Conv2d(35, 32, 3, 1, 1), nn.LeakyReLU(0.2, True), nn.Conv2d(32, 3, 3, 1, 1))

    def forward(self, y):
        f1, f2, f3, f4 = self.enc(y)
        xv = self.proj_vis(f4); b, d, h, w = xv.shape
        if self.use_aux:
            xa = F.interpolate(self.aux(y), size=(h, w), mode="bilinear", align_corners=False)
            tv, ta = xv.flatten(2).transpose(1, 2), xa.flatten(2).transpose(1, 2)
            if self.use_ca:
                tv, ta = tv + self.ca_va(tv, ta), ta + self.ca_av(ta, tv)
            z = torch.cat([tv, ta], -1).transpose(1, 2).reshape(b, 2 * d, h, w)
        else:
            z = xv
        z = self.up1(self.up2(self.up3(self.fuse(z), f3), f2), f1)
        z = F.interpolate(z, size=y.shape[-2:], mode="bilinear", align_corners=False)
        return torch.clamp(y + self.head(torch.cat([z, y], 1)), 0, 1)


def nafnet_compact():
    from NAFNet_arch import NAFNet
    return NAFNet(img_channel=3, width=16, middle_blk_num=2, enc_blk_nums=[1, 1, 1, 2], dec_blk_nums=[1, 1, 1, 1])


def restormer_compact():
    from restormer_arch import Restormer
    return Restormer(dim=24, num_blocks=[1, 2, 2, 3], num_refinement_blocks=1, heads=[1, 2, 4, 8], ffn_expansion_factor=2.0)


class Clamp(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x): return torch.clamp(self.m(x), 0, 1)


def build(cfg):
    a = cfg["arch"]
    if a == "dax":
        return DAXNet(cfg.get("backbone", "squeezenet"), use_ca=cfg.get("ca", True), use_aux=cfg.get("aux", True), use_se=cfg.get("se", True))
    if a == "srcnn": return SRCNN()
    if a == "unet": return UNet(base=32)
    if a == "rrdb": return RRDBNet(nf=32, nb=6, gc=16, unshuffle=2)
    if a == "dce": return DCENet()
    if a == "nafnet": return Clamp(nafnet_compact())
    if a == "restormer": return Clamp(restormer_compact())
    raise ValueError(a)


# ============================================================== losses
def ssim_t(a, b):
    c = 3; ax = torch.arange(11, dtype=a.dtype) - 5; g = torch.exp(-ax ** 2 / 4.5); g = g / g.sum()
    w = (g[:, None] @ g[None, :]).expand(c, 1, 11, 11).contiguous()
    mu_a, mu_b = F.conv2d(a, w, padding=5, groups=c), F.conv2d(b, w, padding=5, groups=c)
    saa = F.conv2d(a * a, w, padding=5, groups=c) - mu_a ** 2
    sbb = F.conv2d(b * b, w, padding=5, groups=c) - mu_b ** 2
    sab = F.conv2d(a * b, w, padding=5, groups=c) - mu_a * mu_b
    return (((2 * mu_a * mu_b + 1e-4) * (2 * sab + 9e-4)) / ((mu_a ** 2 + mu_b ** 2 + 1e-4) * (saa + sbb + 9e-4))).mean()


class VGGLoss(nn.Module):
    """Squared feature distance at VGG-16 relu1_2 and relu2_2 (ImageNet weights), Eq. (19)."""
    def __init__(self):
        super().__init__()
        v = torchvision.models.vgg16(); v.load_state_dict(torch.load(os.path.join(WTS, "vgg16_imagenet.pth")))
        self.s1, self.s2 = v.features[:4].eval(), v.features[4:9].eval()
        for p in self.parameters(): p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, a, b):
        a, b = (a - self.mean) / self.std, (b - self.mean) / self.std
        a1, b1 = self.s1(a), self.s1(b)
        return F.mse_loss(a1, b1) + F.mse_loss(self.s2(a1), self.s2(b1))


def zero_dce_loss(y, out, A):
    """Non-reference losses of Zero-DCE (spatial consistency, exposure, colour constancy, illumination smoothness)."""
    def spa(org, enh):
        org, enh = org.mean(1, keepdim=True), enh.mean(1, keepdim=True)
        po, pe = F.avg_pool2d(org, 4), F.avg_pool2d(enh, 4)
        ks = [torch.tensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]), torch.tensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]),
              torch.tensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]), torch.tensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]])]
        s = 0
        for k in ks:
            k = k.float()[None, None]
            s = s + (F.conv2d(po, k, padding=1) - F.conv2d(pe, k, padding=1)) ** 2
        return s.mean()
    exp_l = (F.avg_pool2d(out.mean(1, keepdim=True), 16) - 0.6).pow(2).mean()
    m = out.mean((2, 3)); col = ((m[:, 0] - m[:, 1]) ** 2 + (m[:, 0] - m[:, 2]) ** 2 + (m[:, 1] - m[:, 2]) ** 2).mean()
    tv = ((A[:, :, 1:, :] - A[:, :, :-1, :]) ** 2).mean() + ((A[:, :, :, 1:] - A[:, :, :, :-1]) ** 2).mean()
    return 200 * tv + spa(y, out) + 5 * col + 10 * exp_l


class DCEWrap(DCENet):
    def forward_with_curves(self, y):
        x1 = F.relu(self.e1(y)); x2 = F.relu(self.e2(x1)); x3 = F.relu(self.e3(x2)); x4 = F.relu(self.e4(x3))
        x5 = F.relu(self.e5(torch.cat([x3, x4], 1))); x6 = F.relu(self.e6(torch.cat([x2, x5], 1)))
        r = torch.tanh(self.e7(torch.cat([x1, x6], 1))); x = y
        for a in torch.split(r, 3, dim=1):
            x = x + a * (x * x - x)
        return x.clamp(0, 1), r


class Disc(nn.Module):
    def __init__(self):
        super().__init__()
        def blk(i, o, s): return [nn.Conv2d(i, o, 3 if s == 1 else 4, s, 1), nn.LeakyReLU(0.2, True)]
        self.net = nn.Sequential(*blk(3, 32, 1), *blk(32, 32, 2), *blk(32, 64, 1), *blk(64, 64, 2), *blk(64, 128, 1), *blk(128, 128, 2))
        self.fc = nn.Linear(128, 1)
    def forward(self, x): return self.fc(self.net(x).mean((2, 3)))


# ============================================================== configs
CONFIGS = {
    # DAX-Net default: MobileNetV2 encoder (selected after the encoder study, Table 3)
    "daxnet":         {"arch": "dax", "backbone": "mobilenetv2", "loss": "dax"},
    "enc_squeezenet": {"arch": "dax", "backbone": "squeezenet", "loss": "dax"},
    "enc_resnet18":   {"arch": "dax", "backbone": "resnet18", "loss": "dax"},
    "enc_resnet50":   {"arch": "dax", "backbone": "resnet50", "loss": "dax"},
    "enc_mobilenet":  {"arch": "dax", "backbone": "mobilenetv2", "loss": "dax"},
    "abl_noca":       {"arch": "dax", "backbone": "mobilenetv2", "ca": False, "loss": "dax"},
    "abl_noaux":      {"arch": "dax", "backbone": "mobilenetv2", "aux": False, "loss": "dax"},
    "abl_nose":       {"arch": "dax", "backbone": "mobilenetv2", "se": False, "loss": "dax"},
    "abl_novgg":      {"arch": "dax", "backbone": "mobilenetv2", "loss": "dax", "l3": 0.0},
    "abl_nossim":     {"arch": "dax", "backbone": "mobilenetv2", "loss": "dax", "l2": 0.0},
    "abl_nolowlight": {"arch": "dax", "backbone": "mobilenetv2", "loss": "dax", "lowlight": False},
    "abl_nodoa":      {"arch": "dax", "backbone": "mobilenetv2", "loss": "dax", "l2": 1.0, "l3": 1.0},
    "srcnn":          {"arch": "srcnn", "loss": "mse", "lr": 1e-4},
    "unet":           {"arch": "unet", "loss": "l1", "lr": 2e-4},
    "esrgan":         {"arch": "rrdb", "loss": "esrgan", "lr": 2e-4, "select": "final"},
    "zerodce":        {"arch": "dce", "loss": "zerodce", "lr": 1e-4},
    "nafnet":         {"arch": "nafnet", "loss": "psnr", "lr": 1e-3},
    "restormer":      {"arch": "restormer", "loss": "l1", "lr": 3e-4},
}


def doa_params():
    p = os.path.join(RES, "doa.json")
    if os.path.exists(p):
        d = json.load(open(p)); return d["best"]["l2"], d["best"]["l3"], d["best"]["eta0"]
    return 1.0, 0.1, 1e-3


# ============================================================== training
def psnr_np(a, b):
    m = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)); return 10 * math.log10(1.0 / max(m, 1e-12))


@torch.no_grad()
def run_model(model, deg, bs=4):
    model.eval(); outs = []
    for i in range(0, len(deg), bs):
        y = torch.from_numpy(np.asarray(deg[i:i + bs])).permute(0, 3, 1, 2).float() / 255
        outs.append((model(y).permute(0, 2, 3, 1).numpy() * 255 + 0.5).clip(0, 255).astype(np.uint8))
    return np.concatenate(outs)


def quick_val(model, n=NVAL):
    deg, clean = load("val", "deg"), load("val")
    idx = np.linspace(0, len(deg) - 1, n).astype(int)
    out = run_model(model, deg[idx])
    from skimage.metrics import structural_similarity as ssim
    ps = [psnr_np(o / 255, np.asarray(clean[i]) / 255) for o, i in zip(out, idx)]
    ss = [ssim(o, np.asarray(clean[i]), channel_axis=-1, data_range=255, gaussian_weights=True, sigma=1.5, use_sample_covariance=False) for o, i in zip(out, idx)]
    g = gmsd_batch(out, np.asarray(clean[idx]))
    return float(np.mean(ps)), float(np.mean(ss)), float(np.mean(g))


_GMSD = None
def gmsd_batch(out_u8, ref_u8):
    global _GMSD
    import pyiqa
    if _GMSD is None: _GMSD = pyiqa.create_metric("gmsd", device="cpu")
    vals = []
    for i in range(0, len(out_u8), 16):
        a = torch.from_numpy(out_u8[i:i + 16]).permute(0, 3, 1, 2).float() / 255
        b = torch.from_numpy(ref_u8[i:i + 16]).permute(0, 3, 1, 2).float() / 255
        vals.append(_GMSD(a, b).view(-1).numpy())
    return np.concatenate(vals)


def train(name, iters=ITERS, l2=None, l3=None, eta0=None, seed=0, save=True, log_curve=True, nval=NVAL):
    cfg = CONFIGS[name] if name in CONFIGS else {"arch": "dax", "loss": "dax"}
    torch.manual_seed(seed)
    model = build(cfg) if cfg["arch"] != "dce" else DCEWrap()
    d2, d3, de = doa_params()
    l2 = cfg.get("l2", d2) if l2 is None else l2
    l3 = cfg.get("l3", d3) if l3 is None else l3
    lr = cfg.get("lr", de) if eta0 is None else eta0
    sampler = Sampler(seed=seed, lowlight=cfg.get("lowlight", True))
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=lr * 0.01)
    vgg = VGGLoss() if cfg["loss"] in ("dax", "esrgan") and (l3 > 0 or cfg["loss"] == "esrgan") else None
    disc = Disc() if cfg["loss"] == "esrgan" else None
    dopt = torch.optim.Adam(disc.parameters(), lr=lr, betas=(0.9, 0.99)) if disc else None
    curve, best, t0 = [], (-1, None), time.time()
    for it in range(1, iters + 1):
        model.train()
        y, x = sampler.batch()
        if cfg["loss"] == "zerodce":
            out, A = model.forward_with_curves(y); loss = zero_dce_loss(y, out, A)
        else:
            out = model(y)
            if cfg["loss"] == "dax":
                loss = F.l1_loss(out, x)
                if l2 > 0: loss = loss + l2 * (1 - ssim_t(out, x))
                if l3 > 0: loss = loss + l3 * vgg(out, x)
            elif cfg["loss"] == "mse": loss = F.mse_loss(out, x)
            elif cfg["loss"] == "l1": loss = F.l1_loss(out, x)
            elif cfg["loss"] == "psnr": loss = 10 * torch.log10(((out - x) ** 2).mean((1, 2, 3)) + 1e-8).mean()
            elif cfg["loss"] == "esrgan":
                if it <= iters // 2:            # PSNR-oriented warm-up, as in ESRGAN
                    loss = F.l1_loss(out, x)
                else:                           # relativistic GAN stage
                    dr, df = disc(x), disc(out.detach())
                    dloss = (F.binary_cross_entropy_with_logits(dr - df.mean(), torch.ones_like(dr)) +
                             F.binary_cross_entropy_with_logits(df - dr.mean(), torch.zeros_like(df))) / 2
                    dopt.zero_grad(); dloss.backward(); dopt.step()
                    dr, df = disc(x).detach(), disc(out)
                    adv = (F.binary_cross_entropy_with_logits(dr - df.mean(), torch.zeros_like(dr)) +
                           F.binary_cross_entropy_with_logits(df - dr.mean(), torch.ones_like(df))) / 2
                    loss = vgg(out, x) + 5e-3 * adv + 1e-2 * F.l1_loss(out, x)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if log_curve and (it % EVAL_EVERY == 0 or it == iters):
            p, s, g = quick_val(model, nval)
            curve.append({"iter": it, "loss": float(loss.detach()), "val_psnr": p, "val_ssim": s, "val_gmsd": g, "time_s": time.time() - t0})
            print(name, curve[-1], flush=True)
            if p > best[0]:
                best = (p, copy.deepcopy(model.state_dict()))
    if not log_curve:
        return model
    if cfg.get("select") == "final":     # adversarial models are not selected on PSNR: keep the final generator
        best = (curve[-1]["val_psnr"], copy.deepcopy(model.state_dict()))
    if save:
        torch.save(best[1], os.path.join(CK, f"{name}.pt"))
        json.dump({"name": name, "cfg": cfg, "l2": l2, "l3": l3, "lr": lr, "iters": iters, "curve": curve,
                   "best_val_psnr": best[0], "train_time_s": time.time() - t0}, open(os.path.join(RES, f"train_{name}.json"), "w"), indent=1)
    return model


# ============================================================== DOA
def doa():
    from doa import doa_minimise
    lower, upper = [0.1, 0.01, -4.0], [2.0, 0.5, -2.5]
    hist = []
    def obj(z):
        l2, l3, eta0 = float(z[0]), float(z[1]), float(10 ** z[2])
        m = train("doa_proxy", iters=200, l2=l2, l3=l3, eta0=eta0, seed=7, save=False, log_curve=False)
        p, s, g = quick_val(m, 100)
        f = p / 40 + s - g
        hist.append({"l2": l2, "l3": l3, "eta0": eta0, "psnr": p, "ssim": s, "gmsd": g, "fitness": f})
        print("DOA", hist[-1], flush=True)
        json.dump({"history": hist}, open(os.path.join(RES, "doa_partial.json"), "w"), indent=1)
        return -f
    best, bf, curve = doa_minimise(obj, lower, upper, n_pop=4, n_iter=2, seed=0)
    res = {"best": {"l2": float(best[0]), "l3": float(best[1]), "eta0": float(10 ** best[2]), "fitness": -float(bf)},
           "history": hist, "best_per_generation": [-float(c) for c in curve], "population": 4, "generations": 2, "proxy_iters": 200}
    json.dump(res, open(os.path.join(RES, "doa.json"), "w"), indent=1); print(res["best"])


# ============================================================== evaluation
def clahe_np(deg):
    import cv2
    out = np.empty_like(deg); cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for i in range(len(deg)):
        lab = cv2.cvtColor(np.asarray(deg[i]), cv2.COLOR_RGB2LAB); lab[..., 0] = cl.apply(lab[..., 0])
        out[i] = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return out


def quality(out, clean, niqe=True):
    from skimage.metrics import structural_similarity as ssim
    ps = np.array([psnr_np(o / 255, c / 255) for o, c in zip(out, clean)])
    ss = np.array([ssim(o, c, channel_axis=-1, data_range=255, gaussian_weights=True, sigma=1.5, use_sample_covariance=False) for o, c in zip(out, clean)])
    g = gmsd_batch(out, clean)
    r = {"psnr": float(ps.mean()), "ssim": float(ss.mean()), "gmsd": float(g.mean()),
         "psnr_std": float(ps.std()), "psnr_per_tile": ps.round(3).tolist(), "ssim_per_tile": ss.round(4).tolist(),
         "gmsd_per_tile": g.round(5).tolist()}
    if niqe:
        import pyiqa
        nq = pyiqa.create_metric("niqe", device="cpu"); vals = []
        sub = out[::3]                                   # every third test tile (700 tiles)
        for i in range(0, len(sub), 8):
            vals.append(nq(torch.from_numpy(np.ascontiguousarray(sub[i:i + 8])).permute(0, 3, 1, 2).float() / 255).view(-1).numpy())
        r["niqe"] = float(np.concatenate(vals).mean())
    return r


def evaluate(name, full=True):
    deg, clean = load("test", "deg"), np.asarray(load("test"))
    t0 = time.time()
    if name == "input": out = np.asarray(deg)
    elif name == "clean": out = clean
    elif name == "clahe": out = clahe_np(np.asarray(deg))
    else:
        cfg = CONFIGS[name]; model = build(cfg) if cfg["arch"] != "dce" else DCEWrap()
        model.load_state_dict(torch.load(os.path.join(CK, f"{name}.pt"))); out = run_model(model, deg)
    secs = (time.time() - t0) / len(deg)
    r = quality(out, clean, niqe=full) if name != "clean" else {"niqe": quality(out, clean)["niqe"]}
    r["sec_per_tile_cpu"] = secs
    if full:
        np.save(os.path.join(OUTS, f"{name}.npy"), out)
    json.dump(r, open(os.path.join(RES, f"eval_{name}.json"), "w"))
    print(name, {k: v for k, v in r.items() if not k.endswith("per_tile")}, flush=True)


# ============================================================== downstream
GROUPS = {"vegetation": [3, 8, 19, 20], "bare": [2, 4, 6], "water": [5, 7], "paved": [1], "built": [9, 10, 11, 12, 13, 14]}
OBJ = [15, 16, 17, 18]


def task_labels(masks):
    L = {"landcover": [], "vegetation": [], "water": [], "object": []}
    for m in masks:
        m = np.asarray(m); n = m.size
        fr = [np.isin(m, c).sum() / n for c in GROUPS.values()]
        L["landcover"].append(int(np.argmax(fr))); L["vegetation"].append(int(fr[0] >= 0.05))
        L["water"].append(int(fr[2] >= 0.01)); L["object"].append(int(np.isin(m, OBJ).sum() / n >= 0.002))
    return {k: np.array(v) for k, v in L.items()}


class Trunk(nn.Module):
    def __init__(self):
        super().__init__()
        net = torchvision.models.squeezenet1_1(); net.load_state_dict(torch.load(os.path.join(WTS, "squeezenet1_1_imagenet.pth")))
        self.f = net.features.eval()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, x):
        x = (x - self.mean) / self.std; outs = []
        for i, l in enumerate(self.f):
            x = l(x)
            if i in (4, 7, 12): outs.append(x.mean((2, 3)))
        return torch.cat(outs, 1)


def feats(trunk, arr):
    out = []
    for i in range(0, len(arr), 16):
        out.append(trunk(torch.from_numpy(np.asarray(arr[i:i + 16])).permute(0, 3, 1, 2).float() / 255))
    return torch.cat(out)


def downstream():
    from metrics import macro_scores
    trunk = Trunk()
    cache = os.path.join(RES, "feat_cache.pt")
    if os.path.exists(cache): C = torch.load(cache)
    else:
        C = {"train": feats(trunk, load("train")), "val": feats(trunk, load("val"))}; torch.save(C, cache)
    Ltr, Lva, Lte = task_labels(load("train", "mask")), task_labels(load("val", "mask")), task_labels(load("test", "mask"))
    ncls = {"landcover": 5, "vegetation": 2, "water": 2, "object": 2}
    mu, sd = C["train"].mean(0), C["train"].std(0) + 1e-6
    heads = {}
    for t, k in ncls.items():
        torch.manual_seed(0)
        h = nn.Sequential(nn.Linear(C["train"].shape[1], 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, k))
        opt = torch.optim.Adam(h.parameters(), 1e-3, weight_decay=1e-4)
        Xtr, ytr = (C["train"] - mu) / sd, torch.from_numpy(Ltr[t]); Xva, yva = (C["val"] - mu) / sd, torch.from_numpy(Lva[t])
        w = torch.bincount(ytr, minlength=k).float(); w = (w.sum() / (k * w.clamp(min=1)))
        best = (-1, None, 0); wait = 0
        for ep in range(60):
            h.train(); perm = torch.randperm(len(Xtr))
            for i in range(0, len(Xtr), 128):
                b = perm[i:i + 128]; loss = F.cross_entropy(h(Xtr[b]), ytr[b], weight=w)
                opt.zero_grad(); loss.backward(); opt.step()
            h.eval(); acc = (h(Xva).argmax(1) == yva).float().mean().item()
            if acc > best[0]: best, wait = (acc, copy.deepcopy(h.state_dict()), ep), 0
            else:
                wait += 1
                if wait >= 8: break
        h.load_state_dict(best[1]); h.eval(); heads[t] = (h, best[2], best[0])
    res = {"label_stats": {t: np.bincount(Lte[t], minlength=ncls[t]).tolist() for t in ncls},
           "heads": {t: {"best_epoch": heads[t][1], "val_acc": heads[t][2]} for t in ncls}, "methods": {}}
    for f in sorted(os.listdir(OUTS)):
        name = f[:-4]; X = (feats(trunk, np.load(os.path.join(OUTS, f), mmap_mode="r")) - mu) / sd
        r = {}
        for t, k in ncls.items():
            pred = heads[t][0](X).argmax(1).numpy(); s = macro_scores(Lte[t], pred, k)
            r[t] = {"acc": float(s["acc"]), "precision": float(s["precision"]), "recall": float(s["recall"]), "f1": float(s["f1"])}
        r["mean_acc"] = float(np.mean([r[t]["acc"] for t in ncls])); r["mean_f1"] = float(np.mean([r[t]["f1"] for t in ncls]))
        r["mean_precision"] = float(np.mean([r[t]["precision"] for t in ncls])); r["mean_recall"] = float(np.mean([r[t]["recall"] for t in ncls]))
        res["methods"][name] = r; print(name, round(r["mean_acc"], 4), round(r["mean_f1"], 4), flush=True)
    json.dump(res, open(os.path.join(RES, "downstream.json"), "w"), indent=1)


# ============================================================== robustness
def robustness(names=("srcnn", "daxnet"), out_name="robustness.json"):
    from degradation import extra_noise
    deg, clean = np.asarray(load("test", "deg")), np.asarray(load("test"))
    settings = [("gaussian", 0.01), ("gaussian", 0.05), ("salt_pepper", 0.02), ("speckle", 0.03)]
    res = {}
    for name in names:
        model = build(CONFIGS[name]); model.load_state_dict(torch.load(os.path.join(CK, f"{name}.pt")))
        for kind, lvl in settings:
            rng = np.random.default_rng(5)
            noisy = np.stack([(extra_noise(d / 255.0, kind, lvl, rng) * 255 + 0.5).astype(np.uint8) for d in deg])
            out = run_model(model, noisy); q = quality(out, clean, niqe=False)
            res[f"{name}|{kind}|{lvl}"] = {"psnr": q["psnr"], "ssim": q["ssim"], "gmsd": q["gmsd"]}
            print(name, kind, lvl, res[f"{name}|{kind}|{lvl}"], flush=True)
    json.dump(res, open(os.path.join(RES, out_name), "w"), indent=1)


# ============================================================== complexity
def complexity():
    from torch.utils.flop_counter import FlopCounterMode
    res = {}
    for name, cfg in CONFIGS.items():
        if name.startswith("abl_"): continue
        m = build(cfg).eval(); x = torch.rand(1, 3, 256, 256)
        with torch.no_grad(), FlopCounterMode(display=False) as fc: m(x)
        res[name] = {"params_M": sum(p.numel() for p in m.parameters()) / 1e6, "gmacs_256": fc.get_total_flops() / 2e9}
        print(name, res[name], flush=True)
    json.dump(res, open(os.path.join(RES, "complexity.json"), "w"), indent=1)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prepare": prepare()
    elif cmd == "doa": doa()
    elif cmd == "train": train(sys.argv[2])
    elif cmd == "evaluate": evaluate(sys.argv[2], full=(len(sys.argv) < 4 or sys.argv[3] != "lite"))
    elif cmd == "downstream": downstream()
    elif cmd == "robustness": robustness()
    elif cmd == "robustness_final": robustness(("abl_novgg",), "robustness_final.json")
    elif cmd == "complexity": complexity()
