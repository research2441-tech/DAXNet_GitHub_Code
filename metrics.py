"""Evaluation metrics: PSNR/MSE (Eq. 22), SSIM, LPIPS, NIQE and the downstream
classification scores (Eq. 23)."""
from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity


def mse(a, b):
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def psnr(a, b, peak=1.0):
    m = mse(a, b)
    return float("inf") if m == 0 else 10.0 * np.log10(peak ** 2 / m)


def ssim(a, b):
    return float(structural_similarity(a, b, channel_axis=-1, data_range=1.0, gaussian_weights=True, sigma=1.5))


def lpips_distance(a, b, net=None):
    """LPIPS (AlexNet) via the `lpips` package; inputs HxWx3 in [0,1]."""
    import torch
    import lpips
    net = net or lpips.LPIPS(net="alex", verbose=False)
    ta = torch.from_numpy(a).permute(2, 0, 1)[None].float() * 2 - 1
    tb = torch.from_numpy(b).permute(2, 0, 1)[None].float() * 2 - 1
    with torch.no_grad():
        return float(net(ta, tb))


def niqe(a):
    """No-reference NIQE via `pyiqa` (optional dependency)."""
    import torch
    import pyiqa
    m = pyiqa.create_metric("niqe", as_loss=False)
    return float(m(torch.from_numpy(a).permute(2, 0, 1)[None].float()))


def confusion(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def macro_scores(y_true, y_pred, n_classes):
    """Accuracy and macro precision / recall / F1 / IoU from the confusion matrix (Eq. 23)."""
    cm = confusion(y_true, y_pred, n_classes)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    pr = tp / np.maximum(tp + fp, 1)
    re = tp / np.maximum(tp + fn, 1)
    f1 = 2 * pr * re / np.maximum(pr + re, 1e-12)
    iou = tp / np.maximum(tp + fp + fn, 1)
    return {"acc": tp.sum() / cm.sum(), "precision": pr.mean(), "recall": re.mean(),
            "f1": f1.mean(), "iou": iou.mean()}
