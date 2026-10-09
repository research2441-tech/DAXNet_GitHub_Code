"""Physically motivated degradation synthesis for UAV tiles (Sect. 3.2 of the paper).

All images are float32 arrays in [0, 1] with shape (H, W, 3).
The composite model is  y = g(k_m * k_g * x, k) + n   (Eq. 10).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve

# Beta-gamma camera response model (Ying et al., ICCVW 2017), Eqs. (3)-(7)
CRF_A = -0.3293
CRF_B = 1.1258


def crf(E: np.ndarray, a: float = CRF_A, b: float = CRF_B) -> np.ndarray:
    """Camera response f(E) = exp(b (1 - E^a)); maps irradiance to pixel value."""
    E = np.clip(E, 1e-6, None)
    return np.exp(b * (1.0 - np.power(E, a)))


def btf(V: np.ndarray, k: float, a: float = CRF_A, b: float = CRF_B) -> np.ndarray:
    """Brightness transfer g(V, k) = beta * V^gamma with gamma = k^a, beta = exp(b (1 - k^a)).

    k < 1 darkens (simulated under-exposure), k > 1 brightens.
    """
    gamma = k ** a
    beta = np.exp(b * (1.0 - gamma))
    return np.clip(beta * np.power(np.clip(V, 0.0, 1.0), gamma), 0.0, 1.0)


def gaussian_kernel(sigma: float) -> np.ndarray:
    """Normalised isotropic Gaussian defocus kernel, Eq. (8)."""
    r = int(np.ceil(3 * sigma))
    ax = np.arange(-r, r + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
    return (k / k.sum()).astype(np.float32)


def motion_kernel(length: int, theta: float) -> np.ndarray:
    """Linear motion-blur kernel of length L along direction theta, Eq. (9)."""
    r = length // 2 + 1
    ax = np.arange(-r, r + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    along = xx * np.cos(theta) + yy * np.sin(theta)
    across = -xx * np.sin(theta) + yy * np.cos(theta)
    k = ((np.abs(across) <= 0.5) & (np.abs(along) <= length / 2.0)).astype(np.float64)
    if k.sum() == 0:
        k[r, r] = 1.0
    return (k / k.sum()).astype(np.float32)


def blur(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return np.stack([convolve(img[..., c], kernel, mode="reflect") for c in range(img.shape[-1])], -1)


def sample_params(rng: np.random.Generator) -> dict:
    """Random degradation parameters (ranges listed in Table S2 of Online Resource 1)."""
    return {
        "k": float(rng.uniform(0.15, 0.6)) if rng.random() < 0.8 else 1.0,
        "sigma_g": float(rng.uniform(0.5, 2.5)) if rng.random() < 0.5 else 0.0,
        "L": int(rng.integers(7, 22)) if rng.random() < 0.5 else 0,
        "theta": float(rng.uniform(-np.pi, np.pi)),
        "sigma_n": float(rng.uniform(0.0, 0.03)) if rng.random() < 0.8 else 0.0,
    }


def degrade(x: np.ndarray, p: dict, rng: np.random.Generator) -> np.ndarray:
    """Composite degradation y = g(k_m * k_g * x, k) + n (Eq. 10)."""
    y = x.astype(np.float32)
    if p.get("sigma_g", 0) > 0:
        y = blur(y, gaussian_kernel(p["sigma_g"]))
    if p.get("L", 0) > 0:
        y = blur(y, motion_kernel(p["L"], p["theta"]))
    if p.get("k", 1.0) != 1.0:
        y = btf(y, p["k"])
    if p.get("sigma_n", 0) > 0:
        y = y + rng.normal(0.0, p["sigma_n"], size=y.shape).astype(np.float32)
    return np.clip(y, 0.0, 1.0)


def extra_noise(y: np.ndarray, kind: str, level: float, rng: np.random.Generator) -> np.ndarray:
    """Additional test-time noise used for the robustness study (Table S4)."""
    if kind == "gaussian":
        z = y + rng.normal(0, level, y.shape)
    elif kind == "salt_pepper":
        z = y.copy()
        m = rng.random(y.shape[:2])
        z[m < level / 2] = 0.0
        z[m > 1 - level / 2] = 1.0
    elif kind == "speckle":
        z = y + y * rng.normal(0, np.sqrt(level), y.shape)
    else:
        raise ValueError(kind)
    return np.clip(z, 0, 1).astype(np.float32)
