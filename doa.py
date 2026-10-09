"""Dingo Optimization Algorithm (Peraza-Vazquez et al., 2021) used to tune the loss weights
and the initial learning rate on the validation split (Sect. 3.4, Eq. 21; update rules Eqs. S1-S4 of Online Resource 1).

The search maximises the validation fitness F = PSNR/40 + SSIM - LPIPS of a short proxy run;
here it is written as minimisation of -F.
"""
from __future__ import annotations

import numpy as np


def doa_minimise(fun, lower, upper, n_pop=8, n_iter=10, p_hunt=0.5, q_group=0.7, seed=0, verbose=False):
    rng = np.random.default_rng(seed)
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    dim = lower.size
    X = lower + rng.random((n_pop, dim)) * (upper - lower)
    fit = np.array([fun(x) for x in X])
    best = X[fit.argmin()].copy()
    best_f = fit.min()
    history = [best_f]
    for _ in range(n_iter):
        na = int(min(n_pop, rng.integers(2, max(2, n_pop // 2) + 1)))   # size of the attacking pack
        fmin, fmax = fit.min(), fit.max()
        survival = (fmax - fit) / (fmax - fmin + 1e-12)         # Eq. (S4)
        Xn = X.copy()
        for i in range(n_pop):
            b1, b2 = rng.uniform(-2, 2), rng.uniform(-1, 1)
            r1 = rng.choice([j for j in range(n_pop) if j != i])
            if rng.random() < p_hunt:
                if rng.random() < q_group:                          # group attack, Eq. (S1)
                    pack = rng.choice(n_pop, size=na, replace=False)
                    Xn[i] = b1 * (X[pack].mean(0) - X[i]) - best
                else:                                               # persecution, Eq. (S2)
                    Xn[i] = best + b1 * np.exp(b2) * (X[r1] - X[i])
            else:                                                   # scavenging, Eq. (S3)
                sgn = 1 if rng.random() < 0.5 else -1
                Xn[i] = 0.5 * (np.exp(b2) * X[r1] - sgn * X[i])
            if survival[i] <= 0.3 and n_pop >= 3:                   # weak survivors are relocated
                r2 = rng.choice([j for j in range(n_pop) if j not in (i, r1)])
                sgn = 1 if rng.random() < 0.5 else -1
                Xn[i] = best + 0.5 * (X[r1] - sgn * X[r2])
            Xn[i] = np.clip(Xn[i], lower, upper)
        fn = np.array([fun(x) for x in Xn])
        improve = fn < fit
        X[improve], fit[improve] = Xn[improve], fn[improve]
        if fit.min() < best_f:
            best_f, best = fit.min(), X[fit.argmin()].copy()
        history.append(best_f)
        if verbose:
            print(f"best {best_f:.5f} at {best}")
    return best, best_f, history


def decode(x):
    """Search vector -> (lambda_2, lambda_3, eta_0); eta_0 is searched on a log10 scale."""
    return float(x[0]), float(x[1]), float(10 ** x[2])


BOUNDS = ([0.1, 0.01, -4.0], [2.0, 0.5, -2.5])
