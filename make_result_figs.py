"""Figures built from the real experiment outputs in results/ and outputs/."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

H = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(H, "results"); O = os.path.join(H, "outputs"); T = os.path.join(H, "tiles")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(H, "figures")
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"], "mathtext.fontset": "stix",
                     "font.size": 8, "axes.linewidth": 0.6, "axes.edgecolor": "#52514e", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "savefig.dpi": 400, "pdf.fonttype": 42})
BLUE, ORANGE, AQUA, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#b9b8b2", "#0b0b0b", "#52514e"
J = lambda n: json.load(open(os.path.join(R, n)))
LABEL = {"input": "Degraded input", "clahe": "CLAHE", "zerodce": "Zero-DCE", "srcnn": "SRCNN", "esrgan": "ESRGAN-C",
         "unet": "U-Net-C", "nafnet": "NAFNet-C", "restormer": "Restormer-C", "daxnet": "DAX-Net", "abl_novgg": "DAX-Net", "clean": "Clean"}


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig_results():
    cx = J("complexity.json"); ds = J("downstream.json")["methods"]; rb = J("summary.json")["robustness"]
    fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.05), gridspec_kw={"wspace": 0.42, "width_ratios": [1.15, 1, 1.15]})
    a = ax[0]
    pts = []
    SM = J("summary.json")["table2"]
    for m in ["srcnn", "zerodce", "unet", "esrgan", "nafnet", "restormer", "daxnet"]:
        pts.append((m, cx[m]["gmacs_256"], SM[m]["psnr"], cx[m]["params_M"]))
    for m, g, p, n in pts:
        col = BLUE if m == "daxnet" else GREY
        a.scatter(g, p, s=14 + 22 * n, color=col, edgecolor="white", linewidth=1.0, zorder=3)
        off = {"esrgan": (-14, 6), "unet": (5, 2), "restormer": (-12, 6)}.get(m, (4, 3))
        a.annotate(LABEL[m], (g, p), textcoords="offset points", xytext=off, fontsize=6, ha="left",
                   color=INK if m == "daxnet" else MUTED, fontweight="bold" if m == "daxnet" else "normal")
    a.set_xscale("log"); a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a.set_xlabel("GMACs per 256×256 tile (log)"); a.set_ylabel("PSNR (dB)")
    a.set_title("(a) Fidelity vs. cost (area ~ parameters)", fontsize=7); a.grid(alpha=0.25, lw=0.4)
    ps = [p for _, _, p, _ in pts]; a.set_ylim(min(ps) - 1.0, max(ps) + 1.0); a.set_xlim(1.0, 32)
    a.set_xticks([1, 3, 10, 30]); a.set_xticklabels(["1", "3", "10", "30"])
    b = ax[1]
    keys = [("gaussian", 0.01), ("gaussian", 0.05), ("salt_pepper", 0.02), ("speckle", 0.03)]
    lab = ["Gauss.\n0.01", "Gauss.\n0.05", "S&P\n0.02", "Speckle\n0.03"]
    base = [rb[f"srcnn|{k}|{l}"]["psnr"] for k, l in keys]; prop = [rb[f"daxnet_final|{k}|{l}"]["psnr"] for k, l in keys]
    xx = np.arange(4); w = 0.36
    b.bar(xx - w / 2 - 0.01, base, w, color=GREY, label="SRCNN", edgecolor="white", linewidth=0.8)
    b.bar(xx + w / 2 + 0.01, prop, w, color=BLUE, label="DAX-Net", edgecolor="white", linewidth=0.8)
    for i in range(4):
        b.text(xx[i] + w / 2, prop[i] + 0.15, f"{prop[i]:.1f}", ha="center", fontsize=5.6, color=INK)
    lo = min(base + prop); b.set_ylim(lo - 2, max(base + prop) + 2.2)
    b.set_xticks(xx); b.set_xticklabels(lab, fontsize=6); b.set_ylabel("PSNR (dB)")
    b.set_title("(b) Robustness to extra test noise", fontsize=7)
    b.legend(frameon=False, fontsize=6, loc="upper center", ncol=2, columnspacing=0.8, handlelength=1.2); b.grid(axis="y", alpha=0.25, lw=0.4)
    c = ax[2]
    tasks = ["landcover", "vegetation", "water", "object"]; tl = ["Land\ncover", "Vege-\ntation", "Water", "Objects"]
    sel = ["input", "nafnet", "restormer", "abl_novgg"]; cols = {"input": "#dcdbd5", "nafnet": GREY, "restormer": AQUA, "abl_novgg": BLUE}
    w = 0.2
    for k, m in enumerate(sel):
        v = [ds[m][t]["f1"] for t in tasks]
        c.bar(np.arange(4) + (k - 1.5) * (w + 0.01), v, w, color=cols[m], label=LABEL[m], edgecolor="white", linewidth=0.6)
    clean = [ds["clean"][t]["f1"] for t in tasks]
    for i in range(4):
        c.plot([i - 0.45, i + 0.45], [clean[i]] * 2, color=INK, lw=0.8, ls=(0, (2, 1.5)))
    c.plot([], [], color=INK, lw=0.8, ls=(0, (2, 1.5)), label="Clean tiles")
    allv = [ds[m][t]["f1"] for m in sel + ["clean"] for t in tasks]
    c.set_ylim(max(0, min(allv) - 0.08), min(1, max(allv) + 0.03))
    c.set_xticks(np.arange(4)); c.set_xticklabels(tl, fontsize=6); c.set_ylabel("macro $F_1$")
    c.set_title("(c) Downstream environmental tasks", fontsize=7)
    c.legend(frameon=False, fontsize=5.6, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2), columnspacing=0.8, handlelength=1.2)
    c.grid(axis="y", alpha=0.25, lw=0.4)
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False); a_.tick_params(labelsize=6, width=0.5, length=2)
    save(fig, "fig5_results")


def fig_qualitative(tiles=None):
    clean = np.load(os.path.join(T, "test_img.npy"), mmap_mode="r")
    params = json.load(open(os.path.join(T, "test_deg_params.json")))
    meths = ["input", "clahe", "zerodce", "unet", "restormer", "abl_novgg", "clean"]
    outs = {m: np.load(os.path.join(O, f"{m}.npy"), mmap_mode="r") for m in meths}
    if tiles is None:
        d = J("eval_abl_novgg.json")["psnr_per_tile"]; inp = J("eval_input.json")["psnr_per_tile"]
        # tiles with strong low light + blur where DAX-Net improves most over the input and that contain objects
        mask = np.load(os.path.join(T, "test_mask.npy"), mmap_mode="r")
        cand = [i for i, p in enumerate(params) if p["k"] < 0.4 and (p["sigma_g"] > 0 or p["L"] > 0)
                and np.isin(np.asarray(mask[i]), [15, 17, 18]).mean() > 0.003]
        cand.sort(key=lambda i: -(d[i] - inp[i])); tiles = [cand[0], cand[len(cand) // 2]]   # largest and median gain
    fig, ax = plt.subplots(len(tiles), len(meths), figsize=(7.1, 1.12 * len(tiles)), gridspec_kw={"wspace": 0.03, "hspace": 0.12})
    for r, t in enumerate(tiles):
        for c, m in enumerate(meths):
            im = np.asarray(outs[m][t]); a = ax[r, c]; a.imshow(im); a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values(): s.set_linewidth(0.3)
            if r == 0: a.set_title(LABEL[m], fontsize=6.4, pad=2)
            if m not in ("clean",):
                e = J(f"eval_{m}.json"); a.text(4, 250, f"{e['psnr_per_tile'][t]:.2f} dB", color="white", fontsize=5.4,
                                                 bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.55, lw=0))
    save(fig, "fig6_qualitative")
    return tiles


def fig_convergence():
    names = ["abl_novgg", "restormer", "nafnet", "unet", "esrgan", "srcnn", "zerodce"]
    cols = {"abl_novgg": BLUE, "restormer": AQUA, "nafnet": ORANGE, "unet": "#4a3aa7", "esrgan": "#e87ba4", "srcnn": "#008300", "zerodce": "#eda100"}
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.1), gridspec_kw={"wspace": 0.3})
    for n in names:
        tr = J(f"train_{n}.json")["curve"]; it = [c["iter"] for c in tr]
        ax[0].plot(it, [c["val_psnr"] for c in tr], color=cols[n], lw=1.2, marker="o", ms=2.5, label=LABEL[n])
        ax[1].plot(it, [c["val_ssim"] for c in tr], color=cols[n], lw=1.2, marker="o", ms=2.5)
    ax[0].set_ylabel("validation PSNR (dB)"); ax[1].set_ylabel("validation SSIM")
    for a in ax:
        a.set_xlabel("training iteration"); a.grid(alpha=0.25, lw=0.4); a.spines[["top", "right"]].set_visible(False); a.tick_params(labelsize=6)
    ax[0].legend(frameon=False, fontsize=5.6, ncol=2)
    ax[0].set_title("(a) PSNR on 100 validation tiles", fontsize=7); ax[1].set_title("(b) SSIM on 100 validation tiles", fontsize=7)
    save(fig, "figS5_convergence")


def fig_ablation(rows=None):
    if rows is None:
        A = J("summary.json")["table4"]
        names = [("full", "Reference (DOA, λ3 = 0.5)"), ("abl_noca", "w/o cross-attention"), ("abl_noaux", "w/o auxiliary branch"),
                 ("abl_nose", "w/o SE gates"), ("abl_novgg", "w/o L_VGG (final model)"), ("abl_nossim", "w/o L_SSIM"),
                 ("abl_nolowlight", "w/o low-light synthesis"), ("abl_nodoa", "w/o DOA (λ2 = λ3 = 1)")]
        rows = [(lab, A[k]["psnr"], A[k]["gmsd"]) for k, lab in names]
    labels = [r[0] for r in rows]; ps = [r[1] for r in rows]; gm = [r[2] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 1.9), gridspec_kw={"wspace": 0.7})
    y = np.arange(len(rows))[::-1]
    for a_, v, lab in [(ax[0], ps, "PSNR (dB), higher is better"), (ax[1], gm, "GMSD, lower is better")]:
        lo, hi = min(v), max(v); pad = (hi - lo) * 0.25 + 1e-6
        cols = [BLUE if i == 0 else GREY for i in range(len(rows))]
        a_.barh(y, [x - (lo - pad) for x in v], left=lo - pad, color=cols, height=0.62, edgecolor="white")
        a_.set_yticks(y); a_.set_yticklabels(labels, fontsize=6); a_.set_xlim(lo - pad, hi + pad); a_.set_xlabel(lab, fontsize=6.5)
        a_.axvline(v[0], color=INK, lw=0.6, ls=(0, (2, 1.5))); a_.spines[["top", "right"]].set_visible(False); a_.tick_params(labelsize=6)
    save(fig, "figS3_ablation")


def fig_doa():
    d = J("doa.json"); h = d["history"]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 1.9), gridspec_kw={"wspace": 0.35})
    f = np.array([x["fitness"] for x in h]); it = np.arange(1, len(f) + 1); ok = f > 0.5
    ax[0].plot(it[ok], f[ok], "o", color=GREY, ms=3, label="evaluated candidate")
    ax[0].plot(it[~ok], np.full((~ok).sum(), 0.976), "v", color=ORANGE, ms=3.5, label="diverged run (F = 0.04, off scale)")
    ax[0].plot(it, np.maximum.accumulate(f), color=BLUE, lw=1.2, label="best so far")
    ax[0].set_ylim(0.972, 1.032)
    ax[0].set_xlabel("fitness evaluation"); ax[0].set_ylabel("fitness F"); ax[0].legend(frameon=False, fontsize=5.5, loc="lower right", ncol=1)
    ax[0].set_title("(a) DOA search trace", fontsize=7)
    hv = [x for x in h if x["fitness"] > 0.5]
    sc = ax[1].scatter([x["l2"] for x in hv], [x["l3"] for x in hv], c=[x["eta0"] * 1e4 for x in hv], cmap="viridis", s=22, edgecolor="#52514e", linewidth=0.4)
    b = d["best"]; ax[1].scatter([b["l2"]], [b["l3"]], marker="*", s=90, color=ORANGE, zorder=3, label="selected")
    ax[1].set_xlabel("$\\lambda_2$"); ax[1].set_ylabel("$\\lambda_3$"); ax[1].legend(frameon=False, fontsize=6, loc="lower left")
    ax[1].set_title("(b) Non-diverged candidates", fontsize=7)
    cb = fig.colorbar(sc, ax=ax[1], fraction=0.05); cb.ax.tick_params(labelsize=6); cb.set_label("$\\eta_0$ ($\\times 10^{-4}$)", fontsize=6)
    for a in ax: a.spines[["top", "right"]].set_visible(False); a.tick_params(labelsize=6); a.grid(alpha=0.25, lw=0.4)
    save(fig, "figS6_doa")


if __name__ == "__main__":
    what = sys.argv[2:] or ["results", "qual", "conv", "doa", "abl"]
    if "abl" in what: fig_ablation()
    if "results" in what: fig_results()
    if "qual" in what: print("qual tiles", fig_qualitative())
    if "conv" in what: fig_convergence()
    if "doa" in what: fig_doa()
