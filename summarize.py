"""Collect every number used in the paper into results/summary.json (identity tiles excluded from full-reference means)."""
import json, os
import numpy as np
from scipy.stats import wilcoxon

H = os.path.dirname(os.path.abspath(__file__)); R = f"{H}/results"; O = f"{H}/outputs"
import sys; sys.path.insert(0, H)
J = lambda n: json.load(open(os.path.join(R, n)))
params = json.load(open(f"{H}/tiles/test_deg_params.json"))
ident = np.array([p["k"] == 1.0 and p["sigma_g"] == 0 and p["L"] == 0 and p["sigma_n"] == 0 for p in params])
keep = ~ident
dark = np.array([p["k"] < 0.4 for p in params]) & keep
blur = np.array([(p["sigma_g"] > 0 or p["L"] > 0) for p in params]) & keep


def per_tile(name):
    e = J(f"eval_{name}.json")
    ps, ss = np.array(e["psnr_per_tile"]), np.array(e["ssim_per_tile"])
    if "gmsd_per_tile" in e:
        g = np.array(e["gmsd_per_tile"])
    else:
        import exp
        out = np.load(os.path.join(O, f"{name}.npy"), mmap_mode="r"); clean = np.asarray(exp.load("test"))
        g = exp.gmsd_batch(np.asarray(out), clean)
        e["gmsd_per_tile"] = g.round(5).tolist(); json.dump(e, open(os.path.join(R, f"eval_{name}.json"), "w"))
    return e, ps, ss, g


def q(name):
    e, ps, ss, g = per_tile(name)
    r = {"psnr": float(ps[keep].mean()), "ssim": float(ss[keep].mean()), "gmsd": float(g[keep].mean()),
         "psnr_dark": float(ps[dark].mean()), "psnr_blur": float(ps[blur].mean()), "sec_per_tile": e.get("sec_per_tile_cpu")}
    if "niqe" in e: r["niqe"] = e["niqe"]
    return r, ps


S = {"n_test": int(len(params)), "n_identity_excluded": int(ident.sum()), "n_dark": int(dark.sum()), "n_blur": int(blur.sum())}
cx = J("complexity.json"); ds = J("downstream.json"); doa = J("doa.json")
S["doa"] = doa["best"]; S["doa_evals"] = len(doa["history"])
main = ["input", "clahe", "zerodce", "srcnn", "esrgan", "unet", "nafnet", "restormer", "daxnet_doa", "daxnet"]
SRC = {"daxnet": "abl_novgg", "daxnet_doa": "daxnet"}      # final model = DOA lambda2/eta0 with lambda3 = 0
T2, PT = {}, {}
for m in main:
    src = SRC.get(m, m)
    r, ps = q(src); PT[m] = ps
    ck = "daxnet" if m.startswith("daxnet") else m
    if ck in cx: r.update(cx[ck])
    d = ds["methods"][src]; r.update({"acc": d["mean_acc"], "f1": d["mean_f1"], "precision": d["mean_precision"], "recall": d["mean_recall"]})
    r["tasks"] = {t: d[t] for t in ("landcover", "vegetation", "water", "object")}
    T2[m] = r
c = ds["methods"]["clean"]; T2["clean"] = {"niqe": J("eval_clean.json")["niqe"], "acc": c["mean_acc"], "f1": c["mean_f1"],
                                           "tasks": {t: c[t] for t in ("landcover", "vegetation", "water", "object")}}
S["table2"] = T2
S["wilcoxon_psnr_dax_vs"] = {m: float(wilcoxon(PT["daxnet"][keep], PT[m][keep]).pvalue) for m in ["restormer", "nafnet", "unet", "daxnet_doa"]}
S["win_rate_vs"] = {m: float((PT["daxnet"][keep] > PT[m][keep]).mean()) for m in ["restormer", "nafnet", "unet", "daxnet_doa"]}
_, _, ss_f, _ = per_tile("abl_novgg"); _, _, ss_r, _ = per_tile("restormer")
S["wilcoxon_ssim_dax_vs_restormer"] = float(wilcoxon(ss_f[keep], ss_r[keep]).pvalue)
enc = {}
for n, key in [("squeezenet", "enc_squeezenet"), ("resnet18", "enc_resnet18"), ("resnet50", "enc_resnet50"), ("mobilenetv2", "daxnet")]:
    r, _ = q(key); r.update(cx[key] if key in cx else cx["enc_" + n] if ("enc_" + n) in cx else {}); r["val_psnr"] = J(f"train_{key}.json")["best_val_psnr"]
    enc[n] = r
S["table3"] = enc
abl = {"full": q("daxnet")[0]}
full_ps = per_tile("daxnet")[1]
for k in ["abl_noca", "abl_noaux", "abl_nose", "abl_novgg", "abl_nossim", "abl_nolowlight", "abl_nodoa"]:
    abl[k] = q(k)[0]; ps_k = per_tile(k)[1]
    abl[k]["delta_psnr_vs_full"] = float((ps_k[keep] - full_ps[keep]).mean())
    abl[k]["wilcoxon_p"] = float(wilcoxon(ps_k[keep], full_ps[keep]).pvalue)
abl["abl_novgg"].update({"acc": ds["methods"]["abl_novgg"]["mean_acc"], "f1": ds["methods"]["abl_novgg"]["mean_f1"]})
abl["full"].update({"acc": ds["methods"]["daxnet"]["mean_acc"], "f1": ds["methods"]["daxnet"]["mean_f1"]})
S["table4"] = abl
rb = J("robustness.json"); rb.update({k.replace("abl_novgg", "daxnet_final"): v for k, v in J("robustness_final.json").items()})
S["robustness"] = rb
S["train"] = {f[6:-5]: {"best_val_psnr": J(f)["best_val_psnr"], "minutes": J(f)["train_time_s"] / 60,
                        "best_iter": max(J(f)["curve"], key=lambda c: c["val_psnr"])["iter"]}
              for f in os.listdir(R) if f.startswith("train_") and f.endswith(".json") and "bestpsnr" not in f}
S["downstream_heads"] = ds["heads"]; S["label_stats"] = ds["label_stats"]
json.dump(S, open(os.path.join(R, "summary.json"), "w"), indent=1)


def fmt(d):
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items() if k != "tasks"}
for k, v in S["table2"].items(): print(k, fmt(v))
print("enc", {k: fmt(v) for k, v in S["table3"].items()})
print("abl", {k: fmt(v) for k, v in S["table4"].items()})
print("wilcoxon", S["wilcoxon_psnr_dax_vs"], S["win_rate_vs"])
