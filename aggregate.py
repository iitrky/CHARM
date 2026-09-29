"""Aggregate seeds -> mean ± std tables (CSV + Markdown) matching manuscript Tables 3-7,
DeLong p-values vs. the strongest baseline, bootstrap 95% CI, and deep-ensemble results.
  python aggregate.py --results results
"""
import argparse, glob, json, os, re
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from metrics_utils import metrics, select_threshold

BASE = "12-lead|both|featmiss=0.0"


def load(res, tag):
    fs = sorted(glob.glob(f"{res}/{tag}_seed[0-9].json"))
    return [json.load(open(f)) for f in fs]


def ms(vals, nd=3):
    v = [x for x in vals if x is not None]
    return f"{np.mean(v):.{nd}f} ± {np.std(v):.{nd}f}" if v else "n/a"


# ---- DeLong (Sun & Xu fast implementation) ----
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1; i = j
    T2 = np.empty(N); T2[J] = T; return T2


def delong(y, p1, p2):
    pos, neg = y == 1, y == 0; m, n = pos.sum(), neg.sum()
    aucs, V10, V01 = [], [], []
    for p in (p1, p2):
        tx, ty, tz = _midrank(p[pos]), _midrank(p[neg]), _midrank(np.r_[p[pos], p[neg]])
        aucs.append((tz[:m].sum() - m * (m + 1) / 2) / (m * n))
        V10.append((tz[:m] - tx) / n); V01.append(1 - (tz[m:] - ty) / m)
    S = np.cov(np.array(V10)) / m + np.cov(np.array(V01)) / n
    z = (aucs[0] - aucs[1]) / np.sqrt(max(S[0, 0] + S[1, 1] - 2 * S[0, 1], 1e-12))
    return aucs, 2 * stats.norm.sf(abs(z))


def boot_ci(y, p, B=1000, rng=np.random.default_rng(0)):
    vals = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        ok = [k for k in range(y.shape[1]) if 0 < y[i, k].sum() < len(i)]
        vals.append(np.mean([roc_auc_score(y[i, k], p[i, k]) for k in ok]))
    return np.percentile(vals, [2.5, 97.5])


def npz(res, tag, seed, key):
    k = key.replace('|', '__').replace('(', '').replace(')', '').replace(',', '-')
    return np.load(f"{res}/{tag}_seed{seed}_{k}.npz")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results", default="results")
    ap.add_argument("--ensemble", default="late_fusion"); a = ap.parse_args(); R = a.results
    tags = sorted({re.sub(r"_seed\d+\.json$", "", os.path.basename(f)) for f in glob.glob(f"{R}/*_seed[0-9].json")})
    out = []

    # deep ensemble from the saved per-seed predictions of `--ensemble`
    ens = []
    for key in load(R, a.ensemble)[0] if load(R, a.ensemble) else []:
        if key == "cost": continue
        seeds = [npz(R, a.ensemble, s, key) for s in range(5) if os.path.exists(f"{R}/{a.ensemble}_seed{s}.json")]
        p = np.mean([s["p"] for s in seeds], 0); y = seeds[0]["y"]
        u = np.mean([s["p"] for s in seeds], 0); u = 1 - np.maximum(u, 1 - u).mean(-1) + np.std([s["p"] for s in seeds], 0).mean(-1)
        ens.append((key, metrics(dict(p=p, y=y, u=u))))
    ens = dict(ens)

    def table(keys, cols, name, fields):
        rows = []
        for t in tags:
            runs = load(R, t)
            rows.append([t] + [ms([r.get(k, {}).get(f) for r in runs]) for k, f in zip(keys, fields)])
        if ens: rows.append([f"deep_ensemble({a.ensemble})"] + [f"{ens[k][f]:.3f}" if k in ens and f in ens[k] else "n/a" for k, f in zip(keys, fields)])
        df = pd.DataFrame(rows, columns=["method"] + cols); df.to_csv(f"{R}/{name}.csv", index=False)
        out.append(f"\n## {name}\n\n" + df.to_markdown(index=False))

    table([BASE] * 4, ["Macro-AUROC", "Macro-F1", "ECE", "AURC"], "table3_main", ["macro_auroc", "macro_f1", "ece", "aurc"])
    leads = ["12-lead", "6-limb", "3-lead(I,II,V2)", "2-lead(I,II)", "1-lead(I)"]
    table([f"{l}|both|featmiss=0.0" for l in leads], leads, "table4_leads", ["macro_auroc"] * 5)
    miss = [f"12-lead|both|featmiss={r}" for r in (0.0, 0.25, 0.5, 0.75, 1.0)] + ["12-lead|ecg_only|featmiss=0.0", "12-lead|clin_only|featmiss=0.0"]
    table(miss, ["0%", "25%", "50%", "75%", "100%", "ECG only", "Clinical only"], "table5_missing", ["macro_auroc"] * 7)
    table([BASE], ["Coverage@rho"], "selective_coverage", ["coverage"])

    # cost
    rows = [[t, ms([r["cost"].get("params_M") for r in load(R, t) if "cost" in r], 2),
             ms([r["cost"].get("cpu_ms_per_ecg") for r in load(R, t) if "cost" in r], 1)] for t in tags]
    out.append("\n## cost\n\n" + pd.DataFrame(rows, columns=["method", "Params (M)", "CPU ms/ECG"]).to_markdown(index=False))

    # external
    ext = glob.glob(f"{R}/external_*_seed*.json")
    if ext:
        df = pd.DataFrame([dict(file=os.path.basename(f), **json.load(open(f))) for f in ext])
        df["cohort_model"] = df.file.str.replace(r"_seed\d+\.json", "", regex=True)
        g = df.groupby("cohort_model")[["macro_auroc", "ece", "coverage"]].agg(["mean", "std"]).round(3)
        g.to_csv(f"{R}/table6_external.csv"); out.append("\n## table6_external\n\n" + g.to_markdown())

    # significance: CHARM vs. best baseline (seed 0 predictions, per-class DeLong + bootstrap CI)
    if "charm" in tags:
        base_tags = [t for t in tags if t != "charm" and not t.startswith("charm_")]
        best = max(base_tags, key=lambda t: np.mean([r[BASE]["macro_auroc"] for r in load(R, t)]))
        c, b = npz(R, "charm", 0, BASE), npz(R, best, 0, BASE)
        lines = [f"\n## significance (seed 0): CHARM vs {best}\n", "| class | AUROC CHARM | AUROC best | DeLong p |", "|---|---|---|---|"]
        SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
        for k, name in enumerate(SUPERCLASSES):
            (a1, a2), pv = delong(c["y"][:, k], c["p"][:, k], b["p"][:, k]); lines.append(f"| {name} | {a1:.3f} | {a2:.3f} | {pv:.2e} |")
        lo, hi = boot_ci(c["y"], c["p"]); lines.append(f"\nCHARM macro-AUROC 95% bootstrap CI: [{lo:.3f}, {hi:.3f}]")
        out.append("\n".join(lines))

    open(f"{R}/ALL_TABLES.md", "w").write("# CHARM results (auto-generated)\n" + "\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
