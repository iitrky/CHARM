"""Manuscript figures from saved predictions.
  python figures.py --results results                       # Fig. 2 (risk-coverage) + Fig. 3 (territory attribution)
  python figures.py --results results --root /ptb-xl --ckpt results/charm_seed0_final.pt   # + Fig. 4, Fig. 5
"""
import argparse, glob, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "12-lead__both__featmiss=0.0"
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
TERR = {"Inferior": ["II", "III", "aVF"], "Lateral": ["I", "aVL", "V5", "V6"], "Septal": ["V1", "V2"],
        "Anterior": ["V3", "V4"], "Right": ["aVR"]}
NICE = {"charm": "CHARM (ours)", "late_fusion": "Late fusion (confidence)", "mc_dropout": "MC-Dropout", "late_fusion_ts": "Late fusion + TS"}


def risk_coverage(p, y, u):
    err = ((p > 0.5) != y.astype(bool)).mean(1); o = np.argsort(u)
    return np.arange(1, len(o) + 1) / len(o), np.cumsum(err[o]) / np.arange(1, len(o) + 1)


def fig2(R):
    plt.figure(figsize=(5, 3.6))
    for tag in ["charm", "late_fusion", "late_fusion_ts", "mc_dropout"]:
        f = f"{R}/{tag}_seed0_{BASE}.npz"
        if not os.path.exists(f): continue
        d = np.load(f); c, r = risk_coverage(d["p"], d["y"], d["u"])
        plt.plot(c, r, label=NICE.get(tag, tag), lw=2 if tag == "charm" else 1.2)
    plt.axhline(0.05, ls="--", c="gray", lw=.8); plt.xlabel("Coverage"); plt.ylabel("Selective risk (Hamming)")
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(f"{R}/fig2_risk_coverage.png", dpi=300); plt.close()


def fig3(R, mi_col=1):
    d = np.load(f"{R}/charm_seed0_{BASE}.npz"); w, y, p = d["lead_w"], d["y"], d["p"]
    sel = (y[:, mi_col] == 1) & (p[:, mi_col] > 0.5)                 # correctly detected MI
    if sel.sum() == 0: print("no correct MI cases"); return
    idx = {l: i for i, l in enumerate(LEADS)}
    terr = np.array([[w[s][[idx[l] for l in L]].sum() for L in TERR.values()] for s in np.where(sel)[0]])
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].bar(LEADS, w[sel].mean(0)); ax[0].set_title("Mean lead weight, detected MI"); ax[0].tick_params(axis="x", rotation=60)
    ax[1].boxplot(terr); ax[1].set_xticks(range(1, len(TERR) + 1), list(TERR)); ax[1].set_title("Territory attribution")
    plt.tight_layout(); plt.savefig(f"{R}/fig3_territory.png", dpi=300); plt.close()


def fig4_fig5(R, root, ckpt):
    import torch
    from torch.utils.data import DataLoader
    from charm_model import CHARM
    from charm_data import load_ptbxl, ECGClinDataset, NUM_FEATS, CAT_LEVELS, SUPERCLASSES
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = CHARM(len(NUM_FEATS), CAT_LEVELS, len(SUPERCLASSES)); m.load_state_dict(ck["state"]); m.eval()
    X, Y, clin, folds = load_ptbxl(root); te = folds == 10
    ds = ECGClinDataset(X[te], Y[te], clin[te], stats=ck["stats"]); b = next(iter(DataLoader(ds, 512)))

    def film_strength(bb):
        with torch.no_grad():
            zc = m.clin(bb["x_num"], bb["x_cat"], bb["m_num"], bb["m_cat"])
            return torch.stack([(g - 1).abs().mean(-1) for g, _ in m.hyper(zc)], -1).mean(-1)   # (B,)
    # Fig. 4: permutation importance of each clinical variable on FiLM modulation strength
    base = film_strength(b); names = NUM_FEATS + ["sex"]; imp = []
    for j, n in enumerate(names):
        bb = {k: v.clone() for k, v in b.items()}; perm = torch.randperm(len(base))
        if j < len(NUM_FEATS): bb["x_num"][:, j] = bb["x_num"][perm, j]; bb["m_num"][:, j] = bb["m_num"][perm, j]
        else: bb["x_cat"][:, 0] = bb["x_cat"][perm, 0]; bb["m_cat"][:, 0] = bb["m_cat"][perm, 0]
        imp.append((film_strength(bb) - base).abs().mean().item())
    plt.figure(figsize=(4.5, 3)); plt.barh(names, imp); plt.xlabel("Δ |γ − 1| (permutation)")
    plt.tight_layout(); plt.savefig(f"{R}/fig4_film_importance.png", dpi=300); plt.close()

    # Fig. 5: counterfactual age sweep on the same ECGs, P(HYP)
    ages = np.arange(25, 86, 5); mu, sd = ck["stats"][0][0], ck["stats"][1][0]
    hyp = SUPERCLASSES.index("HYP"); curves = {}
    for sex, lab in [(0, "male"), (1, "female")]:
        out = []
        for a_ in ages:
            bb = {k: v.clone() for k, v in b.items()}
            bb["x_num"][:, 0] = (a_ - mu) / sd; bb["m_num"][:, 0] = 1; bb["x_cat"][:, 0] = sex; bb["m_cat"][:, 0] = 1
            with torch.no_grad():
                out.append(m(bb["ecg"], bb["lead_mask"], bb["x_num"], bb["x_cat"], bb["m_num"], bb["m_cat"],
                             bb["has_ecg"], bb["has_clin"])["prob"][:, hyp].mean().item())
        curves[lab] = out
    plt.figure(figsize=(4.5, 3))
    for lab, v in curves.items(): plt.plot(ages, v, marker="o", label=lab)
    plt.xlabel("Counterfactual age (years)"); plt.ylabel("Mean P(HYP), same ECGs"); plt.legend()
    plt.tight_layout(); plt.savefig(f"{R}/fig5_counterfactual_age.png", dpi=300); plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--results", default="results")
    ap.add_argument("--root"); ap.add_argument("--ckpt"); a = ap.parse_args()
    fig2(a.results); fig3(a.results)
    if a.root and a.ckpt: fig4_fig5(a.results, a.root, a.ckpt)
    print("figures written to", a.results)
