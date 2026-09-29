"""Torch-free metric utilities shared by all scripts."""
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score


def ece_multilabel(p, y, bins=15):
    p, y = p.ravel(), y.ravel(); e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        if m.any(): e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e


def aurc(p, y, u):
    """Area under risk-coverage curve; risk = sample-level Hamming error at 0.5."""
    err = ((p > 0.5) != y.astype(bool)).mean(1)
    order = np.argsort(u); err = err[order]
    risks = np.cumsum(err) / np.arange(1, len(err) + 1)
    return risks.mean()


def select_threshold(u_val, p_val, y_val, target_risk):
    """Largest coverage whose selective risk on validation <= target."""
    err = ((p_val > 0.5) != y_val.astype(bool)).mean(1)
    order = np.argsort(u_val); risks = np.cumsum(err[order]) / np.arange(1, len(err) + 1)
    ok = np.where(risks <= target_risk)[0]
    return u_val[order][ok[-1]] if len(ok) else u_val.min()


def metrics(r):
    y, p = r["y"], r["p"]
    valid = [k for k in range(y.shape[1]) if 0 < y[:, k].sum() < len(y)]
    out = dict(macro_auroc=float(np.mean([roc_auc_score(y[:, k], p[:, k]) for k in valid])),
               macro_f1=float(f1_score(y, p > 0.5, average="macro", zero_division=0)),
               ece=float(ece_multilabel(p, y)))
    if "u" in r: out["aurc"] = float(aurc(p, y, r["u"]))
    return out


