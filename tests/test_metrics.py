"""Torch-free tests: run with `pytest -q`."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sklearn.metrics import roc_auc_score
from metrics_utils import ece_multilabel, aurc, select_threshold, metrics
from aggregate import delong


def test_ece_perfect_calibration_is_small():
    rng = np.random.default_rng(0); p = rng.random((20000, 1)); y = (rng.random((20000, 1)) < p).astype(float)
    assert ece_multilabel(p, y) < 0.02


def test_aurc_prefers_informative_uncertainty():
    rng = np.random.default_rng(1); y = (rng.random((500, 5)) < .3).astype(float)
    p = np.clip(y + rng.normal(0, .4, y.shape), 0, 1)
    err = ((p > .5) != y.astype(bool)).mean(1)
    assert aurc(p, y, err) < aurc(p, y, -err)


def test_threshold_respects_target_risk():
    rng = np.random.default_rng(2); y = (rng.random((1000, 5)) < .3).astype(float)
    p = np.clip(y + rng.normal(0, .5, y.shape), 0, 1); u = -np.abs(p - .5).mean(1)
    t = select_threshold(u, p, y, 0.05); keep = u <= t
    assert ((p[keep] > .5) != y[keep].astype(bool)).mean() <= 0.05 + 1e-9


def test_delong_matches_sklearn_auc():
    rng = np.random.default_rng(3); y = rng.integers(0, 2, 300); p1 = rng.random(300) + .3 * y
    (a1, _), pv = delong(y, p1, p1)
    assert abs(a1 - roc_auc_score(y, p1)) < 1e-9 and pv > 0.99


def test_metrics_keys():
    rng = np.random.default_rng(4); y = (rng.random((200, 5)) < .3).astype(float); p = rng.random((200, 5))
    m = metrics(dict(p=p, y=y, u=rng.random(200)))
    assert set(m) == {"macro_auroc", "macro_f1", "ece", "aurc"}
