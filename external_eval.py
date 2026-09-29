"""External validation (Table 6) on PhysioNet/CinC-2020 format cohorts (Chapman-Shaoxing, CPSC-2018).
Each record = <name>.mat (key 'val', 12 x N, 500 Hz) + <name>.hea with '#Age:', '#Sex:', '#Dx:' lines.
  python external_eval.py --ext_dir /data/cpsc_2018 --name cpsc --ckpt results/charm_seed0_final.pt
VERIFY the SNOMED mapping below against the challenge's dx_mapping_scored.csv before use.
"""
import argparse, glob, json, os
import numpy as np, pandas as pd, torch
from scipy.io import loadmat
from scipy.signal import resample_poly
from torch.utils.data import DataLoader
from charm_model import CHARM
from charm_data import ECGClinDataset, NUM_FEATS, CAT_LEVELS, SUPERCLASSES
from train_charm import predict
from metrics_utils import metrics, select_threshold

SNOMED = {"NORM": {"426783006"}, "MI": {"164865005", "164861001"},
          "STTC": {"164934002", "59931005", "429622005", "164930006"},
          "CD": {"59118001", "713427006", "164909002", "270492004", "445118002"},
          "HYP": {"164873001", "89792004"}}


def load_external(d, fs_in=500, T=1000):
    X, Y, rows = [], [], []
    for hea in sorted(glob.glob(f"{d}/**/*.hea", recursive=True)):
        meta = {l.split(":")[0][1:].strip(): l.split(":", 1)[1].strip() for l in open(hea) if l.startswith("#") and ":" in l}
        dx = set(meta.get("Dx", "").split(","))
        y = [int(bool(dx & SNOMED[c])) for c in SUPERCLASSES]
        if y[0] and sum(y) > 1: y[0] = 0                     # NORM only if nothing else
        if sum(y) == 0: continue                              # no mappable diagnosis
        x = loadmat(hea[:-4] + ".mat")["val"].astype(np.float32) / 1000.0          # to mV
        x = resample_poly(x, 1, fs_in // 100, axis=1)
        x = x[:, :T] if x.shape[1] >= T else np.pad(x, ((0, 0), (0, T - x.shape[1])))
        age = pd.to_numeric(meta.get("Age"), errors="coerce"); sex = meta.get("Sex", "").lower()
        rows.append(dict(age=age, height=np.nan, weight=np.nan, bmi=np.nan,
                         sex=0.0 if sex.startswith("m") else (1.0 if sex.startswith("f") else np.nan)))
        X.append(x); Y.append(y)
    return np.stack(X), np.array(Y, np.float32), pd.DataFrame(rows)
    # NB: PTB-XL codes sex as 0 = male, 1 = female; this matches.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext_dir", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--ckpt", required=True); ap.add_argument("--target_risk", type=float, default=0.05)
    ap.add_argument("--out", default="results")
    ap.add_argument("--model", default="charm", help="charm or a baselines.py model name")
    a = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    X, Y, clin = load_external(a.ext_dir)
    print(f"{a.name}: {len(Y)} records, label prevalence {Y.mean(0).round(3)}")
    ds = ECGClinDataset(X, Y, clin, stats=ck["stats"])
    if a.model == "charm":
        model = CHARM(len(NUM_FEATS), CAT_LEVELS, len(SUPERCLASSES)).to(dev); model.load_state_dict(ck["state"])
        r = predict(model, DataLoader(ds, 128), dev, ck["d_ref"], ck["omega"])
    else:
        from baselines import Baseline, predict as bpredict
        model = Baseline(a.model).to(dev); model.load_state_dict(ck["state"])
        r = bpredict(model, DataLoader(ds, 128), dev, ck["T"], ck["mc"])
    m = metrics(r); keep = r["u"] <= ck["tau"]            # threshold fixed on PTB-XL validation (no target data used)
    m.update(coverage=float(keep.mean()), n=int(len(Y)),
             selective_risk=float(((r["p"][keep] > .5) != r["y"][keep].astype(bool)).mean()) if keep.any() else None)
    print(m); os.makedirs(a.out, exist_ok=True)
    tag = os.path.basename(a.ckpt).replace("_final.pt", "")
    json.dump(m, open(f"{a.out}/external_{a.name}_{tag}.json", "w"), indent=2)
    np.savez_compressed(f"{a.out}/external_{a.name}_{tag}.npz", **r)


if __name__ == "__main__":
    main()
