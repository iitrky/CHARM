"""PTB-XL data pipeline + missing-modality / reduced-lead augmentation for CHARM.
Download PTB-XL v1.0.3 from https://physionet.org/content/ptb-xl/ and pass its folder as --root.
"""
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
NUM_FEATS = ["age", "height", "weight", "bmi"]      # numeric clinical features
CAT_FEATS = ["sex"]                                  # categorical
CAT_LEVELS = [2]

LEAD_CONFIGS = {                                     # name -> lead indices kept
    "12-lead": list(range(12)),
    "6-limb": [0, 1, 2, 3, 4, 5],
    "2-lead(I,II)": [0, 1],
    "1-lead(I)": [0],
    "3-lead(I,II,V2)": [0, 1, 7],
}


def load_ptbxl(root, sampling_rate=100):
    import wfdb
    df = pd.read_csv(f"{root}/ptbxl_database.csv", index_col="ecg_id")
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)
    agg = pd.read_csv(f"{root}/scp_statements.csv", index_col=0)
    agg = agg[agg.diagnostic == 1]

    def to_super(codes):
        return list({agg.loc[c].diagnostic_class for c in codes if c in agg.index})

    df["superclass"] = df.scp_codes.apply(to_super)
    df = df[df.superclass.map(len) > 0]
    files = df.filename_lr if sampling_rate == 100 else df.filename_hr
    X = np.stack([wfdb.rdsamp(f"{root}/{f}")[0] for f in files]).astype(np.float32)  # (N,T,12)
    X = X.transpose(0, 2, 1)                                                          # (N,12,T)
    Y = np.stack([[int(c in s) for c in SUPERCLASSES] for s in df.superclass]).astype(np.float32)

    clin = pd.DataFrame(index=df.index)
    clin["age"] = df.age.where(df.age < 120)      # PTB-XL encodes age>89 as 300
    clin["height"] = df.height
    clin["weight"] = df.weight
    clin["bmi"] = df.weight / (df.height / 100.0) ** 2
    clin["sex"] = df.sex
    return X, Y, clin, df.strat_fold.values


class ECGClinDataset(Dataset):
    def __init__(self, X, Y, clin, stats=None, train=False,
                 p_drop_ecg=0.15, p_drop_clin=0.15, p_lead_cfg=0.5, p_feat=0.3):
        self.X, self.Y, self.train = X, Y, train
        num = clin[NUM_FEATS].values.astype(np.float32)
        if stats is None:                           # z-score using TRAIN statistics only
            stats = (np.nanmean(num, 0), np.nanstd(num, 0) + 1e-6,
                     X.mean((0, 2), keepdims=True)[0], X.std((0, 2), keepdims=True)[0] + 1e-6)
        self.stats = stats
        self.num = (num - stats[0]) / stats[1]
        self.m_num = (~np.isnan(num)).astype(np.float32)
        cat = clin[CAT_FEATS].values
        self.m_cat = (~pd.isna(cat)).astype(np.float32)
        self.cat = np.nan_to_num(cat.astype(np.float32), nan=0).astype(np.int64)
        self.p = dict(ecg=p_drop_ecg, clin=p_drop_clin, lead=p_lead_cfg, feat=p_feat)
        self.lead_mask_eval = np.ones(12, np.float32)
        self.force = None                           # None | "ecg_only" | "clin_only"
        self.feat_missing_rate = 0.0

    def __len__(self):
        return len(self.Y)

    def set_eval_condition(self, leads="12-lead", force=None, feat_missing_rate=0.0):
        m = np.zeros(12, np.float32); m[LEAD_CONFIGS[leads]] = 1
        self.lead_mask_eval, self.force, self.feat_missing_rate = m, force, feat_missing_rate

    def __getitem__(self, i):
        rng = np.random
        ecg = (self.X[i] - self.stats[2]) / self.stats[3]
        m_num, m_cat = self.m_num[i].copy(), self.m_cat[i].copy()
        has_ecg, has_clin = True, True
        if self.train:
            lead_mask = np.ones(12, np.float32)
            if rng.rand() < self.p["lead"]:        # sample a reduced-lead configuration
                cfg = rng.choice(list(LEAD_CONFIGS) + ["random"])
                if cfg == "random":
                    k = rng.randint(1, 12)
                    lead_mask[:] = 0; lead_mask[rng.choice(12, k, replace=False)] = 1
                else:
                    lead_mask[:] = 0; lead_mask[LEAD_CONFIGS[cfg]] = 1
            if rng.rand() < self.p["feat"]:        # random clinical-feature missingness
                r = rng.rand()
                m_num *= (rng.rand(len(m_num)) > r); m_cat *= (rng.rand(len(m_cat)) > r)
            u = rng.rand()                          # whole-modality dropout (never both)
            if u < self.p["ecg"]:
                has_ecg = False
            elif u < self.p["ecg"] + self.p["clin"]:
                has_clin = False
        else:
            lead_mask = self.lead_mask_eval.copy()
            if self.feat_missing_rate > 0:
                keep = rng.rand(len(m_num)) > self.feat_missing_rate
                m_num *= keep; m_cat *= (rng.rand(len(m_cat)) > self.feat_missing_rate)
            if self.force == "ecg_only":
                has_clin = False
            elif self.force == "clin_only":
                has_ecg = False
        if not has_clin:
            m_num[:] = 0; m_cat[:] = 0
        return dict(ecg=torch.from_numpy(ecg.astype(np.float32)), lead_mask=torch.from_numpy(lead_mask),
                    x_num=torch.from_numpy(self.num[i]), x_cat=torch.from_numpy(self.cat[i]),
                    m_num=torch.from_numpy(m_num), m_cat=torch.from_numpy(m_cat),
                    has_ecg=torch.tensor(has_ecg), has_clin=torch.tensor(has_clin),
                    y=torch.from_numpy(self.Y[i]))
