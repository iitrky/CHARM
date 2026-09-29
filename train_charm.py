"""Train / evaluate CHARM on PTB-XL.
  python train_charm.py --root /path/to/ptb-xl-1.0.3 --seed 0
  python train_charm.py --synthetic            # smoke test without data
"""
import argparse, json, os
import numpy as np
import torch
from torch.utils.data import DataLoader
from metrics_utils import ece_multilabel, aurc, select_threshold, metrics
from charm_model import (CHARM, beta_evidential_loss, beta_kl_to_uniform, imputation_loss,
                         consistency_loss, prototype_loss, uncertainty_score)
from charm_data import (ECGClinDataset, load_ptbxl, NUM_FEATS, CAT_LEVELS, SUPERCLASSES, LEAD_CONFIGS)


def synthetic(n=600, T=1000):
    import pandas as pd
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 12, T)).astype(np.float32)
    Y = (rng.random((n, 5)) < 0.3).astype(np.float32)
    X[:, :, :100] += Y[:, None, :1] * 2
    clin = pd.DataFrame(dict(age=rng.normal(60, 15, n), height=rng.normal(170, 10, n),
                             weight=rng.normal(75, 12, n), sex=rng.integers(0, 2, n).astype(float)))
    clin["bmi"] = clin.weight / (clin.height / 100) ** 2
    clin.loc[rng.random(n) < 0.6, ["height", "bmi"]] = np.nan
    folds = rng.integers(1, 11, n)
    return X, Y, clin, folds


def to_dev(b, dev):
    return {k: v.to(dev) for k, v in b.items()}


def forward(model, b):
    return model(b["ecg"], b["lead_mask"], b["x_num"], b["x_cat"], b["m_num"], b["m_cat"],
                 b["has_ecg"], b["has_clin"])


def train_epoch(model, loader, opt, sched, epoch, args, dev):
    model.train(); tot = 0
    kl_w = min(1.0, epoch / args.kl_anneal) * args.lam_kl
    for b in loader:
        b = to_dev(b, dev); y = b["y"]
        out = forward(model, b)
        if args.bce_loss:
            loss = torch.nn.functional.binary_cross_entropy(out["prob"].clamp(1e-6, 1 - 1e-6), y)
        else:
            loss = beta_evidential_loss(out["alpha"], out["beta"], y)
            loss = loss + kl_w * beta_kl_to_uniform(out["alpha"], out["beta"], y)
        loss = loss + args.lam_imp * imputation_loss(out, b["has_ecg"], b["has_clin"])
        loss = loss + args.lam_proto * prototype_loss(model, out["z"], y)
        # consistency: full-input teacher vs. ECG-only and clinical-only students
        full = b["has_ecg"] & b["has_clin"]
        if full.any():
            bf = {k: v[full] for k, v in b.items()}
            p_full = out["prob"][full]
            p_e = forward(model, {**bf, "has_clin": torch.zeros_like(bf["has_clin"])})["prob"]
            p_c = forward(model, {**bf, "has_ecg": torch.zeros_like(bf["has_ecg"])})["prob"]
            loss = loss + args.lam_cons * (consistency_loss(p_full, p_e) + 0.5 * consistency_loss(p_full, p_c))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        tot += loss.item() * len(y)
    return tot / len(loader.dataset)


@torch.no_grad()
def predict(model, loader, dev, d_ref=None, omega=0.5):
    model.eval(); P, Y, U, Z, W = [], [], [], [], []
    for b in loader:
        b = to_dev(b, dev); out = forward(model, b)
        P.append(out["prob"].cpu()); Y.append(b["y"].cpu()); W.append(out["lead_w"].cpu())
        Z.append(model.proto_distance(out["z"]).min(-1).values.cpu())
        if d_ref is not None:
            U.append(uncertainty_score(model, out, d_ref, omega).cpu())
    r = dict(p=torch.cat(P).numpy(), y=torch.cat(Y).numpy(), dmin=torch.cat(Z).numpy(), lead_w=torch.cat(W).numpy())
    if U: r["u"] = torch.cat(U).numpy()
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root"); ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lam_kl", type=float, default=0.1); ap.add_argument("--kl_anneal", type=int, default=10)
    ap.add_argument("--lam_imp", type=float, default=0.5); ap.add_argument("--lam_cons", type=float, default=0.5)
    ap.add_argument("--lam_proto", type=float, default=0.05)
    ap.add_argument("--target_risk", type=float, default=0.05)
    ap.add_argument("--out", default="results")
    ap.add_argument("--omega", type=float, default=0.5, help="1 = evidential only, 0 = prototype only")
    ap.add_argument("--no_film", action="store_true"); ap.add_argument("--no_prior", action="store_true")
    ap.add_argument("--no_aug", action="store_true"); ap.add_argument("--static_film", action="store_true")
    ap.add_argument("--bce_loss", action="store_true", help="ablation: plain BCE instead of evidential loss"); ap.add_argument("--tag", default="charm")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if args.synthetic:
        X, Y, clin, folds = synthetic(); args.epochs = 2
    else:
        X, Y, clin, folds = load_ptbxl(args.root)
    tr, va, te = folds <= 8, folds == 9, folds == 10          # official PTB-XL split
    aug = dict(p_drop_ecg=0, p_drop_clin=0, p_lead_cfg=0, p_feat=0) if args.no_aug else {}
    ds_tr = ECGClinDataset(X[tr], Y[tr], clin[tr], train=True, **aug)
    ds_va = ECGClinDataset(X[va], Y[va], clin[va], stats=ds_tr.stats)
    ds_te = ECGClinDataset(X[te], Y[te], clin[te], stats=ds_tr.stats)
    dl = lambda d, s: DataLoader(d, args.bs, shuffle=s, num_workers=2 if not args.synthetic else 0)
    model = CHARM(len(NUM_FEATS), CAT_LEVELS, len(SUPERCLASSES), use_film=not args.no_film, use_prior=not args.no_prior,
                  static_film=args.static_film).to(dev)
    opt = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * len(dl(ds_tr, True)))
    best = -1.0
    os.makedirs(args.out, exist_ok=True)
    ck = f"{args.out}/{args.tag}_seed{args.seed}.pt"
    for ep in range(args.epochs):
        loss = train_epoch(model, dl(ds_tr, True), opt, sched, ep, args, dev)
        m = metrics(predict(model, dl(ds_va, False), dev))
        print(f"epoch {ep:3d} loss {loss:.4f} val {m}")
        if m["macro_auroc"] > best:
            best = m["macro_auroc"]; torch.save(model.state_dict(), ck)
    model.load_state_dict(torch.load(ck, map_location=dev))

    # reference distance for prototype uncertainty (training set, clean inputs)
    ds_tr_eval = ECGClinDataset(X[tr], Y[tr], clin[tr], stats=ds_tr.stats)
    d_ref = float(np.percentile(predict(model, dl(ds_tr_eval, False), dev)["dmin"], 95))

    results = {}
    conditions = [(l, None, 0.0) for l in LEAD_CONFIGS] + \
                 [("12-lead", "ecg_only", 0.0), ("12-lead", "clin_only", 0.0)] + \
                 [("12-lead", None, r) for r in (0.25, 0.5, 0.75, 1.0)]
    for leads, force, fr in conditions:
        for d in (ds_va, ds_te): d.set_eval_condition(leads, force, fr)
        rv, rt = predict(model, dl(ds_va, False), dev, d_ref, args.omega), predict(model, dl(ds_te, False), dev, d_ref, args.omega)
        m = metrics(rt)
        thr = select_threshold(rv["u"], rv["p"], rv["y"], args.target_risk)
        keep = rt["u"] <= thr
        m["coverage"] = float(keep.mean())
        m["selective_risk"] = float(((rt["p"][keep] > 0.5) != rt["y"][keep].astype(bool)).mean()) if keep.any() else None
        key = f"{leads}|{force or 'both'}|featmiss={fr}"
        results[key] = m; print(key, m)
        np.savez_compressed(f"{args.out}/{args.tag}_seed{args.seed}_{key.replace('|','__').replace('(','').replace(')','').replace(',','-')}.npz",
                            p=rt["p"], y=rt["y"], u=rt["u"], lead_w=rt["lead_w"])
    for d in (ds_va, ds_te): d.set_eval_condition("12-lead")
    # model cost
    import time
    results["cost"] = dict(params_M=sum(p.numel() for p in model.parameters()) / 1e6)
    model.cpu().eval(); b = next(iter(DataLoader(ds_te, 1)))
    with torch.no_grad():
        for _ in range(5): forward(model, b)
        t0 = time.perf_counter()
        for _ in range(50): forward(model, b)
    results["cost"]["cpu_ms_per_ecg"] = (time.perf_counter() - t0) / 50 * 1000
    torch.save(dict(state=model.state_dict(), d_ref=d_ref, stats=ds_tr.stats, omega=args.omega,
                    tau=float(select_threshold(*[predict(model, dl(ds_va, False), 'cpu', d_ref, args.omega)[k] for k in ('u','p','y')], args.target_risk))),
               f"{args.out}/{args.tag}_seed{args.seed}_final.pt")
    with open(f"{args.out}/{args.tag}_seed{args.seed}.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
