"""Baselines for Tables 3-7, trained with the SAME split, augmentation and metrics as CHARM.
  python baselines.py --root /path/ptb-xl-1.0.3 --model late_fusion --seed 0
Models: xresnet | inception | cnn_lstm_att | transformer | ft_transformer | early_fusion |
        late_fusion | medfuse | mc_dropout | dagat (DA-GAT-v2-style) | hyperfusion (HyperFusion-style)  (mc_dropout = late_fusion evaluated with 20 dropout samples)
Deep ensemble: train late_fusion for seeds 0-4, then `python aggregate.py --ensemble late_fusion`.
Temperature scaling: add --temp_scale to late_fusion.
"""
import argparse, json, os
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from charm_model import ClinicalEncoder
from charm_data import ECGClinDataset, load_ptbxl, NUM_FEATS, CAT_LEVELS, SUPERCLASSES, LEAD_CONFIGS
from train_charm import synthetic, to_dev
from metrics_utils import metrics, select_threshold

K = len(SUPERCLASSES)


# ---------------- ECG backbones (all channels together, absent leads = 0) --------------
class ResBlk(nn.Module):
    def __init__(s, ci, co, st, k=7):
        super().__init__()
        s.f = nn.Sequential(nn.Conv1d(ci, co, k, st, k // 2, bias=False), nn.BatchNorm1d(co), nn.ReLU(),
                            nn.Conv1d(co, co, k, 1, k // 2, bias=False), nn.BatchNorm1d(co))
        s.sk = nn.Sequential(nn.Conv1d(ci, co, 1, st, bias=False), nn.BatchNorm1d(co)) if (ci != co or st > 1) else nn.Identity()
    def forward(s, x): return F.relu(s.f(x) + s.sk(x))


class XResNet1d(nn.Module):
    """Compact XResNet1d-style encoder (Strodthoff et al. use xresnet1d101; set depth via `blocks`)."""
    def __init__(s, cin=12, d=128, blocks=(3, 4, 6, 3), ch=(64, 128, 128, 256)):
        super().__init__()
        s.stem = nn.Sequential(nn.Conv1d(cin, 32, 5, 2, 2, bias=False), nn.BatchNorm1d(32), nn.ReLU(),
                               nn.Conv1d(32, 64, 5, 1, 2, bias=False), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(3, 2, 1))
        L, ci = [], 64
        for n, c in zip(blocks, ch):
            for i in range(n): L.append(ResBlk(ci, c, 2 if i == 0 and c != 64 else 1)); ci = c
        s.body = nn.Sequential(*L); s.proj = nn.Linear(2 * ci, d)
    def forward(s, x):
        h = s.body(s.stem(x)); return s.proj(torch.cat([h.mean(-1), h.amax(-1)], -1))


class Inception1d(nn.Module):
    def __init__(s, cin=12, d=128, nf=32, depth=6):
        super().__init__()
        s.blocks, s.res = nn.ModuleList(), nn.ModuleList(); ci = cin
        for i in range(depth):
            s.blocks.append(nn.ModuleDict(dict(b=nn.Conv1d(ci, nf, 1, bias=False),
                convs=nn.ModuleList([nn.Conv1d(nf, nf, k, padding=k // 2, bias=False) for k in (39, 19, 9)]),
                mp=nn.Sequential(nn.MaxPool1d(3, 1, 1), nn.Conv1d(ci, nf, 1, bias=False)), bn=nn.BatchNorm1d(4 * nf))))
            if i % 3 == 2: s.res.append(nn.Sequential(nn.Conv1d(cin if i == 2 else 4 * nf, 4 * nf, 1, bias=False), nn.BatchNorm1d(4 * nf)))
            ci = 4 * nf
        s.proj = nn.Linear(8 * nf, d)
    def forward(s, x):
        r = x
        for i, b in enumerate(s.blocks):
            z = b["b"](x); x = F.relu(b["bn"](torch.cat([c(z) for c in b["convs"]] + [b["mp"](x)], 1)))
            if i % 3 == 2: x = F.relu(x + s.res[i // 3](r)); r = x
        return s.proj(torch.cat([x.mean(-1), x.amax(-1)], -1))


class CNNLSTMAtt(nn.Module):
    def __init__(s, cin=12, d=128):
        super().__init__()
        s.cnn = nn.Sequential(nn.Conv1d(cin, 64, 7, 2, 3), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
                              nn.Conv1d(64, 128, 5, 1, 2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2))
        s.lstm = nn.LSTM(128, d // 2, batch_first=True, bidirectional=True); s.att = nn.Linear(d, 1)
    def forward(s, x):
        h, _ = s.lstm(s.cnn(x).transpose(1, 2)); w = s.att(h).softmax(1); return (w * h).sum(1)


class TransformerECG(nn.Module):
    def __init__(s, cin=12, d=128, patch=25):
        super().__init__()
        s.patch = nn.Conv1d(cin, d, patch, patch); s.cls = nn.Parameter(torch.zeros(1, 1, d)); s.pos = nn.Parameter(torch.randn(1, 64, d) * .02)
        s.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 2 * d, 0.1, batch_first=True, norm_first=True), 4)
    def forward(s, x):
        t = s.patch(x).transpose(1, 2); t = torch.cat([s.cls.expand(len(t), -1, -1), t], 1)
        return s.tf(t + s.pos[:, :t.size(1)])[:, 0]


BACKBONES = dict(xresnet=XResNet1d, inception=Inception1d, cnn_lstm_att=CNNLSTMAtt, transformer=TransformerECG)


class DAGATStyle(nn.Module):
    """Re-implementation in the spirit of DA-GAT-v2 (Naeem, Sci. Rep. 2026): lead-wise encoder,
    graph attention over leads, FiLM from AGE and SEX applied after every graph-attention layer,
    population-mean fallback when demographics are missing. (Fairness loss / alpha-Net omitted:
    we compare the conditioning mechanism, not the fairness objective.)"""
    def __init__(s, d=128, heads=4, depth=2):
        super().__init__()
        from charm_model import LeadwiseECGEncoder
        s.enc = LeadwiseECGEncoder((32, 64, 128, 128), d)
        s.lead_emb = nn.Parameter(torch.randn(12, d) * .02)
        s.att = nn.ModuleList([nn.MultiheadAttention(d, heads, batch_first=True) for _ in range(depth)])
        s.norm = nn.ModuleList([nn.LayerNorm(d) for _ in range(depth)])
        s.demo = nn.Sequential(nn.Linear(2, 64), nn.ReLU())
        s.film = nn.ModuleList([nn.Linear(64, 2 * d) for _ in range(depth)])
        for f in s.film: nn.init.zeros_(f.weight); nn.init.zeros_(f.bias)
    def forward(s, b):
        ecg = b["ecg"] * b["lead_mask"].unsqueeze(-1) * b["has_ecg"].view(-1, 1, 1).float()
        ones = [(torch.ones(len(ecg), c, device=ecg.device), torch.zeros(len(ecg), c, device=ecg.device)) for c in s.enc.stage_channels]
        h = s.enc(ecg, ones) + s.lead_emb
        # age (z-scored, index 0) and sex; missing -> 0 = training mean after z-scoring / 0.5 for sex
        age = torch.nan_to_num(b["x_num"][:, 0]) * b["m_num"][:, 0]
        sex = b["x_cat"][:, 0].float() * b["m_cat"][:, 0] + 0.5 * (1 - b["m_cat"][:, 0])
        e = s.demo(torch.stack([age, sex], -1))
        pad = ~(b["lead_mask"].bool() & b["has_ecg"].unsqueeze(-1))
        pad = pad & ~pad.all(1, keepdim=True)          # keep at least one key
        for att, nm, fl in zip(s.att, s.norm, s.film):
            a, _ = att(nm(h), nm(h), nm(h), key_padding_mask=pad)
            g, be = fl(e).chunk(2, -1)
            h = h + (1 + g.unsqueeze(1)) * a + be.unsqueeze(1)
        keep = (~pad).float().unsqueeze(-1)
        return (h * keep).sum(1) / keep.sum(1).clamp(min=1)


class HyperFusionStyle(nn.Module):
    """In the spirit of HyperFusion (Duenias et al., MedIA 2025): a hypernetwork maps the clinical
    embedding to the WEIGHTS of the final layer applied to the ECG embedding."""
    def __init__(s, d=128):
        super().__init__()
        s.ecg = XResNet1d(12, d); s.clin = ClinicalEncoder(len(NUM_FEATS), CAT_LEVELS, d)
        s.W0 = nn.Linear(d, K); s.hyper = nn.Linear(d, d * K + K)
        nn.init.zeros_(s.hyper.weight); nn.init.zeros_(s.hyper.bias)
    def forward(s, b):
        ecg = b["ecg"] * b["lead_mask"].unsqueeze(-1) * b["has_ecg"].view(-1, 1, 1).float()
        ze = s.ecg(ecg); zc = s.clin(b["x_num"], b["x_cat"], b["m_num"], b["m_cat"])
        dW = s.hyper(zc); W, bb = dW[:, :-K].view(-1, K, ze.size(-1)), dW[:, -K:]
        return s.W0(ze) + (W @ ze.unsqueeze(-1)).squeeze(-1) + bb


class Baseline(nn.Module):
    def __init__(s, kind, d=128):
        super().__init__()
        s.kind = kind
        if kind == "hyperfusion":
            s.net = HyperFusionStyle(d); return
        if kind == "dagat":
            s.net = DAGATStyle(d); s.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(d, K)); return
        ecg_kind = kind if kind in BACKBONES else "xresnet"
        s.use_ecg = kind != "ft_transformer"
        s.use_clin = kind in ("ft_transformer", "early_fusion", "late_fusion", "medfuse", "mc_dropout")
        if s.use_ecg: s.ecg = BACKBONES[ecg_kind](cin=12 + (2 * len(NUM_FEATS) + 2 if kind == "early_fusion" else 0), d=d)
        if s.use_clin and kind != "early_fusion": s.clin = ClinicalEncoder(len(NUM_FEATS), CAT_LEVELS, d)
        if kind == "medfuse": s.lstm = nn.LSTM(d, d, batch_first=True)
        nin = 2 * d if kind in ("late_fusion", "mc_dropout") else d
        s.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(nin, d), nn.ReLU(), nn.Dropout(0.3), nn.Linear(d, K))

    def forward(s, b):
        if s.kind == "hyperfusion": return s.net(b)
        if s.kind == "dagat": return s.head(s.net(b))
        ecg = b["ecg"] * b["lead_mask"].unsqueeze(-1) * b["has_ecg"].view(-1, 1, 1).float()
        if s.kind == "early_fusion":        # clinical values + masks broadcast as constant channels
            cv = torch.cat([torch.nan_to_num(b["x_num"]) * b["m_num"], b["m_num"], b["x_cat"].float() * b["m_cat"], b["m_cat"]], 1)
            return s.head(s.ecg(torch.cat([ecg, cv.unsqueeze(-1).expand(-1, -1, ecg.size(-1))], 1)))
        if not s.use_clin: return s.head(s.ecg(ecg))
        zc = s.clin(b["x_num"], b["x_cat"], b["m_num"], b["m_cat"])
        if not s.use_ecg: return s.head(zc)
        ze = s.ecg(ecg)
        if s.kind == "medfuse":              # LSTM over available modality tokens (ECG first, then clinical)
            seq = torch.stack([ze, zc], 1)
            lens = b["has_ecg"].long() + b["has_clin"].long()
            order = torch.where(b["has_ecg"].unsqueeze(-1), seq, seq.flip(1))   # put the available token first
            out, _ = s.lstm(order); idx = (lens.clamp(min=1) - 1).view(-1, 1, 1).expand(-1, 1, out.size(-1))
            return s.head(out.gather(1, idx).squeeze(1))
        return s.head(torch.cat([ze, zc], -1))


@torch.no_grad()
def predict(model, loader, dev, T=1.0, mc=0):
    model.eval(); P, Y, U = [], [], []
    if mc:
        for m in model.modules():
            if isinstance(m, nn.Dropout): m.train()
    for b in loader:
        b = to_dev(b, dev)
        if mc:
            ps = torch.stack([torch.sigmoid(model(b) / T) for _ in range(mc)]); p = ps.mean(0)
            u = -(p * p.clamp(1e-6).log() + (1 - p) * (1 - p).clamp(1e-6).log()).mean(-1)      # predictive entropy
        else:
            p = torch.sigmoid(model(b) / T); u = 1 - torch.maximum(p, 1 - p).mean(-1)           # confidence-based
        P.append(p.cpu()); Y.append(b["y"].cpu()); U.append(u.cpu())
    return dict(p=torch.cat(P).numpy(), y=torch.cat(Y).numpy(), u=torch.cat(U).numpy())


def fit_temperature(model, loader, dev):
    model.eval(); L, Y = [], []
    with torch.no_grad():
        for b in loader: b = to_dev(b, dev); L.append(model(b).cpu()); Y.append(b["y"].cpu())
    L, Y = torch.cat(L), torch.cat(Y); logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=100)
    def closure():
        opt.zero_grad(); l = F.binary_cross_entropy_with_logits(L / logT.exp(), Y); l.backward(); return l
    opt.step(closure); return float(logT.exp())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root"); ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--model", required=True, choices=list(BACKBONES) + ["ft_transformer", "early_fusion", "late_fusion", "medfuse", "mc_dropout", "dagat", "hyperfusion"])
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temp_scale", action="store_true"); ap.add_argument("--target_risk", type=float, default=0.05)
    ap.add_argument("--out", default="results")
    a = ap.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X, Y, clin, folds = synthetic() if a.synthetic else load_ptbxl(a.root)
    if a.synthetic: a.epochs = 1
    tr, va, te = folds <= 8, folds == 9, folds == 10
    fusion = a.model in ("early_fusion", "late_fusion", "medfuse", "mc_dropout", "dagat", "hyperfusion")
    aug = {} if fusion else dict(p_drop_ecg=0, p_drop_clin=0)      # modality dropout only makes sense for fusion
    if a.model == "ft_transformer": aug.update(p_lead_cfg=0)
    ds_tr = ECGClinDataset(X[tr], Y[tr], clin[tr], train=True, **aug)
    ds_va = ECGClinDataset(X[va], Y[va], clin[va], stats=ds_tr.stats); ds_te = ECGClinDataset(X[te], Y[te], clin[te], stats=ds_tr.stats)
    dl = lambda d, s: DataLoader(d, a.bs, shuffle=s, num_workers=0 if a.synthetic else 2)
    model = Baseline(a.model).to(dev)
    opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * len(dl(ds_tr, True)))
    tag = a.model + ("_ts" if a.temp_scale else ""); os.makedirs(a.out, exist_ok=True)
    ck, best = f"{a.out}/{tag}_seed{a.seed}.pt", -1
    for ep in range(a.epochs):
        model.train()
        for b in dl(ds_tr, True):
            b = to_dev(b, dev); loss = F.binary_cross_entropy_with_logits(model(b), b["y"])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        m = metrics(predict(model, dl(ds_va, False), dev)); print(ep, m)
        if m["macro_auroc"] > best: best = m["macro_auroc"]; torch.save(model.state_dict(), ck)
    model.load_state_dict(torch.load(ck, map_location=dev))
    T = fit_temperature(model, dl(ds_va, False), dev) if a.temp_scale else 1.0
    mc = 20 if a.model == "mc_dropout" else 0
    res = {}
    conds = [(l, None, 0.0) for l in LEAD_CONFIGS] + [("12-lead", "ecg_only", 0.0), ("12-lead", "clin_only", 0.0)] + \
            [("12-lead", None, r) for r in (0.25, 0.5, 0.75, 1.0)]
    for leads, force, fr in conds:
        for d in (ds_va, ds_te): d.set_eval_condition(leads, force, fr)
        rv, rt = predict(model, dl(ds_va, False), dev, T, mc), predict(model, dl(ds_te, False), dev, T, mc)
        m = metrics(rt); thr = select_threshold(rv["u"], rv["p"], rv["y"], a.target_risk); keep = rt["u"] <= thr
        m["coverage"] = float(keep.mean())
        key = f"{leads}|{force or 'both'}|featmiss={fr}"; res[key] = m; print(key, m)
        np.savez_compressed(f"{a.out}/{tag}_seed{a.seed}_{key.replace('|','__').replace('(','').replace(')','').replace(',','-')}.npz", **rt)
    for d in (ds_va, ds_te): d.set_eval_condition("12-lead")
    rv = predict(model, dl(ds_va, False), dev, T, mc)
    torch.save(dict(state=model.state_dict(), stats=ds_tr.stats, T=T, mc=mc,
                    tau=float(select_threshold(rv["u"], rv["p"], rv["y"], a.target_risk))), f"{a.out}/{tag}_seed{a.seed}_final.pt")
    res["cost"] = dict(params_M=sum(p.numel() for p in model.parameters()) / 1e6, temperature=T)
    json.dump(res, open(f"{a.out}/{tag}_seed{a.seed}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
