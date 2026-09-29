"""
CHARM: Clinically-Conditioned Hypernetwork Fusion of ECG and Patient Risk Factors
with Missing-Modality Robustness and Evidential Abstention.

Model definition and loss functions.
Author: G. Arul Dalton et al.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
# Anatomical territories used for the lead-graph prior
TERRITORY = {
    "inferior": ["II", "III", "aVF"],
    "lateral": ["I", "aVL", "V5", "V6"],
    "septal": ["V1", "V2"],
    "anterior": ["V3", "V4"],
    "right": ["aVR", "V1"],
}


def anatomical_adjacency():
    """12x12 prior: 1 for leads sharing a territory or the same plane, 0 otherwise."""
    idx = {l: i for i, l in enumerate(LEADS)}
    A = torch.zeros(12, 12)
    for leads in TERRITORY.values():
        for a in leads:
            for b in leads:
                A[idx[a], idx[b]] = 1.0
    limb, prec = LEADS[:6], LEADS[6:]
    for plane in (limb, prec):
        for a in plane:
            for b in plane:
                A[idx[a], idx[b]] = max(A[idx[a], idx[b]].item(), 0.5)
    # neighbouring precordial leads
    for i in range(6, 11):
        A[i, i + 1] = A[i + 1, i] = 1.0
    A.fill_diagonal_(1.0)
    return A


# --------------------------------------------------------------------------
# 1. Clinical (tabular) encoder with per-feature missing tokens
# --------------------------------------------------------------------------
class ClinicalEncoder(nn.Module):
    """Feature-tokenising transformer. Each numeric feature j gets its own
    linear embedding w_j * x_j + b_j; a missing value is replaced by a
    learned token m_j, so partially or fully missing records are handled
    natively (fully missing -> a learned population prior)."""

    def __init__(self, n_num, n_cat_levels, d=128, depth=2, heads=4, drop=0.1):
        super().__init__()
        self.n_num, self.n_cat = n_num, len(n_cat_levels)
        self.w = nn.Parameter(torch.randn(n_num, d) * 0.02)
        self.b = nn.Parameter(torch.zeros(n_num, d))
        self.cat_emb = nn.ModuleList([nn.Embedding(k, d) for k in n_cat_levels])
        self.miss = nn.Parameter(torch.randn(n_num + self.n_cat, d) * 0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, drop, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(d)

    def forward(self, x_num, x_cat, m_num, m_cat):
        # x_num: (B, n_num) float, NaNs allowed where m_num==0
        # m_*: 1 = observed, 0 = missing
        x_num = torch.nan_to_num(x_num, 0.0)
        tok_num = x_num.unsqueeze(-1) * self.w + self.b                     # (B,n_num,d)
        toks = [tok_num]
        for j, emb in enumerate(self.cat_emb):
            toks.append(emb(x_cat[:, j].clamp(min=0)).unsqueeze(1))
        tok = torch.cat(toks, 1)                                              # (B,F,d)
        m = torch.cat([m_num, m_cat], 1).unsqueeze(-1).float()                # (B,F,1)
        tok = m * tok + (1 - m) * self.miss.unsqueeze(0)
        tok = torch.cat([self.cls.expand(tok.size(0), -1, -1), tok], 1)
        return self.norm(self.tf(tok)[:, 0])                                   # (B,d)


# --------------------------------------------------------------------------
# 2. Hypernetwork: clinical embedding -> FiLM parameters for every ECG stage
# --------------------------------------------------------------------------
class FiLMHyperNet(nn.Module):
    def __init__(self, d, stage_channels, max_scale=0.5):
        super().__init__()
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2 * c))
                                    for c in stage_channels])
        for h in self.heads:                       # start as identity modulation
            nn.init.zeros_(h[-1].weight); nn.init.zeros_(h[-1].bias)
        self.max_scale = max_scale

    def forward(self, z_c):
        out = []
        for h in self.heads:
            g, b = h(z_c).chunk(2, -1)
            out.append((1.0 + self.max_scale * torch.tanh(g), b))   # bounded gamma
        return out


# --------------------------------------------------------------------------
# 3. Lead-wise 1D ResNet with FiLM conditioning
# --------------------------------------------------------------------------
class ResBlock1d(nn.Module):
    def __init__(self, cin, cout, stride, k=9, drop=0.1):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, k, stride, k // 2, bias=False)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, k, 1, k // 2, bias=False)
        self.b2 = nn.BatchNorm1d(cout)
        self.drop = nn.Dropout(drop)
        self.skip = (nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout))
                     if (cin != cout or stride != 1) else nn.Identity())

    def forward(self, x, gamma, beta):
        h = F.gelu(self.b1(self.c1(x)))
        h = self.drop(h)
        h = self.b2(self.c2(h))
        h = gamma.unsqueeze(-1) * h + beta.unsqueeze(-1)            # FiLM
        return F.gelu(h + self.skip(x))


class LeadwiseECGEncoder(nn.Module):
    """A shared encoder applied to each lead independently -> one token per lead.
    Sharing weights across leads is what makes arbitrary lead subsets
    (12, 6, 2, 1 lead) usable at inference time."""

    def __init__(self, channels=(32, 64, 128, 128), d=128):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(1, channels[0], 15, 2, 7, bias=False),
                                  nn.BatchNorm1d(channels[0]), nn.GELU())
        blocks, cin = [], channels[0]
        for c in channels:
            blocks.append(ResBlock1d(cin, c, 2)); cin = c
        self.blocks = nn.ModuleList(blocks)
        self.proj = nn.Linear(cin * 2, d)
        self.stage_channels = list(channels)

    def forward(self, ecg, films):
        # ecg: (B, 12, T)
        B, L, T = ecg.shape
        x = self.stem(ecg.reshape(B * L, 1, T))
        for blk, (g, b) in zip(self.blocks, films):
            # repeat each patient's FiLM parameters over its 12 leads
            x = blk(x, g.repeat_interleave(L, 0), b.repeat_interleave(L, 0))
        x = torch.cat([x.mean(-1), x.amax(-1)], -1)
        return self.proj(x).view(B, L, -1)                            # (B,12,d)


# --------------------------------------------------------------------------
# 4. Anatomical lead-graph attention
# --------------------------------------------------------------------------
class LeadGraphAttention(nn.Module):
    def __init__(self, d=128, heads=4, depth=2, drop=0.1):
        super().__init__()
        self.register_buffer("A", anatomical_adjacency())
        self.lead_emb = nn.Parameter(torch.randn(12, d) * 0.02)
        self.absent = nn.Parameter(torch.randn(d) * 0.02)
        self.prior_w = nn.Parameter(torch.ones(depth, heads))          # learnable prior strength
        self.heads, self.depth = heads, depth
        self.q = nn.ModuleList([nn.Linear(d, d) for _ in range(depth)])
        self.k = nn.ModuleList([nn.Linear(d, d) for _ in range(depth)])
        self.v = nn.ModuleList([nn.Linear(d, d) for _ in range(depth)])
        self.o = nn.ModuleList([nn.Linear(d, d) for _ in range(depth)])
        self.ff = nn.ModuleList([nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(),
                                               nn.Dropout(drop), nn.Linear(2 * d, d)) for _ in range(depth)])
        self.n1 = nn.ModuleList([nn.LayerNorm(d) for _ in range(depth)])
        self.pool = nn.Linear(d, 1)

    def forward(self, h, lead_mask):
        # h: (B,12,d); lead_mask: (B,12) 1=present
        B, L, d = h.shape
        lm = lead_mask.unsqueeze(-1).float()
        h = lm * h + (1 - lm) * self.absent + self.lead_emb
        key_bias = torch.where(lead_mask.bool(), 0.0, -1e4)[:, None, None, :]   # mask absent keys
        dh = d // self.heads
        for i in range(self.depth):
            x = self.n1[i](h)
            q = self.q[i](x).view(B, L, self.heads, dh).transpose(1, 2)
            k = self.k[i](x).view(B, L, self.heads, dh).transpose(1, 2)
            v = self.v[i](x).view(B, L, self.heads, dh).transpose(1, 2)
            att = q @ k.transpose(-1, -2) / math.sqrt(dh)
            att = att + self.prior_w[i].view(1, -1, 1, 1) * torch.log(self.A + 1e-3) + key_bias
            att = att.softmax(-1)
            h = h + self.o[i]((att @ v).transpose(1, 2).reshape(B, L, d))
            h = h + self.ff[i](h)
        w = self.pool(h).squeeze(-1).masked_fill(~lead_mask.bool(), -1e4).softmax(-1)  # (B,12)
        return (w.unsqueeze(-1) * h).sum(1), w                                         # z_e, lead importance


# --------------------------------------------------------------------------
# 5. Full CHARM model
# --------------------------------------------------------------------------
class CHARM(nn.Module):
    def __init__(self, n_num, n_cat_levels, n_classes, d=128, channels=(32, 64, 128, 128),
                 use_film=True, use_prior=True, static_film=False):
        super().__init__()
        self.use_film, self.static_film = use_film, static_film
        self.clin = ClinicalEncoder(n_num, n_cat_levels, d)
        self.ecg = LeadwiseECGEncoder(channels, d)
        self.hyper = FiLMHyperNet(d, self.ecg.stage_channels)
        self.graph = LeadGraphAttention(d)
        if static_film:                        # ablation: FiLM params not generated from z_c
            self.static = nn.ParameterList([nn.Parameter(torch.zeros(2, c)) for c in self.ecg.stage_channels])
        if not use_prior:                      # ablation: no anatomical prior
            self.graph.prior_w.data.zero_(); self.graph.prior_w.requires_grad_(False)
        # cross-modal latent imputers
        self.e2c = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.c2e = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        # fusion + evidential (Beta) head, one Beta per label (multi-label)
        self.fuse = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(0.2))
        self.evid = nn.Linear(d, 2 * n_classes)
        # class prototypes in fused space
        self.proto = nn.Parameter(torch.randn(n_classes, d) * 0.1)
        self.n_classes = n_classes

    def forward(self, ecg, lead_mask, x_num, x_cat, m_num, m_cat, has_ecg, has_clin):
        """has_ecg / has_clin: (B,) bool — whole-modality availability."""
        z_c = self.clin(x_num, x_cat, m_num, m_cat)
        if self.static_film:
            films = [(1.0 + 0.5 * torch.tanh(p[0]).expand(z_c.size(0), -1), p[1].expand(z_c.size(0), -1))
                     for p in self.static]
        elif self.use_film:
            films = self.hyper(z_c)
        else:                                   # ablation: identity FiLM -> late concatenation only
            films = [(torch.ones(z_c.size(0), c, device=z_c.device), torch.zeros(z_c.size(0), c, device=z_c.device))
                     for c in self.ecg.stage_channels]
        # when the ECG is entirely absent we still run the encoder on zeros but
        # replace its embedding by the imputation below (keeps batching simple)
        lm = lead_mask * has_ecg.unsqueeze(-1).float()
        lm_safe = torch.where(lm.sum(1, keepdim=True) > 0, lm, torch.ones_like(lm))
        z_e, lead_w = self.graph(self.ecg(ecg * lm.unsqueeze(-1), films), lm_safe)

        z_c_hat, z_e_hat = self.e2c(z_e), self.c2e(z_c)
        z_c_used = torch.where(has_clin.unsqueeze(-1), z_c, z_c_hat)
        z_e_used = torch.where(has_ecg.unsqueeze(-1), z_e, z_e_hat)
        z = self.fuse(torch.cat([z_e_used, z_c_used], -1))

        ev = F.softplus(self.evid(z)).view(-1, self.n_classes, 2) + 1.0
        alpha, beta = ev[..., 0], ev[..., 1]
        return dict(alpha=alpha, beta=beta, prob=alpha / (alpha + beta),
                    u=2.0 / (alpha + beta), z=z, z_c=z_c, z_e=z_e,
                    z_c_hat=z_c_hat, z_e_hat=z_e_hat, lead_w=lead_w)

    def proto_distance(self, z):
        return torch.cdist(z, self.proto)                            # (B,K)


# --------------------------------------------------------------------------
# 6. Losses
# --------------------------------------------------------------------------
def beta_evidential_loss(alpha, beta, y):
    """Expected BCE under Beta(alpha,beta)."""
    s = torch.digamma(alpha + beta)
    return (y * (s - torch.digamma(alpha)) + (1 - y) * (s - torch.digamma(beta))).mean()


def beta_kl_to_uniform(alpha, beta, y):
    """KL(Beta(a~,b~) || Beta(1,1)) after removing the evidence that supports the
    correct outcome — penalises only misleading evidence."""
    a = y + (1 - y) * alpha
    b = (1 - y) + y * beta
    lB = torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)
    kl = -lB + (a - 1) * torch.digamma(a) + (b - 1) * torch.digamma(b) - (a + b - 2) * torch.digamma(a + b)
    return kl.mean()


def imputation_loss(out, has_ecg, has_clin):
    both = (has_ecg & has_clin).float().unsqueeze(-1)
    if both.sum() == 0:
        return out["z"].new_zeros(())
    l1 = ((out["z_c_hat"] - out["z_c"].detach()) ** 2 * both).sum() / both.sum()
    l2 = ((out["z_e_hat"] - out["z_e"].detach()) ** 2 * both).sum() / both.sum()
    return l1 + l2


def consistency_loss(p_full, p_partial):
    """Binary KL(p_full || p_partial) per label; p_full is a stop-gradient teacher."""
    p, q = p_full.detach().clamp(1e-4, 1 - 1e-4), p_partial.clamp(1e-4, 1 - 1e-4)
    return (p * (p / q).log() + (1 - p) * ((1 - p) / (1 - q)).log()).mean()


def prototype_loss(model, z, y, margin=1.0):
    """Pull embeddings to the prototypes of their positive labels, push them
    at least `margin` away from prototypes of negative labels."""
    d = model.proto_distance(z)
    pos = (y * d ** 2).sum() / y.sum().clamp(min=1)
    neg = ((1 - y) * F.relu(margin - d) ** 2).sum() / (1 - y).sum().clamp(min=1)
    return pos + neg


def uncertainty_score(model, out, d_ref, omega=0.5):
    """Sample-level abstention score combining evidential and prototype uncertainty.
    d_ref: 95th percentile of the min prototype distance on the training set."""
    u_ev = out["u"].mean(-1)
    dmin = model.proto_distance(out["z"]).min(-1).values
    u_pr = 1.0 - torch.exp(-(dmin / d_ref) ** 2)
    return omega * u_ev + (1 - omega) * u_pr
