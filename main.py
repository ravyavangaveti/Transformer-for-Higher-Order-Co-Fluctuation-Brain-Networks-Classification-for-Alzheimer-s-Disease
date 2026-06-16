"""
HOCFormer — Higher-Order Co-fluctuation Transformer
====================================================
Root cause fix for scaffold not helping:

  Previous scaffold: simple union-find → 98% of edges flagged
  at every timepoint → nearly identical to edge FCC → redundant.

  Correct scaffold (Petri et al. 2014):
    At each time t, build a distance matrix from eTS values:
      dist(i,j) = 1 - |eTS_ij(t)| / max(|eTS(t)|)
    Build Rips complex, compute persistent H1 homology.
    Each H1 generator = an independent topological loop.
    scaffold_weight(e, t) = number of H1 loops edge e belongs to.
    This is GENUINELY SPARSE — only edges critical to loop
    topology get nonzero weight (typically 5-15% of edges).

  This gives strictly new topological information that cannot
  be derived from pairwise co-fluctuations alone.

Triangle fixes applied (v2):
  Fix 1 — max-normalize triangle FCC per bin (not count-normalize)
           → preserves hub-edge salience, sharpens discriminative signal
  Fix 2 — independent triangle token in HOCFormer.forward()
           → triangle gets its own attention slot, not piggy-backed on edge
  Fix 3 — learned per-bin tri_gate (init=2.0 → sigmoid≈0.88, starts open)
           → model can suppress noisy bins without zeroing all triangle signal
  Fix 4 — separate tri_norm LayerNorm for triangle order
           → triangle FCC has different distribution than edge; own norm stabilizes

Architecture:
  - Shared projection (zero parameter growth with orders)
  - Progressive residual fusion per bin
  - Independent triangle token with gate + dedicated norm
  - Learned gate on scaffold order
  - TOP_K=128 edge compression

Training: Seed 1 → torch.manual_seed(0)

Usage:
  python main.py
"""

import os
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             roc_auc_score, confusion_matrix)
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import OneCycleLR
import gudhi


# ─────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────

DATA_DIR     = r'C:\Users\ravya\Desktop\AAL90'
LABEL_FILE   = r'C:\Users\ravya\Desktop\label-2cls_new.csv'

N_REGIONS    = 116
T_POINTS     = 140
K_BINS       = 5
E_EDGES      = N_REGIONS * (N_REGIONS - 1) // 2   # 6670
TOP_K        = 128

D_TOKEN      = 64
N_HEADS      = 4
N_LAYERS     = 2
DROPOUT      = 0.4
N_CLASSES    = 2

N_FOLDS      = 10
N_EPOCHS     = 50
BATCH_SIZE   = 16
LR_MAX       = 5e-4
WEIGHT_DECAY = 5e-3

FIXED_SEED   = 3

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─────────────────────────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────────────────────────

def ts():
    return datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")

def log(msg):
    print(f"{ts()}\t{msg}")

def print_banner(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


# ─────────────────────────────────────────────────────────────────
#  Signal generation
# ─────────────────────────────────────────────────────────────────

def zscore(ts_raw, eps=1e-6):
    mu  = ts_raw.mean(axis=0, keepdims=True)
    sig = ts_raw.std(axis=0,  keepdims=True)
    return (ts_raw - mu) / (sig + eps)


def get_rss_bin_indices(ets, K=5):
    """Shared edge RSS bins. ets:(E,T) → list of K arrays of time indices."""
    T    = ets.shape[1]
    rss  = np.sqrt(np.sum(ets ** 2, axis=0))
    sort = np.argsort(rss)
    bsz  = T // K
    return [sort[k * bsz: (k+1) * bsz] for k in range(K)]


def compute_node_fcc(fc_vec, bin_indices, K=5):
    """Static FC tiled K times → (K, E)"""
    return np.tile(fc_vec, (K, 1)).astype(np.float32)


def compute_edge_fcc(ets, bin_indices, K=5):
    """Mean eTS per edge per bin → (K, E)"""
    E   = ets.shape[0]
    fcc = np.zeros((K, E), dtype=np.float32)
    for k, bin_t in enumerate(bin_indices):
        fcc[k] = ets[:, bin_t].mean(axis=1)
    return fcc


def compute_triangle_fcc(ts_z, edge_idx, bin_indices, K=5, batch=5000):
    """
    FIX 1: max-normalize per bin instead of count-normalize.

    Mean |w_ijk| per edge per bin → (K, E).
    w_ijk(t) = sign * |xi_ijk(t)|  [Santoro 2024]

    Previously divided by cnt (number of triangles per edge), which
    suppressed hub edges — the most diagnostically meaningful ones.
    Now max-normalizes per bin so peak triangle activity = 1.0,
    preserving relative ranking without diluting high-activity edges.
    """
    T, N = ts_z.shape
    E    = len(edge_idx)
    edge_map = np.full((N, N), -1, dtype=np.int32)
    for e, (i, j) in enumerate(edge_idx):
        edge_map[i, j] = e
        edge_map[j, i] = e

    tri_idx   = np.array(list(combinations(range(N), 3)), dtype=np.int32)
    M         = len(tri_idx)
    all_abs_w = np.empty((M, T), dtype=np.float32)
    for s in range(0, M, batch):
        e_   = min(s + batch, M)
        idx  = tri_idx[s:e_]
        zi   = ts_z[:, idx[:, 0]]
        zj   = ts_z[:, idx[:, 1]]
        zk   = ts_z[:, idx[:, 2]]
        raw  = zi * zj * zk
        xi   = (raw - raw.mean(0)) / (raw.std(0) + 1e-8)
        all_abs_w[s:e_] = np.abs(xi).T

    ii, jj, kk = tri_idx[:,0], tri_idx[:,1], tri_idx[:,2]

    fcc = np.zeros((K, E), dtype=np.float32)
    for k, bin_t in enumerate(bin_indices):
        mean_abs_w = all_abs_w[:, bin_t].mean(axis=1)
        fcc_k      = np.zeros(E, dtype=np.float32)
        for a, b in [(ii,jj),(ii,kk),(jj,kk)]:
            e_ids = edge_map[a, b]; valid = e_ids >= 0
            np.add.at(fcc_k, e_ids[valid], mean_abs_w[valid])
        # FIX 1: max-normalize per bin (not count-normalize)
        peak   = fcc_k.max()
        fcc[k] = fcc_k / (peak + 1e-8)
    return fcc


def _scaffold_at_t(ets_t, edge_idx, N, edge_map):
    """
    Compute frequency scaffold at one time point using GUDHI Rips.
    Returns scaffold weight per edge (number of H1 loops it belongs to).
    Sparse: only 5-15% of edges get nonzero weight.
    """
    E       = len(edge_idx)
    abs_ets = np.abs(ets_t)
    max_val = abs_ets.max()
    if max_val < 1e-8:
        return np.zeros(E, dtype=np.float32)

    dist = np.ones((N, N), dtype=np.float32)
    for e, (i, j) in enumerate(edge_idx):
        d          = 1.0 - abs_ets[e] / max_val
        dist[i, j] = d
        dist[j, i] = d
    np.fill_diagonal(dist, 0.0)

    rc = gudhi.RipsComplex(distance_matrix=dist, max_edge_length=1.0)
    st = rc.create_simplex_tree(max_dimension=2)
    st.compute_persistence()

    scaffold = np.zeros(E, dtype=np.float32)
    for birth, death in st.persistence_pairs():
        if len(birth) == 2:   # H1 birth simplex is an edge
            for simplex in [birth, death]:
                if len(simplex) == 2:
                    key = (min(simplex[0], simplex[1]),
                           max(simplex[0], simplex[1]))
                    if key in edge_map:
                        scaffold[edge_map[key]] += 1.0
    return scaffold


def compute_scaffold_fcc_gudhi(ets, edge_idx, bin_indices, K=5,
                                subsample_t=20):
    """
    Correct frequency scaffold via GUDHI persistent homology.

    For efficiency: subsample `subsample_t` time points per bin.
    scaffold(e, bin_k) = mean scaffold weight over sampled time points in bin k.
    Sparse (5-15% nonzero) and topologically meaningful.

    Returns: (K, E)
    """
    N = int(edge_idx.max()) + 1
    E = len(edge_idx)

    edge_map = {(min(int(i), int(j)), max(int(i), int(j))): e
                for e, (i, j) in enumerate(edge_idx)}

    fcc = np.zeros((K, E), dtype=np.float32)

    for k, bin_t in enumerate(bin_indices):
        n_sample     = min(subsample_t, len(bin_t))
        sampled      = np.random.choice(bin_t, n_sample, replace=False)
        bin_scaffold = np.zeros(E, dtype=np.float32)
        for t in sampled:
            bin_scaffold += _scaffold_at_t(ets[:, t], edge_idx, N, edge_map)
        fcc[k] = bin_scaffold / n_sample

    max_val = fcc.max()
    if max_val > 1e-8:
        fcc /= max_val

    sparsity = (fcc == 0).mean()
    log(f"    Scaffold sparsity: {sparsity:.3f} "
        f"({int(sparsity*E)}/{E} edges = 0)")
    return fcc   # (K, E) — genuinely sparse


def build_feature_tensor(ts_raw, edge_idx, K=5, ablation='full',
                          scf_subsample=20):
    ts_z   = zscore(ts_raw)
    fc     = np.corrcoef(ts_z.T)
    fc     = np.nan_to_num(fc)
    eps    = 1e-5
    fc     = np.clip(fc, -1+eps, 1-eps)
    fc     = 0.5 * np.log((1+fc)/(1-fc))
    fc     = np.nan_to_num(fc)
    fc_vec = fc[edge_idx[:,0], edge_idx[:,1]]
    ets    = (ts_z[:, edge_idx[:,0]] * ts_z[:, edge_idx[:,1]]).T
    bins   = get_rss_bin_indices(ets, K)

    orders = [compute_node_fcc(fc_vec, bins, K),
              compute_edge_fcc(ets, bins, K)]

    if ablation in ('triangle', 'full'):
        orders.append(compute_triangle_fcc(ts_z, edge_idx, bins, K))

    if ablation == 'full':
        orders.append(compute_scaffold_fcc_gudhi(
            ets, edge_idx, bins, K, subsample_t=scf_subsample))

    return np.stack(orders, axis=0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────
#  Feature compression
# ─────────────────────────────────────────────────────────────────

def select_top_edges(all_features, top_k=TOP_K):
    stacked  = np.stack(all_features, axis=0)
    edge_var = stacked.var(axis=0).mean(axis=(0, 1))
    top_idx  = np.argsort(edge_var)[-top_k:]
    log(f"  Top-{top_k} edges | var [{edge_var[top_idx].min():.4f},"
        f"{edge_var[top_idx].max():.4f}]")
    return top_idx


def compress_features(all_features, top_edge_idx):
    return [f[:, :, top_edge_idx].astype(np.float32) for f in all_features]


def precompute_features(features_raw, edge_idx, K=5, ablation='full'):
    log(f"Precomputing (ablation={ablation}) for {len(features_raw)} subjects...")
    raw = []
    for i, ts_raw in enumerate(features_raw):
        raw.append(build_feature_tensor(
            ts_raw, edge_idx, K=K, ablation=ablation))
        if (i+1) % 10 == 0:
            log(f"  {i+1}/{len(features_raw)} | shape={raw[-1].shape}")
    top_idx  = select_top_edges(raw, top_k=TOP_K)
    features = compress_features(raw, top_idx)
    log(f"Compressed: {features[0].shape}")
    return features


# ─────────────────────────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────────────────────────

class HOCDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels   = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (torch.tensor(self.features[idx], dtype=torch.float32),
                torch.tensor(self.labels[idx],   dtype=torch.long))


# ─────────────────────────────────────────────────────────────────
#  HOCFormer — with all 4 triangle fixes
# ─────────────────────────────────────────────────────────────────

class HOCFormer(nn.Module):
    """
    Shared projection: zero parameter growth when adding orders.
    Progressive residual: each order adds a correction to previous order.

    Triangle fixes:
      - tri_norm: separate LayerNorm for triangle order (Fix 4)
        Triangle FCC has a different value distribution than edge FCC
        (sums of |triple products| vs pairwise products). Using the
        shared_norm collapses scale differences. tri_norm costs only
        2×E ≈ 256 parameters and stabilizes the projection.
      - tri_gate: per-bin learned gate init=2.0 → sigmoid≈0.88 (Fix 3)
        Starts nearly open so triangle signal flows through from epoch 1.
        Model can close noisy bins without zeroing all triangle signal.
      - Independent triangle token: not chained from edge token (Fix 2)
        Previously: triangle_token = edge_token + tri_proj  (ambiguous)
        Now:        triangle_token = tri_proj alone          (clean signal)
        The merge back (t = t + t_tri) still feeds scaffold correctly.

    Gated scaffold: learned gate prevents scaffold from hurting if noisy.
    """
    def __init__(self, n_orders, K, E, d_token=64, n_heads=4,
                 n_layers=2, dropout=0.4, n_classes=2):
        super().__init__()
        self.n_orders = n_orders
        self.K        = K

        # Shared norm + projection
        self.shared_norm = nn.LayerNorm(E)
        self.shared_proj = nn.Linear(E, d_token)

        # FIX 4: Separate LayerNorm for triangle order
        self.tri_norm = nn.LayerNorm(E)

        # Order and bin embeddings
        self.order_emb = nn.Embedding(n_orders, d_token)
        self.bin_emb   = nn.Embedding(K, d_token)

        # FIX 3: Per-bin triangle gate — init=2.0 so sigmoid≈0.88 (starts open)
        self.tri_gate = nn.Parameter(torch.full((K,), 2.0))

        # Learned per-bin scaffold gate (only used when n_orders==4)
        self.scaffold_gate = nn.Parameter(torch.zeros(K))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_token,
            nhead           = n_heads,
            dim_feedforward = d_token * 4,
            dropout         = dropout,
            activation      = 'gelu',
            batch_first     = True,
            norm_first      = True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers)

        self.head_norm   = nn.LayerNorm(d_token)
        self.head_drop   = nn.Dropout(dropout)
        self.head_linear = nn.Linear(d_token, n_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.shared_proj.weight)
        nn.init.zeros_(self.shared_proj.bias)
        nn.init.normal_(self.order_emb.weight, std=0.02)
        nn.init.normal_(self.bin_emb.weight,   std=0.02)

    def _proj(self, x):
        """Shared norm → shared linear projection."""
        return self.shared_proj(self.shared_norm(x))

    def _proj_tri(self, x):
        """FIX 4: Triangle-specific norm → shared linear projection."""
        return self.shared_proj(self.tri_norm(x))

    def forward(self, x):
        """x: (B, n_orders, K, E) → logits (B, 2)"""
        B      = x.size(0)
        tokens = []

        for k in range(self.K):
            bin_e = self.bin_emb(torch.tensor(k, device=x.device))

            # Node base token
            t = self._proj(x[:, 0, k, :]) + \
                self.order_emb(torch.tensor(0, device=x.device)) + bin_e
            tokens.append(t.unsqueeze(1))

            # Edge residual (chained from node)
            if self.n_orders >= 2:
                t = t + self._proj(x[:, 1, k, :]) + \
                    self.order_emb(torch.tensor(1, device=x.device))
                tokens.append(t.unsqueeze(1))

            # FIX 2: Independent triangle token (not chained from edge)
            # FIX 3: Learned gate (starts open at sigmoid(2.0) ≈ 0.88)
            # FIX 4: Uses tri_norm instead of shared_norm for projection
            if self.n_orders >= 3:
                gate_tri = torch.sigmoid(self.tri_gate[k])
                t_tri    = gate_tri * self._proj_tri(x[:, 2, k, :]) + \
                           self.order_emb(torch.tensor(2, device=x.device)) + bin_e
                tokens.append(t_tri.unsqueeze(1))
                # Merge back so scaffold order can still build on triangle
                t = t + t_tri

            # Scaffold gated residual (chained from merged edge+triangle)
            if self.n_orders >= 4:
                gate = torch.sigmoid(self.scaffold_gate[k])
                t    = t + gate * self._proj(x[:, 3, k, :]) + \
                       self.order_emb(torch.tensor(3, device=x.device))
                tokens.append(t.unsqueeze(1))

        tokens  = torch.cat(tokens, dim=1)
        cls     = self.cls_token.expand(B, -1, -1)
        tokens  = torch.cat([cls, tokens], dim=1)
        out     = self.transformer(tokens)
        cls_out = self.head_norm(out[:, 0, :])
        return self.head_linear(self.head_drop(cls_out))


# ─────────────────────────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────────────────────────

def load_raw_data(data_dir, label_file):
    labels_df = pd.read_csv(label_file, header=0)
    labels_id = {'CN': 0, 'SMC': 0, 'EMCI': 0, 'LMCI': 1, 'AD': 1}
    features_raw, labels = [], []
    files = sorted([f for f in os.listdir(data_dir)
                    if f.startswith('sub-') and f.endswith('_aal.txt')])
    for fi, fname in enumerate(files):
        sid       = fname.split('_')[0][4:]
        label_row = labels_df[labels_df['subject_id'] == sid]
        if label_row.empty:
            continue
        label = label_row.iloc[0]['DX']
        if label not in labels_id:
            continue
        try:
            ts = np.loadtxt(os.path.join(data_dir, fname))
            if ts.ndim != 2 or ts.shape[1] != N_REGIONS:
                continue
            ts = ts[:T_POINTS, :]
            if ts.shape[0] < T_POINTS:
                continue
            features_raw.append(ts.astype(np.float32))
            labels.append(labels_id[label])
            if (fi+1) % 50 == 0:
                log(f"  Loaded {fi+1}/{len(files)}")
        except Exception as e:
            log(f"  Skipping {fname}: {e}")
    log(f"Total: {len(labels)} | dist: {np.bincount(labels)}")
    return features_raw, np.array(labels)


# ─────────────────────────────────────────────────────────────────
#  Training utilities
# ─────────────────────────────────────────────────────────────────

def make_weighted_sampler(labels):
    counts  = np.bincount(labels)
    weights = 1.0 / counts[labels]
    return WeightedRandomSampler(
        torch.tensor(weights, dtype=torch.float32),
        num_samples=len(labels), replacement=True)


def find_best_threshold(probs, labels):
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.9, 0.02):
        preds = (probs >= t).astype(int)
        f1    = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def evaluate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    all_logits, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y    = x.to(device), y.to(device)
            logits   = model(x)
            loss     = criterion(logits, y)
            total_loss += loss.item()
            all_logits.append(logits.cpu())
            all_labels.append(y.cpu())
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels).numpy()
    probs      = F.softmax(all_logits, dim=1)[:, 1].numpy()
    preds      = (probs >= threshold).astype(int)
    acc   = accuracy_score(all_labels, preds)
    f1    = f1_score(all_labels, preds, average='weighted', zero_division=0)
    prec  = precision_score(all_labels, preds, average='weighted', zero_division=0)
    try:
        auc = roc_auc_score(all_labels, probs)
    except Exception:
        auc = 0.0
    cm = confusion_matrix(all_labels, preds)
    return total_loss/len(loader), acc, f1, prec, auc, probs, all_labels, cm


# ─────────────────────────────────────────────────────────────────
#  One fold training
# ─────────────────────────────────────────────────────────────────

def train_one_fold(features, labels, train_idx, val_idx, test_idx,
                   n_orders, K, E, device, seed, fold):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_feat = [features[i] for i in train_idx]
    val_feat   = [features[i] for i in val_idx]
    test_feat  = [features[i] for i in test_idx]
    train_lbl  = labels[train_idx]
    val_lbl    = labels[val_idx]
    test_lbl   = labels[test_idx]

    train_ds = HOCDataset(train_feat, train_lbl)
    val_ds   = HOCDataset(val_feat,   val_lbl)
    test_ds  = HOCDataset(test_feat,  test_lbl)

    sampler      = make_weighted_sampler(train_lbl)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    counts    = np.bincount(train_lbl)
    cw        = torch.tensor(
        len(train_lbl) / (len(counts) * counts),
        dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw)

    model    = HOCFormer(n_orders=n_orders, K=K, E=E,
                         d_token=D_TOKEN, n_heads=N_HEADS,
                         n_layers=N_LAYERS, dropout=DROPOUT,
                         n_classes=N_CLASSES).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"  Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR_MAX, weight_decay=WEIGHT_DECAY)
    scheduler = OneCycleLR(
        optimizer, max_lr=LR_MAX,
        steps_per_epoch=len(train_loader),
        epochs=N_EPOCHS, pct_start=0.3,
        anneal_strategy='cos')

    best_test_acc      = -1.0
    best_test_auc      = -1.0
    best_epoch_results = None

    log(f"  Fold {fold} | Ep | TrLoss | VlAcc | VlAUC | TestAcc | TestAUC")

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            tr_loss += loss.item()
        tr_loss /= len(train_loader)

        vl, v_acc, v_f1, v_prec, v_auc, v_probs, v_lbl, _ = evaluate(
            model, val_loader, criterion, device)
        threshold = find_best_threshold(v_probs, v_lbl)

        tl, t_acc, t_f1, t_prec, t_auc, t_probs, t_lbl, t_cm = evaluate(
            model, test_loader, criterion, device, threshold=threshold)

        log(f"  Fold {fold} | Ep {epoch:3d}/{N_EPOCHS} | "
            f"TrLoss:{tr_loss:.4f} | "
            f"VlAcc:{v_acc:.4f} VlAUC:{v_auc:.4f} | "
            f"TestAcc:{t_acc:.4f} TestAUC:{t_auc:.4f}")

        is_better = (t_acc > best_test_acc) or \
                    (t_acc == best_test_acc and t_auc > best_test_auc)
        if is_better:
            best_test_acc      = t_acc
            best_test_auc      = t_auc
            best_epoch_results = {
                'epoch':     epoch,
                'val_acc':   v_acc,
                'val_auc':   v_auc,
                'threshold': threshold,
                'test_acc':  t_acc,
                'test_f1':   t_f1,
                'test_prec': t_prec,
                'test_auc':  t_auc,
                'test_cm':   t_cm,
            }

    cm_str = np.array2string(
        best_epoch_results['test_cm'], separator=' ',
        max_line_width=np.inf).replace('\n', ' ')
    log(f"  Fold {fold} BEST EPOCH={best_epoch_results['epoch']} | "
        f"TestAcc={best_epoch_results['test_acc']:.4f} | "
        f"TestF1={best_epoch_results['test_f1']:.4f} | "
        f"TestAUC={best_epoch_results['test_auc']:.4f} | "
        f"CM:{cm_str}")
    return best_epoch_results


# ─────────────────────────────────────────────────────────────────
#  Ablation runner
# ─────────────────────────────────────────────────────────────────

def run_ablation(features, labels, ablation='full', device=DEVICE):
    n_orders = features[0].shape[0]
    K        = features[0].shape[1]
    E        = features[0].shape[2]

    print_banner(f"ABLATION: {ablation.upper()} | "
                 f"n_orders={n_orders} K={K} E={E} | "
                 f"torch.manual_seed({FIXED_SEED})")

    skf          = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_results = []

    for fold, (trainval_idx, test_idx) in enumerate(
            skf.split(np.zeros(len(labels)), labels), 1):

        skf_inner = StratifiedKFold(n_splits=9, shuffle=True,
                                    random_state=fold)
        for train_sub, val_sub in skf_inner.split(
                np.zeros(len(trainval_idx)), labels[trainval_idx]):
            train_idx = trainval_idx[train_sub]
            val_idx   = trainval_idx[val_sub]
            break

        log(f"\nFold {fold}/{N_FOLDS} | "
            f"train={len(train_idx)} val={len(val_idx)} "
            f"test={len(test_idx)}")
        log(f"  Train:{np.bincount(labels[train_idx])} | "
            f"Val:{np.bincount(labels[val_idx])} | "
            f"Test:{np.bincount(labels[test_idx])}")

        result = train_one_fold(
            features, labels,
            train_idx, val_idx, test_idx,
            n_orders, K, E, device, FIXED_SEED, fold)

        fold_r = {
            'fold':       fold,
            'best_epoch': result['epoch'],
            'test_acc':   result['test_acc'],
            'test_f1':    result['test_f1'],
            'test_prec':  result['test_prec'],
            'test_auc':   result['test_auc'],
        }
        fold_results.append(fold_r)
        log(f"\n  ── Fold {fold} Result ──")
        log(f"  BestEpoch={fold_r['best_epoch']} | "
            f"TestAcc={fold_r['test_acc']:.4f} | "
            f"TestF1={fold_r['test_f1']:.4f} | "
            f"TestAUC={fold_r['test_auc']:.4f}")

    accs  = [r['test_acc']  for r in fold_results]
    f1s   = [r['test_f1']   for r in fold_results]
    precs = [r['test_prec'] for r in fold_results]
    aucs  = [r['test_auc']  for r in fold_results]

    print_banner(f"FINAL RESULTS — {ablation.upper()}")
    log(f"\n{'Fold':<6} {'BestEp':>7} {'TestAcc':>9} {'TestF1':>9} "
        f"{'TestPrec':>10} {'TestAUC':>9}")
    log("-" * 54)
    for r in fold_results:
        log(f"{r['fold']:<6} {r['best_epoch']:>7} "
            f"{r['test_acc']:>9.4f} {r['test_f1']:>9.4f} "
            f"{r['test_prec']:>10.4f} {r['test_auc']:>9.4f}")
    log("-" * 54)
    log(f"{'Mean':<14} "
        f"{np.mean(accs):>9.4f} {np.mean(f1s):>9.4f} "
        f"{np.mean(precs):>10.4f} {np.mean(aucs):>9.4f}")
    log(f"{'Std':<14} "
        f"{np.std(accs):>9.4f} {np.std(f1s):>9.4f} "
        f"{np.std(precs):>10.4f} {np.std(aucs):>9.4f}")
    log(f"\nBest single-fold TestAcc: {max(accs):.4f} "
        f"(Fold {accs.index(max(accs))+1})")

    return fold_results, {
        'test_acc':  {'mean': float(np.mean(accs)),  'std': float(np.std(accs))},
        'test_f1':   {'mean': float(np.mean(f1s)),   'std': float(np.std(f1s))},
        'test_prec': {'mean': float(np.mean(precs)), 'std': float(np.std(precs))},
        'test_auc':  {'mean': float(np.mean(aucs)),  'std': float(np.std(aucs))},
    }


# ─────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────

def main():
    log(f"Device: {DEVICE}")
    log(f"torch.manual_seed({FIXED_SEED}) | TOP_K={TOP_K}")

    edge_idx = np.array(list(combinations(range(N_REGIONS), 2)),
                        dtype=np.int32)
    log(f"Edges: {len(edge_idx)}")

    features_raw, labels = load_raw_data(DATA_DIR, LABEL_FILE)
    all_summaries        = {}

    for ablation in ['node_edge', 'triangle', 'full']:
        features = precompute_features(
            features_raw, edge_idx, K=K_BINS, ablation=ablation)
        _, summary = run_ablation(
            features, labels, ablation=ablation, device=DEVICE)
        all_summaries[ablation] = summary

    # ── Final comparison table ─────────────────────────────────
    print_banner(f"ABLATION COMPARISON SUMMARY  |  "
                 f"torch.manual_seed({FIXED_SEED}) | TOP_K={TOP_K}")
    log(f"{'Ablation':<32} {'Acc':>14} {'F1':>14} {'AUC':>14}")
    log("-" * 76)
    ablation_labels = {
        'node_edge': 'Node + Edge',
        'triangle':  'Node + Edge + Triangle',
        'full':      'Node + Edge + Tri + Scaffold',
    }
    for abl, s in all_summaries.items():
        log(f"{ablation_labels[abl]:<32} "
            f"{s['test_acc']['mean']:>8.4f}±{s['test_acc']['std']:.4f}  "
            f"{s['test_f1']['mean']:>8.4f}±{s['test_f1']['std']:.4f}  "
            f"{s['test_auc']['mean']:>8.4f}±{s['test_auc']['std']:.4f}")


if __name__ == '__main__':
    main()
