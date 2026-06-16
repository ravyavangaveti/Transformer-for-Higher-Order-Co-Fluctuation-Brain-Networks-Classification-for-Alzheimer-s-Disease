import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset


# ─────────────────────────────────────────────────────────────────
#  Utility functions
# ─────────────────────────────────────────────────────────────────

def zscore(ts_raw, eps=1e-6):
    mu  = ts_raw.mean(axis=0, keepdims=True)
    sig = ts_raw.std(axis=0,  keepdims=True)
    return (ts_raw - mu) / (sig + eps)


def compute_rss(signal_ts):
    """signal_ts: (M, T) → RSS (T,)"""
    return np.sqrt(np.sum(signal_ts ** 2, axis=0))


def rss_bin_means(signal_ts, K=5):
    """
    Sort T timepoints by RSS → K bins → mean per bin
    signal_ts: (M, T)
    returns:   (K, M)
    """
    T    = signal_ts.shape[1]
    rss  = compute_rss(signal_ts)
    sort = np.argsort(rss)
    bsz  = T // K
    bins = np.zeros((K, signal_ts.shape[0]), dtype=np.float32)
    for k in range(K):
        bin_t   = sort[k * bsz: (k + 1) * bsz]
        bins[k] = signal_ts[:, bin_t].mean(axis=1)
    return bins  # (K, M)


def compute_edge_fcc(ts_z, edge_idx, K=5):
    """
    eTS_ij(t) = z_i(t) * z_j(t) → RSS bins
    ts_z:     (T, N)
    edge_idx: (E, 2)
    returns:  (K, E)
    """
    ets = (ts_z[:, edge_idx[:, 0]] *
           ts_z[:, edge_idx[:, 1]]).T          # (E, T)
    return rss_bin_means(ets, K)               # (K, E)


def compute_triangle_fcc(ts_z, triangles, K=5):
    """
    triTS_ijk(t) = z_i(t) * z_j(t) * z_k(t) → RSS bins
    triangles: (T3, 3) array of triplet indices
    returns:   (K, T3)
    """
    i, j, k = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    tri_ts  = (ts_z[:, i] *
               ts_z[:, j] *
               ts_z[:, k]).T                   # (T3, T)
    return rss_bin_means(tri_ts, K)            # (K, T3)


def compute_scaffold_fcc(ts_z, edge_idx, K=5):
    """
    Simplified scaffold: weight each edge by how many
    connected triangles it participates in, then RSS bin.

    Returns: (K, E) — scaffold-weighted edge RSS bins
    """
    T, N   = ts_z.shape
    E      = edge_idx.shape[0]

    # Build adjacency set for fast lookup
    adj = set(map(tuple, edge_idx.tolist()))

    # Count triangles per edge
    scaffold_weight = np.zeros(E, dtype=np.float32)
    for idx, (i, j) in enumerate(edge_idx):
        count = 0
        for k in range(N):
            if k != i and k != j:
                if (i, k) in adj and (j, k) in adj:
                    count += 1
        scaffold_weight[idx] = count

    # Weight the edge co-fluctuation by scaffold weight
    ets = (ts_z[:, edge_idx[:, 0]] *
           ts_z[:, edge_idx[:, 1]]).T          # (E, T)

    # Multiply each edge timeseries by its scaffold weight
    scaffold_weight = scaffold_weight.reshape(-1, 1)
    ets_weighted    = ets * scaffold_weight    # (E, T)

    return rss_bin_means(ets_weighted, K)      # (K, E)


def build_graph(ts, k_neighbors=12, K_bins=5):
    """
    Build graph with all signal types:
    - Node:     FC matrix (116, 116)
    - Edge:     static FC + 5 RSS bins = (E, 6)
    - Triangle: mean triangle FCC per node = (116, K)
    - Scaffold: scaffold-weighted edge FCC per node = (116, K)
    """
    T, N = ts.shape
    ts_z = zscore(ts)

    # Static FC
    fc = np.corrcoef(ts_z.T)
    fc = np.nan_to_num(fc)

    # Build kNN edge index
    edge_index_list = []
    for i in range(N):
        fc_row    = np.abs(fc[i])
        fc_row[i] = -1
        top_k     = np.argsort(fc_row)[-k_neighbors:]
        for j in top_k:
            edge_index_list.append((i, j))
            edge_index_list.append((j, i))

    edges      = list(set(edge_index_list))
    edge_index = torch.tensor(edges, dtype=torch.long).T  # (2, E)
    edge_np    = np.array(edges)                           # (E, 2)

    src = edge_np[:, 0]
    dst = edge_np[:, 1]
    E   = len(edges)

    # ── Edge features: static FC + 5 RSS bins = 6 dim ──
    static_fc = fc[src, dst].reshape(-1, 1)               # (E, 1)
    bin_feats = compute_edge_fcc(ts_z, edge_np, K=K_bins)
    bin_feats = bin_feats.T                                # (E, K)
    edge_attr = np.concatenate(
        [static_fc, bin_feats], axis=1).astype(np.float32)# (E, 6)

    # ── Triangle features: aggregate to nodes ──
    adj_set   = set(map(tuple, edge_np.tolist()))
    triangles = []
    edge_set  = set(map(tuple, edge_np.tolist()))
    for i in range(N):
        nbrs_i = [j for j in range(N) if (i, j) in edge_set]
        for idx_a in range(len(nbrs_i)):
            for idx_b in range(idx_a + 1, len(nbrs_i)):
                j = nbrs_i[idx_a]
                k = nbrs_i[idx_b]
                if (j, k) in edge_set or (k, j) in edge_set:
                    triangles.append([i, j, k])

    if len(triangles) > 0:
        tri_np  = np.array(triangles)                      # (T3, 3)
        tri_fcc = compute_triangle_fcc(
            ts_z, tri_np, K=K_bins)                        # (K, T3)
        tri_fcc = tri_fcc.T                                # (T3, K)

        # Aggregate to nodes: mean of triangles per node
        tri_node = np.zeros((N, K_bins), dtype=np.float32)
        tri_cnt  = np.zeros(N, dtype=np.float32)
        for t_idx, (i, j, k) in enumerate(triangles):
            tri_node[i] += tri_fcc[t_idx]
            tri_node[j] += tri_fcc[t_idx]
            tri_node[k] += tri_fcc[t_idx]
            tri_cnt[i]  += 1
            tri_cnt[j]  += 1
            tri_cnt[k]  += 1
        tri_cnt  = np.maximum(tri_cnt, 1).reshape(-1, 1)
        tri_node = tri_node / tri_cnt                      # (N, K)
    else:
        tri_node = np.zeros((N, K_bins), dtype=np.float32)

    # ── Scaffold features: aggregate to nodes ──
    sca_fcc  = compute_scaffold_fcc(
        ts_z, edge_np, K=K_bins)                           # (K, E)
    sca_fcc  = sca_fcc.T                                   # (E, K)

    # Aggregate scaffold edge features to nodes
    sca_node = np.zeros((N, K_bins), dtype=np.float32)
    sca_cnt  = np.zeros(N, dtype=np.float32)
    for e_idx, (i, j) in enumerate(edge_np):
        sca_node[i] += sca_fcc[e_idx]
        sca_node[j] += sca_fcc[e_idx]
        sca_cnt[i]  += 1
        sca_cnt[j]  += 1
    sca_cnt  = np.maximum(sca_cnt, 1).reshape(-1, 1)
    sca_node = sca_node / sca_cnt                          # (N, K)

    # ── Node features: FC matrix ──
    node_features = torch.tensor(
        fc.astype(np.float32))                             # (116, 116)

    return (
        edge_index,
        torch.tensor(edge_attr,   dtype=torch.float32),   # (E, 6)
        node_features,                                     # (116, 116)
        torch.tensor(tri_node,    dtype=torch.float32),   # (116, K)
        torch.tensor(sca_node,    dtype=torch.float32),   # (116, K)
    )


# ─────────────────────────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────────────────────────

class Dataset_ADNI_Ablation(Dataset):
    def __init__(self, data_dir, label_file,
                 num_classes=2, k_neighbors=12, K_bins=5):
        self.data_dir    = data_dir
        self.label_file  = label_file
        self.num_classes = num_classes
        self.k_neighbors = k_neighbors
        self.K_bins      = K_bins
        self.data        = []
        self.labels      = []
        self._load()

    def _load(self):
        labels_df = pd.read_csv(self.label_file)
        if self.num_classes == 2:
            labels_id = {
                'CN': 0, 'SMC': 0, 'EMCI': 0, 'LMCI': 1, 'AD': 1}
        else:
            labels_id = {
                'CN': 0, 'SMC': 1, 'EMCI': 2, 'LMCI': 3, 'AD': 4}

        files = sorted([
            f for f in os.listdir(self.data_dir)
            if f.startswith('sub-') and f.endswith('_aal.txt')
        ])

        for fname in files:
            sid       = fname.split('_')[0][4:]
            label_row = labels_df[
                labels_df['subject_id'] == sid]
            if label_row.empty:
                continue
            label = label_row.iloc[0]['DX']
            if label not in labels_id:
                continue

            try:
                ts = np.loadtxt(
                    os.path.join(self.data_dir, fname))
            except Exception:
                continue

            if ts.ndim != 2 or ts.shape[1] != 116:
                continue
            if ts.shape[0] < 140:
                continue
            ts = ts[:140, :]

            print(f"  Processing {sid}...", end='\r')

            (edge_index, edge_attr,
             node_feat, tri_feat, sca_feat) = build_graph(
                ts,
                k_neighbors=self.k_neighbors,
                K_bins=self.K_bins)

            y     = torch.tensor(
                labels_id[label], dtype=torch.long)
            graph = Data(
                x          = node_feat,      # (116, 116)
                edge_index = edge_index,
                edge_attr  = edge_attr,      # (E, 6)
                tri_feat   = tri_feat,       # (116, K)
                sca_feat   = sca_feat,       # (116, K)
                y          = y
            )

            self.data.append(graph)
            self.labels.append(labels_id[label])

        print(f"\nLoaded {len(self.data)} subjects | "
              f"dist: {np.bincount(self.labels)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
