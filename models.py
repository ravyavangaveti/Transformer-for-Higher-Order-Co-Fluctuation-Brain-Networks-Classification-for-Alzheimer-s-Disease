import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────
# AblationTransformer
#
# 4 variants controlled by 'mode':
#   'node'              — node features only
#   'node_edge'         — node + edge features
#   'node_edge_tri'     — node + edge + triangle
#   'node_edge_tri_sca' — node + edge + triangle + scaffold
#
# Each brain region = one token
# Token = Linear(input_dim → d_model)
# Vanilla Transformer (no Yeo7 ordering)
# CLS → MLP → CN/AD
# ─────────────────────────────────────────────────────────────────

class AblationTransformer(nn.Module):
    def __init__(self, mode='node_edge',
                 node_dim=116, edge_dim=6, tri_dim=5, sca_dim=5,
                 d_model=256, n_heads=4, n_layers=3,
                 dropout=0.5, n_classes=2, n_regions=116):
        super().__init__()
        self.mode      = mode
        self.n_regions = n_regions
        self.d_model   = d_model

        # Compute input dim based on mode
        if mode == 'node':
            token_dim = node_dim                                  # 116
        elif mode == 'node_edge':
            token_dim = node_dim + edge_dim                       # 122
        elif mode == 'node_edge_tri':
            token_dim = node_dim + edge_dim + tri_dim             # 127
        elif mode == 'node_edge_tri_sca':
            token_dim = node_dim + edge_dim + tri_dim + sca_dim   # 132
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.token_norm = nn.LayerNorm(token_dim)
        self.token_proj = nn.Linear(token_dim, d_model)
        self.region_emb = nn.Embedding(n_regions, d_model)

        self.cls_token  = nn.Parameter(
            torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = n_heads,
            dim_feedforward = d_model * 4,
            dropout         = dropout,
            activation      = 'gelu',
            batch_first     = True,
            norm_first      = True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.token_proj.weight)
        nn.init.zeros_(self.token_proj.bias)
        nn.init.normal_(self.region_emb.weight, std=0.02)

    def forward(self, data):
        x          = data.x            # (B*N, 116)
        edge_index = data.edge_index   # (2, E)
        edge_attr  = data.edge_attr    # (E, 6)
        batch      = data.batch        # (B*N,)

        B = batch.max().item() + 1
        N = self.n_regions

        # ── Always: aggregate edge features to nodes ──
        if self.mode != 'node':
            dst       = edge_index[1]
            agg_edges = torch.zeros(
                B * N, edge_attr.size(1), device=x.device)
            count     = torch.zeros(
                B * N, 1, device=x.device)
            agg_edges.scatter_add_(
                0,
                dst.unsqueeze(1).expand(
                    -1, edge_attr.size(1)),
                edge_attr)
            count.scatter_add_(
                0,
                dst.unsqueeze(1),
                torch.ones(dst.size(0), 1, device=x.device))
            count     = count.clamp(min=1)
            agg_edges = agg_edges / count  # (B*N, 6)

        # ── Build tokens based on mode ──
        if self.mode == 'node':
            tokens = x                                        # (B*N, 116)

        elif self.mode == 'node_edge':
            tokens = torch.cat([x, agg_edges], dim=-1)       # (B*N, 122)

        elif self.mode == 'node_edge_tri':
            tri    = data.tri_feat                            # (B*N, K)
            tokens = torch.cat(
                [x, agg_edges, tri], dim=-1)                  # (B*N, 127)

        elif self.mode == 'node_edge_tri_sca':
            tri    = data.tri_feat                            # (B*N, K)
            sca    = data.sca_feat                            # (B*N, K)
            tokens = torch.cat(
                [x, agg_edges, tri, sca], dim=-1)             # (B*N, 132)

        # ── Project + position embedding ──
        tokens  = self.token_norm(tokens)
        tokens  = self.token_proj(tokens)                     # (B*N, d_model)
        pos_ids = torch.arange(
            N, device=x.device).repeat(B)
        tokens  = tokens + self.region_emb(pos_ids)
        tokens  = tokens.view(B, N, self.d_model)             # (B, 116, d_model)

        # ── CLS + Transformer ──
        cls     = self.cls_token.expand(B, -1, -1)
        tokens  = torch.cat([cls, tokens], dim=1)             # (B, 117, d_model)
        out     = self.transformer(tokens)
        cls_out = out[:, 0, :]                                # (B, d_model)

        logits  = self.head(cls_out)
        return logits, cls_out


def get_ablation_model(mode, args):
    """
    Factory function — instantiate AblationTransformer from argparse args.

    Usage:
        model = get_ablation_model('node_edge_tri_sca', args)
    """
    return AblationTransformer(
        mode      = mode,
        node_dim  = 116,
        edge_dim  = 6,
        tri_dim   = 5,
        sca_dim   = 5,
        d_model   = args.hidden_dim,
        n_heads   = 4,
        n_layers  = args.n_layers,
        dropout   = args.dropout,
        n_classes = args.num_classes,
        n_regions = 116
    )
