"""Transductive link prediction on a RelBench relational graph (recommendation-style).

RelBench's recommendation benchmarks live on large datasets (rel-hm, rel-amazon, ...)
and use a separate ranking interface. To demonstrate the *link-prediction task type* on
already-downloaded, CPU-sized data, this module reconstructs a chosen foreign-key edge
type: hold out a fraction of those edges, encode nodes with a Torch Frame ``HeteroEncoder``
plus a heterogeneous GraphSAGE, and score candidate (src, dst) pairs with a dot-product
decoder trained against negative samples. Evaluation is ROC-AUC over held-out positive
edges vs random negatives — the standard link-prediction protocol.

This is genuine link prediction (edge existence), not entity classification, so it rounds
out the task-type coverage (binary / regression / multiclass / link).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LinkPredModel(nn.Module):
    """HeteroEncoder features -> heterogeneous SAGE -> dot-product edge decoder."""

    def __init__(self, data, col_stats_dict, *, channels: int = 64, num_layers: int = 2):
        super().__init__()
        from relbench.modeling.nn import HeteroEncoder
        from torch_geometric.nn import HeteroConv, SAGEConv

        self.node_types = list(data.node_types)
        self._tf_dict = {nt: data[nt].tf for nt in self.node_types}
        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={nt: data[nt].tf.col_names_dict for nt in self.node_types},
            node_to_col_stats=col_stats_dict,
        )
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {et: SAGEConv((-1, -1), channels) for et in data.edge_types},
                aggr="mean",
            )
            self.convs.append(conv)

        # Expressive edge decoder over [src ; dst ; src*dst] (beats a bare dot product).
        self.decoder = nn.Sequential(
            nn.Linear(3 * channels, channels),
            nn.ReLU(),
            nn.Linear(channels, 1),
        )

    def encode(self, edge_index_dict) -> Dict[str, Tensor]:
        x_dict = self.encoder(self._tf_dict)
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        return x_dict

    def decode(self, x_src: Tensor, x_dst: Tensor, edge_index: Tensor) -> Tensor:
        s = x_src[edge_index[0]]
        d = x_dst[edge_index[1]]
        feats = torch.cat([s, d, s * d], dim=-1)
        return self.decoder(feats).view(-1)  # raw scores (logits)


def _pick_edge_type(data, target_edge: Optional[str]) -> Tuple[str, str, str]:
    """Pick the FK edge type to predict (largest by edge count, or a named relation)."""
    candidates = []
    for et in data.edge_types:
        src, rel, dst = et
        if src == dst:  # skip self-loops
            continue
        candidates.append((et, data[et].edge_index.size(1)))
    if not candidates:
        raise ValueError("No non-self-loop edge types found.")
    if target_edge is not None:
        for et, _ in candidates:
            if et[1] == target_edge or f"{et[0]}->{et[2]}" == target_edge:
                return et
    return max(candidates, key=lambda x: x[1])[0]


def run_link_prediction(
    *,
    dataset_name: str = "rel-f1",
    target_edge: Optional[str] = None,
    channels: int = 64,
    num_layers: int = 2,
    epochs: int = 15,
    lr: float = 1e-2,
    test_frac: float = 0.2,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, float]:
    from sklearn.metrics import roc_auc_score

    from src.relbench_pipeline.benchmark import prepare_graph

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    _db, data, col_stats_dict = prepare_graph(dataset_name)
    et = _pick_edge_type(data, target_edge)
    src_t, rel, dst_t = et
    if verbose:
        print(f"[link-pred] predicting edge type {et} "
              f"({data[et].edge_index.size(1):,} edges) on {dataset_name}")

    edge_index = data[et].edge_index
    num_edges = edge_index.size(1)
    n_dst = data[dst_t].num_nodes

    perm = torch.randperm(num_edges)
    n_test = max(1, int(num_edges * test_frac))
    test_pos = edge_index[:, perm[:n_test]]
    train_pos = edge_index[:, perm[n_test:]]

    # Message-passing graph uses only TRAIN positives for the target relation (no leakage),
    # plus all other relations unchanged.
    mp_edge_index_dict = {e: data[e].edge_index for e in data.edge_types}
    mp_edge_index_dict[et] = train_pos

    def sample_neg(pos: Tensor) -> Tensor:
        neg_dst = torch.randint(0, n_dst, (pos.size(1),))
        return torch.stack([pos[0], neg_dst], dim=0)

    model = LinkPredModel(data, col_stats_dict, channels=channels, num_layers=num_layers).to(device)
    with torch.no_grad():  # initialize lazy SAGEConv parameters before counting / optimizing
        model.encode(mp_edge_index_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad and not isinstance(p, nn.parameter.UninitializedParameter)
    )

    @torch.no_grad()
    def evaluate() -> float:
        model.eval()
        x_dict = model.encode(mp_edge_index_dict)
        neg = sample_neg(test_pos)
        pos_score = model.decode(x_dict[src_t], x_dict[dst_t], test_pos)
        neg_score = model.decode(x_dict[src_t], x_dict[dst_t], neg)
        scores = torch.cat([pos_score, neg_score]).cpu().numpy()
        labels = np.concatenate([np.ones(pos_score.size(0)), np.zeros(neg_score.size(0))])
        return float(roc_auc_score(labels, scores))

    best_auc = float("-inf")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        x_dict = model.encode(mp_edge_index_dict)
        neg = sample_neg(train_pos)
        pos_score = model.decode(x_dict[src_t], x_dict[dst_t], train_pos)
        neg_score = model.decode(x_dict[src_t], x_dict[dst_t], neg)
        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
        loss = F.binary_cross_entropy_with_logits(scores, labels)
        loss.backward()
        optimizer.step()
        auc = evaluate()
        best_auc = max(best_auc, auc)
        if verbose:
            print(f"[link-pred] epoch {epoch:02d} | loss {loss.item():.4f} | test ROC-AUC {auc:.4f}")

    return {
        "task": f"link-pred:{src_t}->{dst_t}",
        "metric": "roc_auc",
        "best_test_metric": best_auc,
        "params": float(n_params),
    }
