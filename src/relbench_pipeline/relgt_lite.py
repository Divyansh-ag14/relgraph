"""RelGT-lite: a from-scratch, CPU-runnable subset of the Relational Graph Transformer.

This re-implements the *core ideas* of RelGT (Dwivedi et al., 2025, arXiv:2505.10960)
inside our own pipeline, rather than importing the official repo. It is meant as a
demonstration of understanding the architecture, validated against the official model.

What we faithfully implement
----------------------------
1. **Multi-element tokenization (5 elements).** Each node in a seed's sampled
   neighborhood becomes a token built from:
     (a) node *features*      - a TorchFrame ``HeteroEncoder`` trained end-to-end
     (b) node *type*          - learned embedding over table types
     (c) *hop* distance       - learned embedding over BFS hop to the seed
     (d) *time*               - sinusoidal encoding of (seed_time - node_time)
     (e) local *structure*    - Random-Walk Structural Encoding (RWSE) on the
                                token-induced subgraph (our stand-in for GNN-PE)
2. **Local attention** over the K sampled tokens via a Transformer encoder; the seed
   token (position 0) is the readout.
3. **Global attention to learnable centroids** - the seed representation attends over
   a learnable centroid table, capturing database-wide context beyond the local sample.

Documented simplifications vs the paper (kept small so it runs on a laptop CPU)
------------------------------------------------------------------------------
* The feature encoder is trained end-to-end, but we encode the *whole* node set each
  forward pass and gather token features, instead of the paper's per-token inline
  encoding with sampled mini-batches. (Fine because RelBench's ``rel-f1`` is small.)
* Global centroids are plain learnable parameters with softmax attention, instead of
  the paper's EMA K-Means vector-quantized codebook (``VectorQuantizerEMA``).
* RWSE is used as the structural element instead of a full GNN positional encoder.
* Single RelGT layer; seeds are subsampled for CPU runs.

The point: same architectural skeleton (5-token + local/global attention), trained and
compared against both the official RelGT and our HeteroGraphSAGE baseline.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# --------------------------------------------------------------------------------------
# Tokenizer
# --------------------------------------------------------------------------------------
@dataclass
class TokenBatch:
    """Padded token tensors for a set of seeds. K = max tokens per seed."""

    global_ids: Tensor  # [B, K]  index into the flat node-feature table
    type_ids: Tensor  # [B, K]   node type id (num_types = pad)
    hops: Tensor  # [B, K]        BFS hop to seed (max_hop+1 = pad)
    time_delta: Tensor  # [B, K]  (seed_time - node_time), >= 0, normalized
    rwse: Tensor  # [B, K, P]     random-walk structural encoding
    pad_mask: Tensor  # [B, K]    True where padded (ignored by attention)
    labels: Optional[Tensor]  # [B]


class RelGraphTokenizer:
    """Builds RelGT-style token sequences from a HeteroData relational graph.

    Adjacency is materialized once (undirected). For each seed we BFS up to
    ``max_hop`` hops, respecting temporal validity (neighbor time <= seed time),
    and keep up to K tokens (seed first).
    """

    def __init__(self, data, *, max_hop: int = 2, rwse_steps: int = 4):
        self.data = data
        self.max_hop = max_hop
        self.rwse_steps = rwse_steps

        self.node_types: List[str] = list(data.node_types)
        self.type_to_idx = {nt: i for i, nt in enumerate(self.node_types)}
        self.num_types = len(self.node_types)

        # Flat node id space: global_id = offset[type] + local_idx.
        self.offset: Dict[str, int] = {}
        running = 0
        for nt in self.node_types:
            self.offset[nt] = running
            running += data[nt].num_nodes
        self.total_nodes = running

        # Per-node time (NaN -> +inf so it never violates the temporal filter as a seed,
        # and is treated as "always allowed" neighbor only if <= seed time).
        self.node_time: Dict[str, Optional[np.ndarray]] = {}
        for nt in self.node_types:
            if "time" in data[nt]:
                self.node_time[nt] = data[nt].time.cpu().numpy()
            else:
                self.node_time[nt] = None

        self._build_adjacency()

    def _build_adjacency(self) -> None:
        # adj[(type_idx, local_idx)] -> list of (type_idx, local_idx), undirected.
        self.adj: Dict[Tuple[int, int], List[Tuple[int, int]]] = defaultdict(list)
        for (src_t, _rel, dst_t) in self.data.edge_types:
            si, di = self.type_to_idx[src_t], self.type_to_idx[dst_t]
            ei = self.data[(src_t, _rel, dst_t)].edge_index
            src = ei[0].cpu().numpy()
            dst = ei[1].cpu().numpy()
            for s, d in zip(src.tolist(), dst.tolist()):
                self.adj[(si, s)].append((di, d))
                self.adj[(di, d)].append((si, s))

    def _node_time(self, type_idx: int, local_idx: int) -> Optional[float]:
        arr = self.node_time[self.node_types[type_idx]]
        if arr is None:
            return None
        return float(arr[local_idx])

    def _sample_one(
        self, seed_type_idx: int, seed_idx: int, seed_time: Optional[float], K: int
    ) -> List[Tuple[int, int, int]]:
        """Return [(type_idx, local_idx, hop)] for up to K tokens, seed first."""
        visited = {(seed_type_idx, seed_idx)}
        order: List[Tuple[int, int, int]] = [(seed_type_idx, seed_idx, 0)]
        frontier = [(seed_type_idx, seed_idx)]
        for hop in range(1, self.max_hop + 1):
            nxt: List[Tuple[int, int]] = []
            for node in frontier:
                for nb in self.adj.get(node, ()):  # (type_idx, local_idx)
                    if nb in visited:
                        continue
                    if seed_time is not None:
                        nt = self._node_time(nb[0], nb[1])
                        if nt is not None and nt > seed_time:
                            continue
                    visited.add(nb)
                    order.append((nb[0], nb[1], hop))
                    nxt.append(nb)
                    if len(order) >= K:
                        return order
            frontier = nxt
            if not frontier:
                break
        return order

    def _rwse(self, token_nodes: List[Tuple[int, int, int]], K: int) -> np.ndarray:
        """Random-walk structural encoding: landing-return probs over P steps."""
        n = len(token_nodes)
        node_to_pos = {(t, i): p for p, (t, i, _h) in enumerate(token_nodes)}
        A = np.zeros((n, n), dtype=np.float64)
        for p, (t, i, _h) in enumerate(token_nodes):
            for nb in self.adj.get((t, i), ()):  # (type_idx, local_idx)
                q = node_to_pos.get(nb)
                if q is not None:
                    A[p, q] = 1.0
        A += np.eye(n)  # self-loops
        deg = A.sum(axis=1, keepdims=True)
        deg[deg == 0] = 1.0
        P = A / deg
        out = np.zeros((K, self.rwse_steps), dtype=np.float32)
        cur = P.copy()
        for s in range(self.rwse_steps):
            diag = np.diagonal(cur)
            out[:n, s] = diag.astype(np.float32)
            cur = cur @ P
        return out

    def build(
        self,
        seed_type: str,
        seed_idxs: Tensor,
        seed_times: Optional[Tensor],
        targets: Optional[Tensor],
        *,
        K: int,
    ) -> TokenBatch:
        seed_type_idx = self.type_to_idx[seed_type]
        B = len(seed_idxs)

        global_ids = np.zeros((B, K), dtype=np.int64)
        type_ids = np.full((B, K), self.num_types, dtype=np.int64)  # pad = num_types
        hops = np.full((B, K), self.max_hop + 1, dtype=np.int64)  # pad = max_hop+1
        time_delta = np.zeros((B, K), dtype=np.float32)
        rwse = np.zeros((B, K, self.rwse_steps), dtype=np.float32)
        pad_mask = np.ones((B, K), dtype=bool)

        seed_idx_list = seed_idxs.cpu().numpy().tolist()
        seed_time_list = (
            seed_times.cpu().numpy().tolist() if seed_times is not None else [None] * B
        )

        for b in range(B):
            st = seed_time_list[b]
            tokens = self._sample_one(seed_type_idx, int(seed_idx_list[b]), st, K)
            rwse[b] = self._rwse(tokens, K)
            for p, (t_idx, l_idx, hop) in enumerate(tokens):
                global_ids[b, p] = self.offset[self.node_types[t_idx]] + l_idx
                type_ids[b, p] = t_idx
                hops[b, p] = hop
                pad_mask[b, p] = False
                if st is not None:
                    nt = self._node_time(t_idx, l_idx)
                    if nt is not None:
                        time_delta[b, p] = max(0.0, float(st) - nt)

        # Normalize time deltas (log1p compresses the long tail of seconds/days).
        time_delta = np.log1p(time_delta)
        tmax = time_delta.max()
        if tmax > 0:
            time_delta = time_delta / tmax

        return TokenBatch(
            global_ids=torch.from_numpy(global_ids),
            type_ids=torch.from_numpy(type_ids),
            hops=torch.from_numpy(hops),
            time_delta=torch.from_numpy(time_delta),
            rwse=torch.from_numpy(rwse),
            pad_mask=torch.from_numpy(pad_mask),
            labels=(targets if targets is not None else None),  # cast per task at train time
        )


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
class SinusoidalTime(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(dim, dim)

    def forward(self, t: Tensor) -> Tensor:  # t: [B, K]
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float32)
            * (-math.log(10000.0) / max(half - 1, 1))
        )
        ang = t.unsqueeze(-1) * freqs  # [B, K, half]
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # [B, K, 2*half]
        if emb.size(-1) < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.size(-1)))
        return self.proj(emb)


class RelGTLite(nn.Module):
    def __init__(
        self,
        *,
        data,
        col_stats_dict,
        num_types: int,
        max_hop: int,
        rwse_steps: int,
        channels: int = 64,
        heads: int = 4,
        local_layers: int = 2,
        num_centroids: int = 256,
        dropout: float = 0.1,
        use_global: bool = True,
        out_channels: int = 1,
    ):
        super().__init__()
        from relbench.modeling.nn import HeteroEncoder

        self.channels = channels
        self.use_global = use_global
        self.out_channels = out_channels

        # Trainable TorchFrame feature encoder; we encode only the nodes present in
        # each batch (the paper's sampled encoding), then gather token features.
        self.node_types = list(data.node_types)
        self._tf_dict = {nt: data[nt].tf for nt in self.node_types}
        self._num_nodes = {nt: data[nt].num_nodes for nt in self.node_types}
        self._offsets: Dict[str, int] = {}
        running = 0
        for nt in self.node_types:
            self._offsets[nt] = running
            running += self._num_nodes[nt]
        self.total_nodes = running

        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={nt: data[nt].tf.col_names_dict for nt in self.node_types},
            node_to_col_stats=col_stats_dict,
        )
        self.feat_proj = nn.Linear(channels, channels)

        # +1 row for pad index on type/hop embeddings.
        self.type_emb = nn.Embedding(num_types + 1, channels, padding_idx=num_types)
        self.hop_emb = nn.Embedding(max_hop + 2, channels, padding_idx=max_hop + 1)
        self.time_enc = SinusoidalTime(channels)
        self.rwse_proj = nn.Linear(rwse_steps, channels)

        self.in_mix = nn.Sequential(
            nn.Linear(5 * channels, 2 * channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * channels, channels),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=heads,
            dim_feedforward=2 * channels,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.local = nn.TransformerEncoder(enc_layer, num_layers=local_layers)

        if use_global:
            self.centroids = nn.Parameter(torch.randn(num_centroids, channels) * 0.02)
            self.q_proj = nn.Linear(channels, channels)
            self.k_proj = nn.Linear(channels, channels)
            self.v_proj = nn.Linear(channels, channels)
            self.global_norm = nn.LayerNorm(channels)

        self.head = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels, out_channels),
        )

    def _encode_token_features(self, batch: TokenBatch, device) -> Tensor:
        """Encode only the unique nodes present in this batch -> [B, K, D]."""
        B, K = batch.global_ids.shape
        gids = batch.global_ids.to(device).reshape(-1)
        uniq, inv = torch.unique(gids, return_inverse=True)
        out = torch.zeros(uniq.size(0), self.channels, device=device)
        for nt in self.node_types:
            lo = self._offsets[nt]
            hi = lo + self._num_nodes[nt]
            mask = (uniq >= lo) & (uniq < hi)
            if not mask.any():
                continue
            local = (uniq[mask] - lo).cpu()
            sub_tf = self._tf_dict[nt][local]
            out[mask] = self.encoder({nt: sub_tf})[nt]
        return out[inv].view(B, K, self.channels)

    def _tokenize(self, batch: TokenBatch, device) -> Tensor:
        feats = self._encode_token_features(batch, device)  # [B, K, D]
        x_feat = self.feat_proj(feats)
        x_type = self.type_emb(batch.type_ids.to(device))
        x_hop = self.hop_emb(batch.hops.to(device))
        x_time = self.time_enc(batch.time_delta.to(device))
        x_pe = self.rwse_proj(batch.rwse.to(device))
        x = torch.cat([x_feat, x_type, x_hop, x_time, x_pe], dim=-1)
        return self.in_mix(x)  # [B, K, D]

    def _global_attention(self, seed: Tensor) -> Tensor:
        # seed: [B, D] attends over learnable centroids.
        q = self.q_proj(seed)  # [B, D]
        k = self.k_proj(self.centroids)  # [C, D]
        v = self.v_proj(self.centroids)  # [C, D]
        scale = 1.0 / math.sqrt(self.channels)
        attn = torch.softmax(q @ k.t() * scale, dim=-1)  # [B, C]
        ctx = attn @ v  # [B, D]
        return self.global_norm(ctx)

    def forward(self, batch: TokenBatch, device) -> Tensor:
        x = self._tokenize(batch, device)  # [B, K, D]
        pad_mask = batch.pad_mask.to(device)
        z = self.local(x, src_key_padding_mask=pad_mask)  # [B, K, D]
        seed = z[:, 0, :]  # seed token readout
        if self.use_global:
            seed = seed + self._global_attention(seed)
        out = self.head(seed)  # [B, out_channels]
        return out.view(-1) if self.out_channels == 1 else out


# --------------------------------------------------------------------------------------
# Experiment driver
# --------------------------------------------------------------------------------------
def _fill_numerical_nans(data, col_stats_dict) -> None:
    """Mean-fill NaNs in materialized numerical TensorFrames.

    TorchFrame's numerical encoder fills NaNs in its *output*, but the gradient w.r.t.
    the encoder weight is ``input`` -> NaN for NaN inputs, producing NaN grads when we
    encode the full node set. Replacing raw NaNs with the column mean is a no-op after
    the encoder's standardization and keeps backprop finite.
    """
    import torch_frame
    from torch_frame.data.stats import StatType

    num_key = torch_frame.stype.numerical
    for nt in data.node_types:
        tf = data[nt].tf
        if num_key not in tf.feat_dict:
            continue
        feat = tf.feat_dict[num_key]
        cols = tf.col_names_dict[num_key]
        for j, c in enumerate(cols):
            mean = float(col_stats_dict.get(nt, {}).get(c, {}).get(StatType.MEAN, 0.0))
            col = feat[..., j]
            col[torch.isnan(col)] = mean
        tf.feat_dict[num_key] = feat


def run_relgt_lite_experiment(
    *,
    dataset_name: str = "rel-f1",
    task_name: str = "driver-dnf",
    train_seeds: int = 512,
    val_seeds: int = 256,
    K: int = 32,
    channels: int = 64,
    heads: int = 4,
    local_layers: int = 2,
    num_centroids: int = 256,
    max_hop: int = 2,
    rwse_steps: int = 4,
    epochs: int = 6,
    batch_size: int = 64,
    lr: float = 5e-4,
    use_global: bool = True,
    seed: int = 42,
    cache_subdir: str = "relbench_cache/relgt_lite",
    verbose: bool = True,
) -> Dict[str, float]:
    from pathlib import Path

    from relbench.datasets import get_dataset
    from relbench.modeling.graph import get_node_train_table_input, make_pkey_fkey_graph
    from relbench.modeling.utils import get_stype_proposal
    from relbench.tasks import get_task

    from src.relbench_pipeline.stypes import sanitize_col_to_stype_dict
    from src.relbench_pipeline.task_utils import cast_labels, compute_metric, make_loss, task_spec

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    dataset = get_dataset(dataset_name, download=True)
    task = get_task(dataset_name, task_name, download=True)
    db = dataset.get_db()
    spec = task_spec(task)  # raises on unsupported (e.g. link-prediction) task types

    col_to_stype_dict = sanitize_col_to_stype_dict(get_stype_proposal(db))
    materialized = Path(cache_subdir) / f"{dataset_name}_materialized"
    materialized.mkdir(parents=True, exist_ok=True)
    data, col_stats_dict = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=col_to_stype_dict,
        text_embedder_cfg=None,
        cache_dir=str(materialized),
    )
    _fill_numerical_nans(data, col_stats_dict)

    tokenizer = RelGraphTokenizer(data, max_hop=max_hop, rwse_steps=rwse_steps)

    def make_token_batches(split: str, n_seeds: int) -> List[TokenBatch]:
        table = task.get_table(split)
        ti = get_node_train_table_input(table=table, task=task)
        seed_type = ti.nodes[0]
        idxs = ti.nodes[1][:n_seeds]
        times = ti.time[:n_seeds] if ti.time is not None else None
        target = ti.target[:n_seeds] if ti.target is not None else None
        batches: List[TokenBatch] = []
        for start in range(0, len(idxs), batch_size):
            end = min(start + batch_size, len(idxs))
            batches.append(
                tokenizer.build(
                    seed_type,
                    idxs[start:end],
                    times[start:end] if times is not None else None,
                    target[start:end] if target is not None else None,
                    K=K,
                )
            )
        return batches

    if verbose:
        print(f"[relgt-lite] tokenizing {train_seeds} train / {val_seeds} val seeds (K={K})...")
    train_batches = make_token_batches("train", train_seeds)
    val_batches = make_token_batches("val", val_seeds)

    model = RelGTLite(
        data=data,
        col_stats_dict=col_stats_dict,
        num_types=tokenizer.num_types,
        max_hop=max_hop,
        rwse_steps=rwse_steps,
        channels=channels,
        heads=heads,
        local_layers=local_layers,
        num_centroids=num_centroids,
        use_global=use_global,
        out_channels=spec["out_channels"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(
            f"[relgt-lite] params: {n_params:,} | use_global={use_global} | "
            f"channels={channels} | K={K} | task_type={task.task_type} | metric={spec['metric']}"
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    train_labels = np.concatenate([b.labels.numpy() for b in train_batches])
    loss_fn = make_loss(task, train_labels)
    metric_name = spec["metric"]
    higher_is_better = spec["higher_is_better"]

    @torch.no_grad()
    def evaluate(batches: List[TokenBatch]) -> float:
        model.eval()
        preds, labels = [], []
        for tb in batches:
            preds.append(model(tb, device).cpu().numpy())
            labels.append(tb.labels.cpu().numpy())
        return compute_metric(task, np.concatenate(labels), np.concatenate(preds))

    best_metric = float("-inf") if higher_is_better else float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        order = np.random.permutation(len(train_batches))
        total, count = 0.0, 0
        for bi in order:
            tb = train_batches[bi]
            optimizer.zero_grad()
            out = model(tb, device)
            loss = loss_fn(out, cast_labels(tb.labels, task).to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            bs = tb.global_ids.size(0)
            total += loss.item() * bs
            count += bs
        val_metric = evaluate(val_batches)
        improved = (
            val_metric > best_metric if higher_is_better else val_metric < best_metric
        )
        if not np.isnan(val_metric) and improved:
            best_metric = val_metric
        if verbose:
            print(
                f"epoch {epoch:02d} | train loss {total / max(count, 1):.4f} "
                f"| val {metric_name} {val_metric:.4f}"
            )

    return {
        "metric": metric_name,
        "best_val_metric": best_metric,
        "params": float(n_params),
        "train_seeds": float(sum(b.global_ids.size(0) for b in train_batches)),
        "val_seeds": float(sum(b.global_ids.size(0) for b in val_batches)),
    }
