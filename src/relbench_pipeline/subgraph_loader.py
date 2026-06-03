"""Temporal subgraph construction for CPU mini-batches (pyg-lib-free)."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType


def _sample_neighbors(
    edge_index: Tensor,
    src_nodes: Tensor,
    *,
    num_neighbors: int,
    dst_time: Optional[Tensor] = None,
    src_time: Optional[Tensor] = None,
) -> Tensor:
    """Sample up to num_neighbors dst nodes per src from edge_index [2, E]."""
    src, dst = edge_index
    if src_nodes.numel() == 0:
        return torch.empty(0, dtype=torch.long)

    mask = torch.isin(src, src_nodes)
    if not mask.any():
        return torch.empty(0, dtype=torch.long)

    src_m = src[mask]
    dst_m = dst[mask]

    if dst_time is not None and src_time is not None:
        tmask = dst_time[dst_m] <= src_time[src_m]
        src_m = src_m[tmask]
        dst_m = dst_m[tmask]
        if dst_m.numel() == 0:
            return torch.empty(0, dtype=torch.long)

    picked: list[Tensor] = []
    for s in src_nodes.unique():
        candidates = dst_m[src_m == s]
        if candidates.numel() == 0:
            continue
        k = min(num_neighbors, candidates.numel())
        perm = torch.randperm(candidates.numel())[:k]
        picked.append(candidates[perm])
    if not picked:
        return torch.empty(0, dtype=torch.long)
    return torch.cat(picked).unique()


def build_temporal_subgraph(
    data: HeteroData,
    entity_table: NodeType,
    node_ids: Tensor,
    seed_time: Tensor,
    num_neighbors: List[int],
) -> HeteroData:
    """2-hop temporal heterogeneous subgraph for one mini-batch of seeds."""
    sampled: Dict[NodeType, Tensor] = {entity_table: node_ids.unique()}
    per_seed_time = {int(n.item()): int(t.item()) for n, t in zip(node_ids, seed_time)}

    for num_n in num_neighbors:
        for edge_type in data.edge_types:
            src_t, _, dst_t = edge_type
            if src_t not in sampled or sampled[src_t].numel() == 0:
                continue
            edge_index = data[edge_type].edge_index
            dst_time = data[dst_t].time if "time" in data[dst_t] else None
            src_time_vec = None
            if dst_time is not None:
                src_time_vec = torch.zeros(data[src_t].num_nodes, dtype=dst_time.dtype)
                for nid in sampled[src_t]:
                    src_time_vec[nid] = int(per_seed_time.get(int(nid.item()), seed_time.max().item()))

            new_dst = _sample_neighbors(
                edge_index,
                sampled[src_t],
                num_neighbors=num_n,
                dst_time=dst_time,
                src_time=src_time_vec,
            )
            if new_dst.numel():
                sampled[dst_t] = torch.cat([sampled.get(dst_t, torch.empty(0, dtype=torch.long)), new_dst]).unique()

    # Local reindex
    local_map: Dict[NodeType, Tensor] = {}
    for ntype, nodes in sampled.items():
        local_map[ntype] = nodes

    out = HeteroData()
    batch_dict: Dict[NodeType, Tensor] = {}

    for ntype, global_ids in local_map.items():
        global_ids = global_ids.sort().values
        out[ntype].n_id = global_ids
        out[ntype].num_nodes = global_ids.numel()
        out[ntype].tf = data[ntype].tf[global_ids]
        if "time" in data[ntype]:
            out[ntype].time = data[ntype].time[global_ids]

        if ntype == entity_table:
            # Map seeds to batch positions 0..B-1
            b = torch.full((global_ids.numel(),), -1, dtype=torch.long)
            for i, sid in enumerate(node_ids):
                pos = (global_ids == sid).nonzero(as_tuple=True)[0]
                if pos.numel():
                    b[pos[0]] = i
            batch_dict[ntype] = b.clamp(min=0)
        else:
            batch_dict[ntype] = torch.zeros(global_ids.numel(), dtype=torch.long)

    out.batch_dict = batch_dict
    out.time_dict = {nt: out[nt].time for nt in out.node_types if "time" in out[nt]}

    for edge_type in data.edge_types:
        src_t, rel, dst_t = edge_type
        if src_t not in local_map or dst_t not in local_map:
            continue
        ei = data[edge_type].edge_index
        src_global, dst_global = ei[0], ei[1]
        src_mask = torch.isin(src_global, local_map[src_t])
        dst_mask = torch.isin(dst_global, local_map[dst_t])
        mask = src_mask & dst_mask
        if not mask.any():
            continue
        sg, dg = src_global[mask], dst_global[mask]
        src_local = torch.searchsorted(local_map[src_t], sg)
        dst_local = torch.searchsorted(local_map[dst_t], dg)
        out[edge_type].edge_index = torch.stack([src_local, dst_local], dim=0)

    out[entity_table].seed_time = seed_time
    out[entity_table].input_id = torch.arange(node_ids.numel())

    # Ensure every node type has at least a self-loop for message passing.
    for ntype in out.node_types:
        n = out[ntype].num_nodes
        if n == 0:
            continue
        self_type = (ntype, "self_loop", ntype)
        if self_type not in out.edge_types:
            idx = torch.arange(n)
            out[self_type].edge_index = torch.stack([idx, idx], dim=0)

    out.tf_dict = {nt: out[nt].tf for nt in out.node_types}
    return out
