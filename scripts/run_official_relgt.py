"""Run the *official* snap-stanford/relgt model on a RelBench task, single-process, CPU-friendly.

The upstream entry point (``external/relgt/main_node_ddp.py``) is hard-wired for an
8x A100 NVIDIA DDP setup (``nccl``, ``pynvml``, ``wandb``, CUDA, GloVe text embeddings).
This driver imports the *real* RelGT model + tokenizer from that repo unchanged, but:

  * runs in a single process on CPU (no DDP / nccl / pynvml),
  * stubs out ``sentence_transformers`` (we sanitize text columns to categorical
    instead of downloading GloVe),
  * patches the multiprocessing ``Pool`` used by the tokenizer to run *serially*
    in-process (macOS ``spawn`` + per-task pickling of the full graph is brittle),
  * subsamples seeds so a smoke run finishes in minutes rather than days.

The goal is a real, citable number from the official architecture on this laptop,
to use as a correctness reference for the from-scratch RelGT-lite implementation.

Example:
    python scripts/run_official_relgt.py --dataset rel-f1 --task driver-dnf \
        --train-seeds 256 --val-seeds 128 --k 32 --channels 64 --epochs 3
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
RELGT_DIR = ROOT / "external" / "relgt"

# external/relgt uses flat top-level imports (model, utils, encoders, ...).
# Put it first so ``import model`` resolves to RelGT's model.py, not ours.
sys.path.insert(0, str(RELGT_DIR))
sys.path.insert(0, str(ROOT))


def _stub_sentence_transformers() -> None:
    """utils.py does ``from sentence_transformers import SentenceTransformer`` at import
    time. We never use GloVe (text -> categorical), so inject a lightweight stub."""
    if "sentence_transformers" in sys.modules:
        return
    stub = types.ModuleType("sentence_transformers")

    class _SentenceTransformer:  # pragma: no cover - never instantiated
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Stubbed SentenceTransformer should not be used.")

    stub.SentenceTransformer = _SentenceTransformer
    sys.modules["sentence_transformers"] = stub


class _SerialPool:
    """Drop-in replacement for multiprocessing.Pool that runs in-process.

    The tokenizer's ``local_nodes_hetero`` builds tasks that embed the full
    ``HeteroData`` object; pickling that per task under macOS ``spawn`` is slow
    and fragile. Running serially with the same ``initializer`` (which sets module
    globals in *this* process) is correct and far more robust for a CPU smoke run.
    """

    def __init__(self, processes=None, initializer=None, initargs=()):
        if initializer is not None:
            initializer(*initargs)

    def map(self, fn, tasks):
        return [fn(t) for t in tasks]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def run_official_relgt(
    *,
    dataset: str = "rel-f1",
    task: str = "driver-dnf",
    train_seeds: int = 512,
    val_seeds: int = 256,
    k: int = 32,
    channels: int = 64,
    heads: int = 4,
    local_layers: int = 1,
    num_centroids: int = 256,
    conv_type: str = "full",
    epochs: int = 6,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 42,
    cache_dir: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """Train the official RelGT on a RelBench binary task (CPU, subsampled). Returns metrics."""
    cache_dir = cache_dir or str(ROOT / "relbench_cache" / "official_relgt")
    _stub_sentence_transformers()

    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader

    from relbench.base import TaskType
    from relbench.datasets import get_dataset
    from relbench.modeling.graph import make_pkey_fkey_graph
    from relbench.modeling.utils import get_stype_proposal
    from relbench.tasks import get_task

    from src.relbench_pipeline.stypes import sanitize_col_to_stype_dict

    # Import RelGT pieces (resolve to external/relgt) and patch its Pool to serial.
    import utils as relgt_utils
    from utils import RelGTTokens
    from model import RelGT

    relgt_utils.Pool = _SerialPool

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    if verbose:
        print(f"== Official RelGT (single-process CPU) on {dataset}/{task} ==")
    ds_obj = get_dataset(dataset, download=True)
    task_obj = get_task(dataset, task, download=True)
    db = ds_obj.get_db()

    if task_obj.task_type != TaskType.BINARY_CLASSIFICATION:
        raise SystemExit(f"This driver reports ROC-AUC; task {task} is {task_obj.task_type}.")

    col_to_stype_dict = sanitize_col_to_stype_dict(get_stype_proposal(db))
    materialized = Path(cache_dir) / f"{dataset}_materialized"
    materialized.mkdir(parents=True, exist_ok=True)

    data, col_stats_dict = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=col_to_stype_dict,
        text_embedder_cfg=None,  # text columns sanitized to categorical
        cache_dir=str(materialized),
    )

    def build_tokens(split: str, n_seeds: int) -> "RelGTTokens":
        precomp_dir = str(Path(cache_dir) / "precomputed" / dataset / task / split)
        tokens = RelGTTokens(
            data=data,
            task=task_obj,
            K=k,
            split=split,
            undirected=True,
            precompute=False,  # we truncate seeds first, then precompute manually
            precomputed_dir=precomp_dir,
            num_workers=1,  # value is irrelevant once Pool is serial-patched
        )
        n = min(n_seeds, len(tokens.node_idxs))
        tokens.node_idxs = tokens.node_idxs[:n]
        if tokens.target is not None:
            tokens.target = tokens.target[:n]
        if tokens.time is not None:
            tokens.time = tokens.time[:n]
        tokens.precomputed_path = str(Path(precomp_dir) / str(k) / f"{split}_n{n}.h5")
        Path(tokens.precomputed_path).parent.mkdir(parents=True, exist_ok=True)
        if not Path(tokens.precomputed_path).exists():
            if verbose:
                print(f"[{split}] precomputing {n} seeds (K={k})...")
            tokens._precompute_sampling()
        return tokens

    train_ds = build_tokens("train", train_seeds)
    val_ds = build_tokens("val", val_seeds)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=train_ds.collate, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=val_ds.collate, num_workers=0
    )

    model = RelGT(
        num_nodes=data.num_nodes,
        max_neighbor_hop=train_ds.max_neighbor_hop,
        node_type_map=train_ds.node_type_to_index,
        col_names_dict={nt: data[nt].tf.col_names_dict for nt in data.node_types},
        col_stats_dict=col_stats_dict,
        local_num_layers=local_layers,
        channels=channels,
        out_channels=1,
        global_dim=max(channels // 2, 8),
        heads=heads,
        ff_dropout=0.1,
        attn_dropout=0.1,
        conv_type=conv_type,
        num_centroids=num_centroids,
        sample_node_len=k,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"RelGT params: {n_params:,} | conv_type={conv_type} | K={k}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def forward_batch(batch):
        grouped_tf_dict = {
            "grouped_tfs": batch["grouped_tfs"],
            "grouped_indices": batch["grouped_indices"],
            "flat_batch_idx": batch["flat_batch_idx"],
            "flat_nbr_idx": batch["flat_nbr_idx"],
        }
        pred = model(
            batch["neighbor_types"].to(device),
            batch["node_indices"].to(device),
            batch["neighbor_hops"].to(device),
            batch["neighbor_times"].to(device),
            grouped_tf_dict,
            edge_index=batch["edge_index"].to(device),
            batch=batch["batch"].to(device),
        )
        return pred.view(-1) if pred.size(1) == 1 else pred

    @torch.no_grad()
    def evaluate(loader) -> float:
        model.eval()
        preds, labels = [], []
        for batch in loader:
            pred = torch.sigmoid(forward_batch(batch))
            preds.append(pred.cpu().numpy())
            labels.append(batch["labels"].cpu().numpy())
        y = np.concatenate(labels)
        p = np.concatenate(preds)
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y, p))

    best_auc = float("-inf")
    for epoch in range(1, epochs + 1):
        model.train()
        total, count = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            pred = forward_batch(batch)
            loss = loss_fn(pred.float(), batch["labels"].float().to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * pred.size(0)
            count += pred.size(0)
        val_auc = evaluate(val_loader)
        best_auc = max(best_auc, val_auc)
        if verbose:
            print(
                f"epoch {epoch:02d} | train loss {total / max(count, 1):.4f} "
                f"| val ROC-AUC {val_auc:.4f}"
            )

    if verbose:
        print(f"\nBEST val ROC-AUC (subset of {len(val_ds.node_idxs)} seeds): {best_auc:.4f}")
        print(
            "NOTE: subsampled seeds + tiny model on CPU -> a sanity/correctness number, "
            "not the paper's full-scale result."
        )
    return {"metric": "roc_auc", "best_val_metric": best_auc, "params": float(n_params)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="rel-f1")
    parser.add_argument("--task", type=str, default="driver-dnf")
    parser.add_argument("--train-seeds", type=int, default=512)
    parser.add_argument("--val-seeds", type=int, default=256)
    parser.add_argument("--k", type=int, default=32, help="tokens per subgraph (sample_node_len)")
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--local-layers", type=int, default=1)
    parser.add_argument("--num-centroids", type=int, default=256)
    parser.add_argument("--conv-type", type=str, default="full", choices=["local", "global", "full"])
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=str, default=None)
    args = parser.parse_args()

    run_official_relgt(
        dataset=args.dataset,
        task=args.task,
        train_seeds=args.train_seeds,
        val_seeds=args.val_seeds,
        k=args.k,
        channels=args.channels,
        heads=args.heads,
        local_layers=args.local_layers,
        num_centroids=args.num_centroids,
        conv_type=args.conv_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
