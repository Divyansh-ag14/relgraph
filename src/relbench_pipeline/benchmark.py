"""Unified benchmark: every model on every task, one results matrix.

Runs three models under one shared, comparable protocol (first-N seeds, CPU):

  * **flat**       - entity-table features only, sklearn (no graph message passing)
  * **graphsage**  - RelBench HeteroEncoder + HeteroGraphSAGE on temporal subgraphs
  * **relgt-lite** - our from-scratch Relational Graph Transformer

Each task reports its *native* metric (binary -> ROC-AUC, regression -> MAE,
multiclass -> accuracy) via ``task_utils``, so the matrix is consistent across task
types. This is the single place that answers "all models on all tasks".
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from src.relbench_pipeline.task_utils import cast_labels, compute_metric, make_loss, task_spec


def prepare_graph(dataset_name: str, cache_root: str = "relbench_cache/benchmark"):
    """Load a RelBench dataset, materialize the FK graph, mean-fill numerical NaNs."""
    from relbench.datasets import get_dataset
    from relbench.modeling.graph import make_pkey_fkey_graph
    from relbench.modeling.utils import get_stype_proposal

    from src.relbench_pipeline.relgt_lite import _fill_numerical_nans
    from src.relbench_pipeline.stypes import sanitize_col_to_stype_dict

    dataset = get_dataset(dataset_name, download=True)
    db = dataset.get_db()
    col_to_stype_dict = sanitize_col_to_stype_dict(get_stype_proposal(db))
    materialized = Path(cache_root) / f"{dataset_name}_materialized"
    materialized.mkdir(parents=True, exist_ok=True)
    data, col_stats_dict = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=col_to_stype_dict,
        text_embedder_cfg=None,
        cache_dir=str(materialized),
    )
    _fill_numerical_nans(data, col_stats_dict)
    return db, data, col_stats_dict


def _seed_inputs(task, split: str, n_seeds: int):
    from relbench.modeling.graph import get_node_train_table_input

    ti = get_node_train_table_input(table=task.get_table(split), task=task)
    seed_type = ti.nodes[0]
    idxs = ti.nodes[1][:n_seeds]
    times = ti.time[:n_seeds] if ti.time is not None else torch.zeros(len(idxs), dtype=torch.long)
    target = ti.target[:n_seeds] if ti.target is not None else None
    return seed_type, idxs, times, target


# --------------------------------------------------------------------------------------
# Model 1: flat baseline (entity table only, no graph)
# --------------------------------------------------------------------------------------
def run_flat(db, task, *, train_seeds: int, val_seeds: int) -> Dict[str, float]:
    import pandas as pd
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from relbench.base import TaskType

    spec = task_spec(task)
    entity_df = db.table_dict[task.entity_table].df.copy()
    drop = {c for c in entity_df.columns if c.endswith("Id")} | {task.entity_col}
    feat_df = entity_df.drop(columns=[c for c in drop if c in entity_df.columns], errors="ignore")
    for col in feat_df.columns:
        if feat_df[col].dtype == object or str(feat_df[col].dtype).startswith("datetime"):
            feat_df[col] = pd.Categorical(feat_df[col].astype(str)).codes
    X_all = feat_df.select_dtypes(include=[np.number]).fillna(0).to_numpy(dtype=np.float32)

    def split_xy(split: str, n: int):
        _t, idxs, _tm, target = _seed_inputs(task, split, n)
        return X_all[idxs.numpy()], target.numpy()

    X_tr, y_tr = split_xy("train", train_seeds)
    X_va, y_va = split_xy("val", val_seeds)

    if task.task_type == TaskType.REGRESSION:
        model = Pipeline([("s", StandardScaler()), ("m", Ridge())])
        model.fit(X_tr, y_tr)
        val = float(mean_absolute_error(y_va, model.predict(X_va)))
    elif task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
        model = Pipeline([("s", StandardScaler()), ("m", LogisticRegression(max_iter=1000))])
        model.fit(X_tr, y_tr)
        val = float(accuracy_score(y_va, model.predict(X_va)))
    else:  # binary
        model = Pipeline(
            [("s", StandardScaler()), ("m", LogisticRegression(max_iter=1000, class_weight="balanced"))]
        )
        model.fit(X_tr, y_tr)
        prob = model.predict_proba(X_va)[:, 1]
        val = float(roc_auc_score(y_va, prob)) if len(np.unique(y_va)) > 1 else float("nan")

    return {"metric": spec["metric"], "best_val_metric": val, "params": float(X_all.shape[1])}


# --------------------------------------------------------------------------------------
# Model 2: HeteroGraphSAGE on temporal subgraphs
# --------------------------------------------------------------------------------------
def run_graphsage(
    data,
    col_stats_dict,
    task,
    *,
    train_seeds: int,
    val_seeds: int,
    channels: int = 64,
    num_layers: int = 2,
    num_neighbors: Optional[List[int]] = None,
    epochs: int = 8,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, float]:
    from src.relbench_pipeline.model import RelBenchModel
    from src.relbench_pipeline.subgraph_loader import build_temporal_subgraph

    num_neighbors = num_neighbors or [32] * num_layers
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")
    spec = task_spec(task)

    def make_batches(split: str, n: int):
        seed_type, idxs, times, target = _seed_inputs(task, split, n)
        batches = []
        for s in range(0, len(idxs), batch_size):
            e = min(s + batch_size, len(idxs))
            b = build_temporal_subgraph(data, seed_type, idxs[s:e], times[s:e], num_neighbors)
            if target is not None:
                b[seed_type].y = target[s:e]
            batches.append(b)
        return seed_type, batches

    entity_table, train_batches = make_batches("train", train_seeds)
    _, val_batches = make_batches("val", val_seeds)

    model = RelBenchModel(
        data, col_stats_dict, num_layers=num_layers, channels=channels,
        out_channels=spec["out_channels"], aggr="mean",
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_labels = np.concatenate([b[entity_table].y.numpy() for b in train_batches])
    loss_fn = make_loss(task, train_labels)
    higher = spec["higher_is_better"]

    @torch.no_grad()
    def evaluate(batches) -> float:
        model.eval()
        preds, labels = [], []
        for b in batches:
            out = model(b.to(device), entity_table)
            out = out.view(-1) if spec["out_channels"] == 1 else out
            preds.append(out.cpu().numpy())
            labels.append(b[entity_table].y.numpy())
        return compute_metric(task, np.concatenate(labels), np.concatenate(preds))

    best = float("-inf") if higher else float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        total = count = 0.0
        for bi in np.random.permutation(len(train_batches)):
            b = train_batches[bi]
            optimizer.zero_grad()
            out = model(b.to(device), entity_table)
            out = out.view(-1) if spec["out_channels"] == 1 else out
            loss = loss_fn(out, cast_labels(b[entity_table].y, task).to(device))
            loss.backward()
            optimizer.step()
            total += loss.item() * out.size(0)
            count += out.size(0)
        val = evaluate(val_batches)
        if not np.isnan(val) and ((val > best) if higher else (val < best)):
            best = val
        if verbose:
            print(f"[graphsage] epoch {epoch:02d} | loss {total / max(count, 1):.4f} "
                  f"| val {spec['metric']} {val:.4f}")

    return {"metric": spec["metric"], "best_val_metric": best, "params": float(n_params)}


# --------------------------------------------------------------------------------------
# Matrix runner
# --------------------------------------------------------------------------------------
def benchmark(
    *,
    dataset_name: str,
    task_names: List[str],
    models: List[str],
    train_seeds: int = 512,
    val_seeds: int = 256,
    K: int = 32,
    channels: int = 64,
    epochs: int = 8,
    batch_size: int = 64,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    from relbench.tasks import get_task

    from src.relbench_pipeline.relgt_lite import run_relgt_lite_experiment

    db, data, col_stats_dict = prepare_graph(dataset_name)

    # model -> task -> {"metric", "value"}
    matrix: Dict[str, Dict[str, dict]] = {m: {} for m in models}

    for task_name in task_names:
        task = get_task(dataset_name, task_name, download=True)
        try:
            spec = task_spec(task)
        except ValueError as exc:
            if verbose:
                print(f"[skip] {task_name}: {exc}")
            for m in models:
                matrix[m][task_name] = {"metric": "n/a", "value": float("nan")}
            continue

        if verbose:
            print(f"\n===== task {task_name} ({task.task_type}, metric={spec['metric']}) =====")

        if "flat" in models:
            r = run_flat(db, task, train_seeds=train_seeds, val_seeds=val_seeds)
            matrix["flat"][task_name] = {"metric": r["metric"], "value": r["best_val_metric"]}
        if "graphsage" in models:
            r = run_graphsage(
                data, col_stats_dict, task, train_seeds=train_seeds, val_seeds=val_seeds,
                channels=channels, epochs=epochs, batch_size=batch_size, seed=seed, verbose=verbose,
            )
            matrix["graphsage"][task_name] = {"metric": r["metric"], "value": r["best_val_metric"]}
        if "relgt-lite" in models:
            r = run_relgt_lite_experiment(
                dataset_name=dataset_name, task_name=task_name, train_seeds=train_seeds,
                val_seeds=val_seeds, K=K, channels=channels, epochs=epochs,
                batch_size=batch_size, seed=seed, verbose=verbose,
            )
            matrix["relgt-lite"][task_name] = {"metric": r["metric"], "value": r["best_val_metric"]}

    return {
        "dataset": dataset_name,
        "protocol": {
            "train_seeds": train_seeds, "val_seeds": val_seeds, "K": K,
            "channels": channels, "epochs": epochs, "device": "cpu",
        },
        "tasks": task_names,
        "models": models,
        "matrix": matrix,
    }
