# RelGraph

Relational deep learning on [RelBench](https://relbench.stanford.edu): multi-table databases become temporal heterogeneous graphs, then we train flat, GraphSAGE, and a from-scratch Relational Graph Transformer (RelGT-lite). Includes a small PQL compiler and a unified benchmark runner.

CPU runs on a laptop are subset/short-epoch sanity checks, not paper-scale numbers.

## Install

```bash
git clone <this-repo> rdl-project && cd rdl-project
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Datasets download on first use into `relbench_cache/`.

## Commands

```bash
# List tasks and verify the DB loads
python scripts/run_relbench.py --dataset rel-f1

# Train/eval: pick any models and tasks
python scripts/benchmark.py --dataset rel-f1 \
  --tasks driver-dnf driver-top3 driver-position \
  --models flat graphsage relgt-lite

python scripts/benchmark.py --tasks driver-dnf --models relgt-lite

# RelGT comparison (lite vs ablation vs official)
python scripts/compare_relgt.py --skip-official
git clone https://github.com/snap-stanford/relgt.git external/relgt
python scripts/compare_relgt.py

# PQL label compiler demo
python scripts/pql_demo.py --dataset rel-f1

# Link prediction
python scripts/train_link_prediction.py --dataset rel-f1

# Print saved metrics
python scripts/show_results.py
```

Entity tasks report **validation** metrics only (RelBench hides test labels for the leaderboard). Link prediction has a local test split.

| Script | Purpose |
|--------|---------|
| `benchmark.py` | `flat`, `graphsage`, `relgt-lite` on entity tasks |
| `compare_relgt.py` | RelGT variants on one task |
| `pql_demo.py` | Parse/compile/validate PQL queries |
| `train_link_prediction.py` | FK edge reconstruction |
| `show_results.py` | Tables from `outputs/*.json` |
| `run_relbench.py` | List tasks, no training |

## Task types

Handled in `src/relbench_pipeline/task_utils.py`:

| Type | `rel-f1` example | Metric |
|------|------------------|--------|
| Binary | `driver-dnf`, `driver-top3` | ROC-AUC |
| Regression | `driver-position` | MAE |
| Multiclass | other datasets | accuracy |
| Link prediction | `results → constructors` | ROC-AUC |

## Results (CPU subset)

`rel-f1`, 1024 train / 512 val seeds, K=32, 64 channels, 8 epochs. Full matrix: `outputs/benchmark_matrix.json`.

| Model | driver-dnf | driver-top3 | driver-position |
|-------|------------|-------------|-----------------|
| flat | 0.588 | 0.571 | 3.95 MAE |
| graphsage | 0.505 | 0.523 | 4.78 MAE |
| relgt-lite | **0.679** | **0.669** | **3.76 MAE** |

RelGT comparison on `driver-dnf` (`outputs/relgt_comparison.json`): official RelGT 0.692, RelGT-lite 0.679, local-only ablation 0.664.

Link prediction test ROC-AUC ≈ 0.96 on `results → constructors` (`outputs/link_prediction.json`).

## RelGT-lite

From-scratch [RelGT](https://arxiv.org/abs/2505.10960) in `src/relbench_pipeline/relgt_lite.py`: five token elements (features, node type, hop, time gap, RWSE structure), local Transformer over the neighborhood, global attention to learnable centroids. Compared to [snap-stanford/relgt](https://github.com/snap-stanford/relgt) on the same seed protocol.

CPU simplifications: softmax centroids instead of the paper's EMA K-Means codebook, RWSE instead of a full GNN positional encoder, one layer, subsampled seeds.

## PQL

`src/relbench_pipeline/pql.py` compiles queries like:

```text
PREDICT COUNT(results, 365) = 0 FOR EACH drivers
PREDICT SUM(results.points, 90) FOR EACH drivers
```

into `[entity, timestamp, target]` label tables via temporal FK aggregation. `scripts/pql_demo.py` validates against an independent pandas recomputation. Training still uses RelBench's built-in tasks unless you wire custom labels in yourself.

## Layout

```text
src/relbench_pipeline/   library (benchmark, model, relgt_lite, pql, link_prediction, ...)
scripts/                 CLI wrappers
external/relgt/          official RelGT (clone on demand)
outputs/                 benchmark_matrix.json, relgt_comparison.json, link_prediction.json, report.txt
```

## References

- [RelGT](https://arxiv.org/abs/2505.10960) · [RDL](https://arxiv.org/html/2312.04615v1) · [RelBench](https://github.com/snap-stanford/relbench) · [PyG](https://pytorch-geometric.readthedocs.io/) · [PyTorch Frame](https://github.com/pyg-team/pytorch-frame)

MIT
