"""Task-type-aware helpers shared across models (binary / regression / multiclass).

RelBench entity tasks come in several flavors; this centralizes the per-type choices
(output dim, loss, label dtype, primary metric) so every model in the repo reports the
same metric for a given task and the unified benchmark stays consistent.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from relbench.base import TaskType


def task_spec(task) -> dict:
    """Return output dim, primary metric, direction, and label dtype for a task."""
    tt = task.task_type
    if tt == TaskType.BINARY_CLASSIFICATION:
        return {
            "metric": "roc_auc",
            "higher_is_better": True,
            "out_channels": 1,
            "label_dtype": "float",
        }
    if tt == TaskType.REGRESSION:
        return {
            "metric": "mae",
            "higher_is_better": False,
            "out_channels": 1,
            "label_dtype": "float",
        }
    if tt == TaskType.MULTICLASS_CLASSIFICATION:
        return {
            "metric": "accuracy",
            "higher_is_better": True,
            "out_channels": int(task.num_classes),
            "label_dtype": "long",
        }
    raise ValueError(f"Unsupported task type for this pipeline: {tt}")


def make_loss(task, train_labels: Optional[np.ndarray] = None) -> torch.nn.Module:
    """Loss matched to the task type. Binary gets class-imbalance pos_weight."""
    tt = task.task_type
    if tt == TaskType.BINARY_CLASSIFICATION:
        pos_weight = None
        if train_labels is not None and len(train_labels) > 0:
            pos = float(np.sum(train_labels))
            neg = float(len(train_labels) - pos)
            pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)
        return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if tt == TaskType.REGRESSION:
        return torch.nn.L1Loss()
    if tt == TaskType.MULTICLASS_CLASSIFICATION:
        return torch.nn.CrossEntropyLoss()
    raise ValueError(f"Unsupported task type for this pipeline: {tt}")


def cast_labels(labels: torch.Tensor, task) -> torch.Tensor:
    """Cast a label tensor to the dtype the loss expects."""
    return labels.long() if task_spec(task)["label_dtype"] == "long" else labels.float()


def compute_metric(task, y_true, raw_pred) -> float:
    """Primary metric from raw model outputs (logits / regression values)."""
    from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score

    tt = task.task_type
    y_true = np.asarray(y_true)
    raw_pred = np.asarray(raw_pred)

    if tt == TaskType.BINARY_CLASSIFICATION:
        from scipy.special import expit

        prob = expit(raw_pred.reshape(-1))
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, prob))
    if tt == TaskType.REGRESSION:
        return float(mean_absolute_error(y_true, raw_pred.reshape(-1)))
    if tt == TaskType.MULTICLASS_CLASSIFICATION:
        return float(accuracy_score(y_true, raw_pred.argmax(-1)))
    raise ValueError(f"Unsupported task type for this pipeline: {tt}")
