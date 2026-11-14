"""Classification metrics and the documented latency protocol.

Macro-F1 (``average="macro"`` with all three canonical labels forced) is the
primary model-selection metric because it weights all classes equally
regardless of support. ``zero_division=0`` is passed explicitly so that a
class absent from both truth and predictions scores 0.0 instead of raising
or producing NaN — the behavior is documented here, not implicit.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from vfi.data.loading import CANONICAL_LABELS


def compute_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, object]:
    """Compute the full classification report over the canonical labels.

    Returns a JSON-serializable dict with accuracy, macro precision/recall/F1,
    per-class precision/recall/F1/support, the 3x3 confusion matrix (rows =
    true label, columns = predicted label, canonical order), and sample count.

    Raises:
        ValueError: when the inputs have different lengths.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true and y_pred lengths differ: {len(y_true)} vs {len(y_pred)}")
    labels = list(CANONICAL_LABELS)
    accuracy = float(accuracy_score(y_true, y_pred))
    per_class = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    macro = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "labels": labels,
        "accuracy": accuracy,
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "per_class": {
            label: {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": int(support),
            }
            for label, precision, recall, f1, support in zip(
                labels, per_class[0], per_class[1], per_class[2], per_class[3], strict=True
            )
        },
        "confusion_matrix": matrix.tolist(),
        "n_samples": len(y_true),
    }


def measure_latency(
    predict_fn: Callable[[list[str]], list[Any]],
    sample_texts: Sequence[str],
    *,
    warmup_calls: int = 10,
    timed_calls: int = 100,
    seed: int = 42,
) -> dict[str, object]:
    """Single-text prediction latency under a fixed, documented protocol.

    Protocol: batch size 1; ``warmup_calls`` untimed warm-up requests on the
    first sample; then ``timed_calls`` requests, each on a single sample drawn
    uniformly with a seeded RNG; wall-clock measured with ``time.perf_counter``
    around each call; statistics are mean/median/p95 in milliseconds. The
    protocol is recorded in the returned dict so reported numbers are
    interpretable and reproducible.
    """
    texts = list(sample_texts)
    if not texts:
        raise ValueError("sample_texts must not be empty")
    if timed_calls < 1:
        raise ValueError("timed_calls must be >= 1")
    rng = np.random.default_rng(seed)
    for _ in range(warmup_calls):
        predict_fn([texts[0]])
    durations_ms = []
    for _ in range(timed_calls):
        batch = [texts[int(rng.integers(0, len(texts)))]]
        started = time.perf_counter()
        predict_fn(batch)
        durations_ms.append((time.perf_counter() - started) * 1000.0)
    durations = np.asarray(durations_ms, dtype=float)
    return {
        "protocol": {
            "batch_size": 1,
            "warmup_calls": warmup_calls,
            "timed_calls": timed_calls,
            "seed": seed,
        },
        "mean_ms": float(durations.mean()),
        "median_ms": float(np.median(durations)),
        "p95_ms": float(np.percentile(durations, 95)),
    }
