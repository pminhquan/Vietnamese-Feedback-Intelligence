"""Tests for metrics correctness on hand-computed examples."""

from __future__ import annotations

import math

import pytest
from tests.conftest import make_records

from vfi.evaluation.metrics import compute_metrics, measure_latency
from vfi.models.baseline import MajorityBaseline


class TestComputeMetrics:
    Y_TRUE = ["positive", "positive", "negative", "neutral"]
    Y_PRED = ["positive", "negative", "negative", "neutral"]

    def test_accuracy(self) -> None:
        metrics = compute_metrics(self.Y_TRUE, self.Y_PRED)
        # Matches: index 0 (positive) and indices 2, 3 (negative, neutral).
        assert metrics["accuracy"] == pytest.approx(0.75)

    def test_per_class_hand_computed(self) -> None:
        metrics = compute_metrics(self.Y_TRUE, self.Y_PRED)
        per_class = metrics["per_class"]
        # negative: predicted {1,2}, true {2}
        assert per_class["negative"]["precision"] == pytest.approx(0.5)
        assert per_class["negative"]["recall"] == pytest.approx(1.0)
        assert per_class["negative"]["f1"] == pytest.approx(2 * 0.5 * 1.0 / 1.5)
        assert per_class["negative"]["support"] == 1
        # neutral: predicted {3}, true {3}
        assert per_class["neutral"]["f1"] == pytest.approx(1.0)
        # positive: predicted {0}, true {0,1}
        assert per_class["positive"]["precision"] == pytest.approx(1.0)
        assert per_class["positive"]["recall"] == pytest.approx(0.5)
        assert per_class["positive"]["support"] == 2

    def test_macro_values(self) -> None:
        metrics = compute_metrics(self.Y_TRUE, self.Y_PRED)
        assert metrics["macro_precision"] == pytest.approx((1.0 + 0.5 + 1.0) / 3)
        assert metrics["macro_recall"] == pytest.approx((0.5 + 1.0 + 1.0) / 3)
        f1_neg = 2 * 0.5 * 1.0 / 1.5
        f1_pos = 2 * 1.0 * 0.5 / 1.5
        assert metrics["macro_f1"] == pytest.approx((f1_neg + 1.0 + f1_pos) / 3)

    def test_confusion_matrix_layout(self) -> None:
        # Canonical label order: negative=0, neutral=1, positive=2.
        metrics = compute_metrics(self.Y_TRUE, self.Y_PRED)
        matrix = metrics["confusion_matrix"]
        assert matrix == [
            [1, 0, 0],  # true negative -> predicted negative
            [0, 1, 0],  # true neutral  -> predicted neutral
            [1, 0, 1],  # true positive -> one predicted negative, one positive
        ]

    def test_zero_division_is_zero_not_nan(self) -> None:
        metrics = compute_metrics(["positive", "negative"], ["negative", "negative"])
        assert metrics["per_class"]["neutral"]["precision"] == 0.0
        assert metrics["per_class"]["neutral"]["recall"] == 0.0
        assert metrics["per_class"]["neutral"]["f1"] == 0.0
        assert math.isnan(metrics["per_class"]["neutral"]["f1"]) is False

    def test_perfect_predictions(self) -> None:
        metrics = compute_metrics(self.Y_TRUE, self.Y_TRUE)
        assert metrics["macro_f1"] == pytest.approx(1.0)
        assert metrics["accuracy"] == pytest.approx(1.0)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="lengths differ"):
            compute_metrics(["positive"], ["positive", "negative"])


class TestMeasureLatency:
    def test_protocol_reported_and_positive(self) -> None:
        model = MajorityBaseline().fit(make_records(2))
        texts = [r.text for r in make_records(2)]
        latency = measure_latency(model.predict, texts, warmup_calls=2, timed_calls=5)
        assert latency["protocol"] == {
            "batch_size": 1,
            "warmup_calls": 2,
            "timed_calls": 5,
            "seed": 42,
        }
        assert latency["median_ms"] > 0
        assert latency["p95_ms"] >= latency["median_ms"]

    def test_empty_samples_raise(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            measure_latency(lambda batch: batch, [])

    def test_zero_timed_calls_raise(self) -> None:
        with pytest.raises(ValueError, match="timed_calls"):
            measure_latency(lambda batch: batch, ["a"], timed_calls=0)
