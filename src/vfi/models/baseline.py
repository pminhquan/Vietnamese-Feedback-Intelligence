"""Classical baseline models: majority class and TF-IDF + Logistic Regression.

These are traditional supervised learning baselines, not fine-tuned language
models. All training happens on the training split only; model selection uses
the validation split; the test split is evaluated separately (see
``docs/experiment-protocol.md``).
"""

from __future__ import annotations

import json
import time
import warnings
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from vfi.config import AppConfig, TfidfLogRegConfig, config_hash
from vfi.data.loading import CANONICAL_LABELS, LABEL_TO_ID, Record
from vfi.evaluation.metrics import compute_metrics

#: Model identifiers and versions recorded in artifacts and predictions.
MODEL_IDS: dict[str, str] = {
    "majority": "majority_v1",
    "tfidf-logreg": "tfidf_logreg_v1",
}

#: Small, justified hyperparameter search space (4 candidates, evaluated on
#: the validation split only). Word 1-2 grams capture local word order such
#: as negation ("không tốt"); char_wb 2-5 grams are robust to Vietnamese
#: spelling variants. ``min_df``/``max_df`` stay fixed at corpus-scale priors.
SEARCH_SPACE: list[dict[str, Any]] = [
    {"analyzer": "word", "ngram_range": (1, 2), "C": 1.0},
    {"analyzer": "word", "ngram_range": (1, 2), "C": 4.0},
    {"analyzer": "char_wb", "ngram_range": (2, 5), "C": 1.0},
    {"analyzer": "char_wb", "ngram_range": (2, 5), "C": 4.0},
]


class MajorityBaseline:
    """Predicts the most frequent label seen during fitting.

    Deterministic tie-break: among equally frequent labels the one with the
    lowest canonical id wins. Probabilities are the degenerate one-hot of the
    majority class (reported honestly as constant, not learned).
    """

    def __init__(self) -> None:
        self.majority_label: str | None = None

    def fit(self, records: Sequence[Record]) -> "MajorityBaseline":
        counts: Counter[str] = Counter(record.label for record in records)
        if not counts:
            raise ValueError("Cannot fit MajorityBaseline on an empty record list")
        self.majority_label = min(counts, key=lambda label: (-counts[label], LABEL_TO_ID[label]))
        return self

    def predict(self, texts: Sequence[str]) -> list[str]:
        if self.majority_label is None:
            raise RuntimeError("MajorityBaseline must be fitted before predict")
        return [self.majority_label] * len(texts)

    def predict_proba(self, texts: Sequence[str]) -> list[list[float]]:
        if self.majority_label is None:
            raise RuntimeError("MajorityBaseline must be fitted before predict_proba")
        rows = [[0.0] * len(CANONICAL_LABELS) for _ in texts]
        for row in rows:
            row[LABEL_TO_ID[self.majority_label]] = 1.0
        return rows


def build_tfidf_logreg_pipeline(config: TfidfLogRegConfig) -> Pipeline:
    """Build an unfitted sklearn Pipeline: TF-IDF features -> Logistic Regression.

    The TF-IDF vocabulary is learned inside ``pipeline.fit`` on the training
    texts only; validation/test texts are transformed with that vocabulary.
    """
    vectorizer = TfidfVectorizer(
        analyzer=config.analyzer,
        ngram_range=tuple(config.ngram_range),
        min_df=config.min_df,
        max_df=config.max_df,
        sublinear_tf=config.sublinear_tf,
        lowercase=True,
    )
    classifier = LogisticRegression(
        C=config.C,
        max_iter=config.max_iter,
        class_weight=config.class_weight,
        random_state=config.random_state,
    )
    return Pipeline([("tfidf", vectorizer), ("clf", classifier)])


@dataclass(frozen=True)
class SearchTrial:
    """One hyperparameter candidate and its validation macro-F1."""

    params: dict[str, Any]
    val_macro_f1: float
    fit_seconds: float


def run_small_search(
    train_records: Sequence[Record],
    validation_records: Sequence[Record],
    base_config: TfidfLogRegConfig,
) -> tuple[Pipeline, list[SearchTrial]]:
    """Evaluate the small search space on the validation split.

    Returns the pipeline refit on the training split with the best
    hyperparameters (ties break to the first candidate in ``SEARCH_SPACE``).
    The validation split is used for selection only and is never refit on.
    """
    x_train = [record.text for record in train_records]
    y_train = [record.label for record in train_records]
    x_val = [record.text for record in validation_records]
    y_val = [record.label for record in validation_records]

    trials: list[SearchTrial] = []
    best: SearchTrial | None = None
    for override in SEARCH_SPACE:
        candidate = replace(base_config, **override)
        pipeline = build_tfidf_logreg_pipeline(candidate)
        started = time.perf_counter()
        pipeline.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - started
        val_metrics = compute_val_macro_f1(pipeline, x_val, y_val)
        trial = SearchTrial(
            params=override,
            val_macro_f1=val_metrics,
            fit_seconds=fit_seconds,
        )
        trials.append(trial)
        if best is None or trial.val_macro_f1 > best.val_macro_f1:
            best = trial

    assert best is not None  # SEARCH_SPACE is non-empty
    best_pipeline = build_tfidf_logreg_pipeline(replace(base_config, **best.params))
    best_pipeline.fit(x_train, y_train)
    return best_pipeline, trials


def compute_val_macro_f1(pipeline: Pipeline, texts: Sequence[str], labels: Sequence[str]) -> float:
    """Validation macro-F1 for selection (zero_division=0, canonical labels)."""
    predictions = pipeline.predict(list(texts))
    return compute_metrics(list(labels), list(predictions))["macro_f1"]


def save_model(
    model_dir: str | Path,
    model_id: str,
    model: Any,
    metadata: dict[str, Any],
) -> Path:
    """Persist a fitted model plus a metadata sidecar under ``model_dir``.

    The sidecar (``meta.json``) carries the trusted-artifact provenance
    (versions, config hash, dataset fingerprints, timestamps) that
    :func:`load_model` checks before deserializing.
    """
    if model_id not in MODEL_IDS.values():
        raise ValueError(f"Unknown model_id {model_id!r}; expected one of {sorted(MODEL_IDS.values())}")
    out_dir = Path(model_dir) / model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")
    (out_dir / "meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return out_dir


def load_model(model_dir: str | Path, model_id: str) -> tuple[Any, dict[str, Any]]:
    """Load a model artifact saved by :func:`save_model`.

    Trusted-artifact policy: only load artifacts produced by this project's
    ``vfi train`` command from the local ``models/`` directory, and only
    after the metadata sidecar passes these checks — known model_id, present
    config hash, and a matching scikit-learn major version (mismatch warns;
    a different major version can change predictions, so retrain instead).

    Raises:
        FileNotFoundError: with a clear message when the artifact or its
            metadata sidecar is missing.
    """
    if model_id not in MODEL_IDS.values():
        raise FileNotFoundError(f"Unknown model_id {model_id!r}; expected one of {sorted(MODEL_IDS.values())}")
    artifact_dir = Path(model_dir) / model_id
    meta_path = artifact_dir / "meta.json"
    model_path = artifact_dir / "model.joblib"
    if not meta_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            f"No trained artifact for {model_id!r} under {artifact_dir}. "
            "Run 'vfi train' first; only load artifacts produced by this project."
        )
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if "config_hash" not in metadata:
        raise FileNotFoundError(f"Artifact metadata for {model_id!r} is missing 'config_hash'")
    if metadata.get("sklearn_version") != sklearn.__version__:
        warnings.warn(
            f"Artifact {model_id!r} was trained with scikit-learn "
            f"{metadata.get('sklearn_version')} but installed is {sklearn.__version__}; "
            "retrain before trusting predictions across major versions.",
            UserWarning,
            stacklevel=2,
        )
    model = joblib.load(model_path)
    return model, metadata


def build_training_metadata(
    app_config: AppConfig,
    dataset_fingerprints: dict[str, str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble provenance metadata recorded next to every trained model."""
    metadata: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config_to_dict_safe(app_config),
        "config_hash": config_hash(app_config),
        "dataset_fingerprints": dataset_fingerprints,
        "sklearn_version": sklearn.__version__,
        "python_version": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
        "seed": app_config.data.seed,
    }
    if extra:
        metadata.update(extra)
    return metadata


def config_to_dict_safe(config: AppConfig) -> dict[str, Any]:
    """JSON-safe config dict (tuples become lists via asdict -> json)."""
    return json.loads(json.dumps(asdict(config), sort_keys=True))
