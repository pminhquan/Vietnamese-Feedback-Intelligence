"""Configuration loading and validation for VFI experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

_ALLOWED_ANALYZERS = ("word", "char", "char_wb")


class ConfigError(Exception):
    """Raised when a configuration file is missing or invalid."""


@dataclass(frozen=True)
class DataConfig:
    """Where raw data lives and how splits are derived.

    ``use_official_splits=True`` uses the official UIT-VSFC train/validation
    (dev)/test files. When ``False``, a deterministic stratified validation
    split is carved out of the official train file with ``validation_size``
    and ``seed`` (the official test file is still used for testing).
    """

    raw_dir: str = "data/raw/uit_vsfc"
    use_official_splits: bool = True
    validation_size: float = 0.2
    seed: int = 42


@dataclass(frozen=True)
class TfidfLogRegConfig:
    """Hyperparameters of the TF-IDF + Logistic Regression baseline."""

    analyzer: str = "word"
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 2
    max_df: float = 0.95
    sublinear_tf: bool = True
    C: float = 4.0
    max_iter: int = 1000
    class_weight: str = "balanced"
    random_state: int = 42


@dataclass(frozen=True)
class OutputConfig:
    model_dir: str = "models"
    reports_dir: str = "reports/artifacts"


@dataclass(frozen=True)
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    tfidf_logreg: TfidfLogRegConfig = field(default_factory=TfidfLogRegConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def config_to_dict(config: AppConfig) -> dict[str, Any]:
    """Serialize a config to plain nested dicts (JSON/YAML compatible)."""
    return asdict(config)


def config_hash(config: AppConfig) -> str:
    """Stable short hash identifying the full configuration."""
    payload = json.dumps(config_to_dict(config), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _parse_data(raw: Any) -> DataConfig:
    raw = raw or {}
    validation_size = float(raw.get("validation_size", DataConfig.validation_size))
    if not 0.0 < validation_size < 1.0:
        raise ConfigError("data.validation_size must be strictly between 0 and 1")
    seed = int(raw.get("seed", DataConfig.seed))
    if seed < 0:
        raise ConfigError("data.seed must be a non-negative integer")
    return DataConfig(
        raw_dir=str(raw.get("raw_dir", DataConfig.raw_dir)),
        use_official_splits=bool(raw.get("use_official_splits", True)),
        validation_size=validation_size,
        seed=seed,
    )


def _parse_tfidf_logreg(raw: Any) -> TfidfLogRegConfig:
    raw = raw or {}
    defaults = TfidfLogRegConfig()
    analyzer = str(raw.get("analyzer", defaults.analyzer))
    if analyzer not in _ALLOWED_ANALYZERS:
        raise ConfigError(
            f"models.tfidf_logreg.analyzer must be one of {_ALLOWED_ANALYZERS}, got {analyzer!r}"
        )
    ngram_raw = raw.get("ngram_range", defaults.ngram_range)
    try:
        ngram_range = (int(ngram_raw[0]), int(ngram_raw[1]))
    except (TypeError, IndexError) as exc:
        raise ConfigError("models.tfidf_logreg.ngram_range must be a pair of integers") from exc
    if ngram_range[0] < 1 or ngram_range[1] < ngram_range[0]:
        raise ConfigError("models.tfidf_logreg.ngram_range must satisfy 1 <= min <= max")
    min_df = int(raw.get("min_df", defaults.min_df))
    if min_df < 1:
        raise ConfigError("models.tfidf_logreg.min_df must be >= 1")
    max_df = float(raw.get("max_df", defaults.max_df))
    if not 0.0 < max_df <= 1.0:
        raise ConfigError("models.tfidf_logreg.max_df must be in (0, 1]")
    c_value = float(raw.get("C", defaults.C))
    if c_value <= 0:
        raise ConfigError("models.tfidf_logreg.C must be > 0")
    max_iter = int(raw.get("max_iter", defaults.max_iter))
    if max_iter < 1:
        raise ConfigError("models.tfidf_logreg.max_iter must be >= 1")
    random_state = int(raw.get("random_state", defaults.random_state))
    if random_state < 0:
        raise ConfigError("models.tfidf_logreg.random_state must be a non-negative integer")
    class_weight = str(raw.get("class_weight", defaults.class_weight))
    if class_weight not in (None, "balanced"):
        raise ConfigError("models.tfidf_logreg.class_weight must be 'balanced' or null")
    return TfidfLogRegConfig(
        analyzer=analyzer,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=bool(raw.get("sublinear_tf", defaults.sublinear_tf)),
        C=c_value,
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=random_state,
    )


def _parse_output(raw: Any) -> OutputConfig:
    raw = raw or {}
    return OutputConfig(
        model_dir=str(raw.get("model_dir", OutputConfig.model_dir)),
        reports_dir=str(raw.get("reports_dir", OutputConfig.reports_dir)),
    )


def load_config(path: str | Path) -> AppConfig:
    """Load and validate a YAML configuration file.

    Raises:
        ConfigError: if the file is missing, unreadable YAML, or contains
            out-of-range values.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Configuration file not found: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"Configuration root must be a mapping, got {type(raw).__name__}")
    return AppConfig(
        data=_parse_data(raw.get("data")),
        tfidf_logreg=_parse_tfidf_logreg(raw.get("models", {}).get("tfidf_logreg")),
        output=_parse_output(raw.get("output")),
    )
