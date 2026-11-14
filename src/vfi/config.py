"""Configuration loading and validation for VFI experiments."""

from __future__ import annotations

import hashlib
import json
import math
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
    class_weight: str | None = "balanced"
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


def _ensure_mapping(raw: Any, section: str) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{section} must be a mapping, got {type(raw).__name__}")
    return raw


def _parse_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{field_name} must be a boolean, got {value!r}")
    return value


def _parse_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{field_name} must be an integer, got {value!r}")
    return int(value)


def _parse_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{field_name} must be a finite number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"{field_name} must be a finite number, got {value!r}")
    return result


def _parse_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field_name} must be a non-empty string, got {value!r}")
    return value


def _parse_data(raw: Any) -> DataConfig:
    raw = _ensure_mapping(raw, "data")
    raw_dir = _parse_str(raw.get("raw_dir", DataConfig.raw_dir), "data.raw_dir")
    use_official_splits = _parse_bool(
        raw.get("use_official_splits", DataConfig.use_official_splits),
        "data.use_official_splits",
    )
    validation_size = _parse_float(
        raw.get("validation_size", DataConfig.validation_size),
        "data.validation_size",
    )
    if not 0.0 < validation_size < 1.0:
        raise ConfigError("data.validation_size must be strictly between 0 and 1")
    seed = _parse_int(raw.get("seed", DataConfig.seed), "data.seed")
    if seed < 0:
        raise ConfigError("data.seed must be a non-negative integer")
    return DataConfig(
        raw_dir=raw_dir,
        use_official_splits=use_official_splits,
        validation_size=validation_size,
        seed=seed,
    )


def _parse_tfidf_logreg(raw: Any) -> TfidfLogRegConfig:
    raw = _ensure_mapping(raw, "models.tfidf_logreg")
    defaults = TfidfLogRegConfig()
    analyzer = raw.get("analyzer", defaults.analyzer)
    if not isinstance(analyzer, str) or analyzer not in _ALLOWED_ANALYZERS:
        raise ConfigError(
            f"models.tfidf_logreg.analyzer must be one of {_ALLOWED_ANALYZERS}, got {analyzer!r}"
        )
    ngram_raw = raw.get("ngram_range", defaults.ngram_range)
    if (
        not isinstance(ngram_raw, (list, tuple))
        or len(ngram_raw) != 2
        or any(isinstance(bound, bool) or not isinstance(bound, int) for bound in ngram_raw)
    ):
        raise ConfigError("models.tfidf_logreg.ngram_range must be a pair of integers")
    ngram_range = (int(ngram_raw[0]), int(ngram_raw[1]))
    if ngram_range[0] < 1 or ngram_range[1] < ngram_range[0]:
        raise ConfigError("models.tfidf_logreg.ngram_range must satisfy 1 <= min <= max")
    min_df = _parse_int(raw.get("min_df", defaults.min_df), "models.tfidf_logreg.min_df")
    if min_df < 1:
        raise ConfigError("models.tfidf_logreg.min_df must be >= 1")
    max_df = _parse_float(raw.get("max_df", defaults.max_df), "models.tfidf_logreg.max_df")
    if not 0.0 < max_df <= 1.0:
        raise ConfigError("models.tfidf_logreg.max_df must be in (0, 1]")
    sublinear_tf = _parse_bool(
        raw.get("sublinear_tf", defaults.sublinear_tf),
        "models.tfidf_logreg.sublinear_tf",
    )
    c_value = _parse_float(raw.get("C", defaults.C), "models.tfidf_logreg.C")
    if c_value <= 0:
        raise ConfigError("models.tfidf_logreg.C must be > 0")
    max_iter = _parse_int(raw.get("max_iter", defaults.max_iter), "models.tfidf_logreg.max_iter")
    if max_iter < 1:
        raise ConfigError("models.tfidf_logreg.max_iter must be >= 1")
    random_state = _parse_int(
        raw.get("random_state", defaults.random_state),
        "models.tfidf_logreg.random_state",
    )
    if random_state < 0:
        raise ConfigError("models.tfidf_logreg.random_state must be a non-negative integer")
    class_weight = raw.get("class_weight", defaults.class_weight)
    if class_weight not in (None, "balanced"):
        raise ConfigError("models.tfidf_logreg.class_weight must be 'balanced' or null")
    return TfidfLogRegConfig(
        analyzer=analyzer,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        C=c_value,
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=random_state,
    )


def _parse_output(raw: Any) -> OutputConfig:
    raw = _ensure_mapping(raw, "output")
    return OutputConfig(
        model_dir=_parse_str(raw.get("model_dir", OutputConfig.model_dir), "output.model_dir"),
        reports_dir=_parse_str(
            raw.get("reports_dir", OutputConfig.reports_dir),
            "output.reports_dir",
        ),
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
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"Invalid YAML in {config_path}: {exc}") from exc
    raw = _ensure_mapping(loaded, "Configuration root")
    models_raw = _ensure_mapping(raw.get("models"), "models")
    return AppConfig(
        data=_parse_data(raw.get("data")),
        tfidf_logreg=_parse_tfidf_logreg(models_raw.get("tfidf_logreg")),
        output=_parse_output(raw.get("output")),
    )
