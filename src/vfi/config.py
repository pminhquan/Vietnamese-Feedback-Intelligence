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
    # Dataset independence: which adapter loads the data (see
    # vfi.data.adapters). The default keeps the historical UIT-VSFC layout;
    # the project no longer depends on it.
    dataset_adapter: str = "uit_vsfc"
    filename: str = ""
    # Raw-label provenance gate: training through the uit_vsfc adapter is
    # refused unless the mapping is declared as synthetic_fixture (tests) or
    # owner_confirmed (documented evidence) - the SEACrowd reading is
    # UNRESOLVED and never accepted for training by itself.
    label_mapping_source: str = ""
    label_mapping_evidence: str = ""
    label_mapping: str = ""
    # NEU-ESC toxic handling: Toxic is a distinct class and is never mapped
    # into the 3-class space; exclude/keep require a written justification.
    toxic_policy: str = "error"
    toxic_justification: str = ""


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


_DATASET_ADAPTERS = ("uit_vsfc", "five_level_csv", "neu_esc", "custom_jsonl")
_LABEL_MAPPING_SOURCES = ("synthetic_fixture", "owner_confirmed")


def _require_bool(raw: dict[str, Any], key: str, default: bool) -> bool:
    """Strict boolean parsing: YAML booleans only, 'true'/'false' strings rejected."""
    value = raw.get(key, default)
    if not isinstance(value, bool):
        raise ConfigError(
            f"data.{key} must be a real boolean (true/false), got {value!r}; "
            "quote-free YAML booleans are required"
        )
    return value


def _parse_data(raw: Any) -> DataConfig:
    raw = raw or {}
    allowed_data_keys = {
        "raw_dir",
        "use_official_splits",
        "validation_size",
        "seed",
        "dataset_adapter",
        "filename",
        "label_mapping_source",
        "label_mapping_evidence",
        "label_mapping",
        "toxic_policy",
        "toxic_justification",
    }
    unknown = sorted(set(raw) - allowed_data_keys)
    if unknown:
        raise ConfigError(
            f"unknown data config key(s): {unknown}; allowed: {sorted(allowed_data_keys)}"
        )
    validation_size = float(raw.get("validation_size", DataConfig.validation_size))
    if not 0.0 < validation_size < 1.0:
        raise ConfigError("data.validation_size must be strictly between 0 and 1")
    seed_value = raw.get("seed", DataConfig.seed)
    if isinstance(seed_value, bool) or not isinstance(seed_value, int):
        raise ConfigError(f"data.seed must be an integer, got {seed_value!r}")
    if seed_value < 0:
        raise ConfigError("data.seed must be a non-negative integer")
    adapter = str(raw.get("dataset_adapter", DataConfig.dataset_adapter))
    if adapter not in _DATASET_ADAPTERS:
        raise ConfigError(
            f"data.dataset_adapter must be one of {_DATASET_ADAPTERS}, got {adapter!r}"
        )
    label_source = str(raw.get("label_mapping_source", ""))
    label_evidence = str(raw.get("label_mapping_evidence", ""))
    if label_source and label_source not in _LABEL_MAPPING_SOURCES:
        raise ConfigError(
            f"data.label_mapping_source must be one of {_LABEL_MAPPING_SOURCES} "
            "(the unverified SEACrowd reading is never a training-approved source)"
        )
    if (label_source and not label_evidence.strip()) or (label_evidence and not label_source):
        raise ConfigError(
            "data.label_mapping_evidence is required with label_mapping_source "
            "(record where the mapping was confirmed, e.g. author reply or audit run id)"
        )
    label_mapping = str(raw.get("label_mapping", ""))
    if label_source and not label_mapping.strip():
        raise ConfigError(
            "data.label_mapping is required with label_mapping_source (a compact spec "
            "like '0:positive,1:negative,2:neutral'); the mapping itself must be "
            "declared, never assumed"
        )
    if label_mapping and not label_source:
        raise ConfigError(
            "data.label_mapping requires data.label_mapping_source "
            "(a mapping without provenance is unverifiable)"
        )
    toxic_policy = str(raw.get("toxic_policy", DataConfig.toxic_policy))
    if toxic_policy not in ("error", "exclude", "keep"):
        raise ConfigError("data.toxic_policy must be one of ('error', 'exclude', 'keep')")
    toxic_justification = str(raw.get("toxic_justification", ""))
    if toxic_policy in ("exclude", "keep") and not toxic_justification.strip():
        raise ConfigError(
            f"data.toxic_justification is required for toxic_policy={toxic_policy!r} "
            "(write down the scientific reason for excluding or keeping toxic records)"
        )
    return DataConfig(
        raw_dir=str(raw.get("raw_dir", DataConfig.raw_dir)),
        use_official_splits=_require_bool(raw, "use_official_splits", True),
        validation_size=validation_size,
        seed=seed_value,
        dataset_adapter=adapter,
        filename=str(raw.get("filename", "")),
        label_mapping_source=label_source,
        label_mapping_evidence=label_evidence,
        label_mapping=label_mapping,
        toxic_policy=toxic_policy,
        toxic_justification=toxic_justification,
    )


def _parse_tfidf_logreg(raw: Any) -> TfidfLogRegConfig:
    raw = raw or {}
    defaults = TfidfLogRegConfig()
    unknown = sorted(
        set(raw)
        - {
            "analyzer",
            "ngram_range",
            "min_df",
            "max_df",
            "sublinear_tf",
            "C",
            "max_iter",
            "class_weight",
            "random_state",
        }
    )
    if unknown:
        raise ConfigError(f"unknown models.tfidf_logreg key(s): {unknown}")
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
    class_weight = raw.get("class_weight", defaults.class_weight)
    if class_weight is not None and class_weight != "balanced":
        raise ConfigError(
            f"models.tfidf_logreg.class_weight must be 'balanced' or null, got {class_weight!r}"
        )
    sublinear_tf = raw.get("sublinear_tf", defaults.sublinear_tf)
    if not isinstance(sublinear_tf, bool):
        raise ConfigError(
            f"models.tfidf_logreg.sublinear_tf must be a real boolean, got {sublinear_tf!r}"
        )
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
    raw = raw or {}
    unknown = sorted(set(raw) - {"model_dir", "reports_dir"})
    if unknown:
        raise ConfigError(f"unknown output config key(s): {unknown}")
    return OutputConfig(
        model_dir=str(raw.get("model_dir", OutputConfig.model_dir)),
        reports_dir=str(raw.get("reports_dir", OutputConfig.reports_dir)),
    )


def load_config(path: str | Path) -> AppConfig:
    """Load and validate a YAML configuration file.

    Strict validation: unknown top-level or nested keys, string booleans,
    and out-of-range values are rejected with informative errors. Missing
    sections fall back to documented defaults.

    Raises:
        ConfigError: if the file is missing, unreadable YAML, or invalid.
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
    unknown_top = sorted(set(raw) - {"data", "models", "output"})
    if unknown_top:
        raise ConfigError(f"unknown top-level config section(s): {unknown_top}")
    models_raw = raw.get("models", {})
    if not isinstance(models_raw, dict):
        raise ConfigError("the 'models' section must be a mapping")
    unknown_models = sorted(set(models_raw) - {"tfidf_logreg"})
    if unknown_models:
        raise ConfigError(f"unknown models config key(s): {unknown_models}")
    return AppConfig(
        data=_parse_data(raw.get("data")),
        tfidf_logreg=_parse_tfidf_logreg(models_raw.get("tfidf_logreg")),
        output=_parse_output(raw.get("output")),
    )
