"""Tests for strict configuration validation (P1 correctness)."""

from pathlib import Path

import pytest

from vfi.config import ConfigError, config_hash, load_config


def _write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")
    return path


BASE = """
data:
  raw_dir: {raw}
  label_mapping_source: synthetic_fixture
  label_mapping: "0:positive,1:negative,2:neutral"
  label_mapping_evidence: synthetic fixture
"""


class TestStrictValidation:
    def test_unknown_top_level_section_rejected(self, tmp_path: Path) -> None:
        path = _write(tmp_path, BASE.format(raw="data/raw") + "deploy:\n  key: value\n")
        with pytest.raises(ConfigError, match="unknown top-level config section"):
            load_config(path)

    def test_unknown_data_key_rejected(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            BASE.format(raw="data/raw") + "  dataset_adapter: uit_vsfc\n  bogus: 1\n",
        )
        with pytest.raises(ConfigError, match="unknown data config key"):
            load_config(path)

    def test_boolean_string_rejected(self, tmp_path: Path) -> None:
        path = _write(tmp_path, BASE.format(raw="data/raw") + "  use_official_splits: 'true'\n")
        with pytest.raises(ConfigError, match="real boolean"):
            load_config(path)

    def test_unknown_models_key_rejected(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            BASE.format(raw="data/raw") + "models:\n  tfidf_logreg: {}\n  svm: {}\n",
        )
        with pytest.raises(ConfigError, match="unknown models config key"):
            load_config(path)

    def test_class_weight_null_accepted_string_rejected(self, tmp_path: Path) -> None:
        ok = _write(
            tmp_path,
            BASE.format(raw="data/raw") + "models:\n  tfidf_logreg:\n    class_weight: null\n",
        )
        assert load_config(ok).tfidf_logreg.class_weight is None
        bad = _write(
            tmp_path,
            BASE.format(raw="data/raw") + "models:\n  tfidf_logreg:\n    class_weight: none\n",
        )
        with pytest.raises(ConfigError, match="class_weight"):
            load_config(bad)

    def test_sublinear_tf_boolean_string_rejected(self, tmp_path: Path) -> None:
        bad = _write(
            tmp_path,
            BASE.format(raw="data/raw") + "models:\n  tfidf_logreg:\n    sublinear_tf: 'yes'\n",
        )
        with pytest.raises(ConfigError, match="sublinear_tf"):
            load_config(bad)

    def test_missing_sections_fall_back_to_defaults(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "data:\n  raw_dir: data/raw\n")
        config = load_config(path)
        assert config.data.dataset_adapter == "uit_vsfc"
        assert config.tfidf_logreg.C == 4.0
        assert config.output.model_dir == "models"


class TestLabelGate:
    def test_source_without_mapping_rejected(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            "data:\n  raw_dir: x\n  label_mapping_source: owner_confirmed\n"
            "  label_mapping_evidence: email 2026-11-07\n",
        )
        with pytest.raises(ConfigError, match="label_mapping is required"):
            load_config(path)

    def test_mapping_without_source_rejected(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path, "data:\n  raw_dir: x\n  label_mapping: '0:negative,1:neutral,2:positive'\n"
        )
        with pytest.raises(ConfigError, match="requires data.label_mapping_source"):
            load_config(path)

    def test_evidence_required(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            "data:\n  raw_dir: x\n  label_mapping_source: owner_confirmed\n"
            "  label_mapping: '0:negative,1:neutral,2:positive'\n",
        )
        with pytest.raises(ConfigError, match="label_mapping_evidence is required"):
            load_config(path)

    def test_unverified_source_rejected(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            "data:\n  raw_dir: x\n  label_mapping_source: seacrowd_unverified\n"
            "  label_mapping: '0:positive,1:negative,2:neutral'\n  label_mapping_evidence: x\n",
        )
        with pytest.raises(ConfigError, match="label_mapping_source must be one of"):
            load_config(path)


class TestToxicPolicy:
    def test_exclude_requires_justification(self, tmp_path: Path) -> None:
        path = _write(tmp_path, BASE.format(raw="data/raw") + "  toxic_policy: exclude\n")
        with pytest.raises(ConfigError, match="toxic_justification is required"):
            load_config(path)

    def test_exclude_with_justification_accepted(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            BASE.format(raw="data/raw")
            + "  toxic_policy: exclude\n  toxic_justification: documented pilot decision\n",
        )
        config = load_config(path)
        assert config.data.toxic_policy == "exclude"

    def test_invalid_policy_rejected(self, tmp_path: Path) -> None:
        path = _write(tmp_path, BASE.format(raw="data/raw") + "  toxic_policy: merge\n")
        with pytest.raises(ConfigError, match="toxic_policy must be one of"):
            load_config(path)


class TestDeterminism:
    def test_config_hash_stable(self, tmp_path: Path) -> None:
        first = _write(tmp_path, BASE.format(raw="data/raw"))
        second = tmp_path / "other.yaml"
        second.write_text(BASE.format(raw="data/raw"), encoding="utf-8")
        assert config_hash(load_config(first)) == config_hash(load_config(second))
