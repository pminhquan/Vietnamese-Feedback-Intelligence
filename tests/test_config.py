"""Tests for YAML configuration loading, validation, and hashing."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from vfi.config import (
    AppConfig,
    ConfigError,
    DataConfig,
    OutputConfig,
    TfidfLogRegConfig,
    config_hash,
    config_to_dict,
    load_config,
)


def _write_yaml(tmp_path: Path, content: str, name: str = "config.yaml") -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


class TestBaselineAndCompatibility:
    def test_baseline_yaml_matches_defaults_and_hash(self) -> None:
        baseline_path = Path(__file__).resolve().parents[1] / "configs" / "baseline.yaml"
        loaded = load_config(baseline_path)
        defaults = AppConfig()
        assert loaded == defaults
        assert config_to_dict(loaded) == config_to_dict(defaults)

        loaded_hash = config_hash(loaded)
        assert loaded_hash == config_hash(defaults) == "ed23ff1174968172"
        assert re.fullmatch(r"[0-9a-f]{16}", loaded_hash)

        modified = AppConfig(tfidf_logreg=TfidfLogRegConfig(class_weight=None))
        assert config_hash(modified) != loaded_hash

    @pytest.mark.parametrize(
        "content",
        [
            "",
            "# comment only\n",
            "null\n",
            "data: null\nmodels: null\noutput: null\n",
            "models:\n  tfidf_logreg: null\n",
        ],
    )
    def test_omitted_and_null_sections_keep_defaults(self, tmp_path: Path, content: str) -> None:
        path = _write_yaml(tmp_path, content)
        assert load_config(path) == AppConfig()

    def test_partial_section_preserves_sibling_defaults(self, tmp_path: Path) -> None:
        path = _write_yaml(
            tmp_path,
            "data:\n  seed: 7\nmodels:\n  tfidf_logreg:\n    C: 2\n    max_df: 1\n",
        )
        loaded = load_config(path)
        assert loaded.data == DataConfig(seed=7)
        assert loaded.tfidf_logreg == TfidfLogRegConfig(C=2.0, max_df=1.0)
        assert loaded.output == OutputConfig()

    def test_missing_file_and_invalid_yaml_syntax_raise_config_error(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="not found"):
            load_config(tmp_path / "missing.yaml")

        bad_yaml = _write_yaml(tmp_path, "data: [unclosed\n")
        with pytest.raises(ConfigError, match="Invalid YAML"):
            load_config(bad_yaml)


class TestClassWeight:
    def test_class_weight_null_accepted(self, tmp_path: Path) -> None:
        path = _write_yaml(
            tmp_path,
            "models:\n  tfidf_logreg:\n    class_weight: null\n",
        )
        loaded = load_config(path)
        assert loaded.tfidf_logreg.class_weight is None

    def test_class_weight_balanced_accepted(self, tmp_path: Path) -> None:
        path = _write_yaml(
            tmp_path,
            "models:\n  tfidf_logreg:\n    class_weight: balanced\n",
        )
        loaded = load_config(path)
        assert loaded.tfidf_logreg.class_weight == "balanced"

    @pytest.mark.parametrize("bad_value", ['"None"', '"unbalanced"', "123", "true", "[]"])
    def test_class_weight_invalid_rejected(self, tmp_path: Path, bad_value: str) -> None:
        path = _write_yaml(
            tmp_path,
            f"models:\n  tfidf_logreg:\n    class_weight: {bad_value}\n",
        )
        with pytest.raises(ConfigError, match="class_weight"):
            load_config(path)


class TestBooleanValidation:
    @pytest.mark.parametrize("flag", [True, False])
    def test_valid_booleans_accepted(self, tmp_path: Path, flag: bool) -> None:
        yaml_flag = "true" if flag else "false"
        path = _write_yaml(
            tmp_path,
            (
                f"data:\n  use_official_splits: {yaml_flag}\n"
                f"models:\n  tfidf_logreg:\n    sublinear_tf: {yaml_flag}\n"
            ),
        )
        loaded = load_config(path)
        assert loaded.data.use_official_splits is flag
        assert loaded.tfidf_logreg.sublinear_tf is flag

    @pytest.mark.parametrize(
        "bad_bool",
        ['"false"', '"true"', '"garbage"', "0", "1", "null", "[]"],
    )
    def test_invalid_use_official_splits_rejected(self, tmp_path: Path, bad_bool: str) -> None:
        path = _write_yaml(tmp_path, f"data:\n  use_official_splits: {bad_bool}\n")
        with pytest.raises(ConfigError, match="use_official_splits"):
            load_config(path)

    @pytest.mark.parametrize(
        "bad_bool",
        ['"false"', '"true"', '"garbage"', "0", "1", "null", "[]"],
    )
    def test_invalid_sublinear_tf_rejected(self, tmp_path: Path, bad_bool: str) -> None:
        path = _write_yaml(
            tmp_path,
            f"models:\n  tfidf_logreg:\n    sublinear_tf: {bad_bool}\n",
        )
        with pytest.raises(ConfigError, match="sublinear_tf"):
            load_config(path)


class TestMalformedSections:
    @pytest.mark.parametrize(
        "root_yaml", ["[]\n", "[1, 2]\n", '"scalar"\n', '""\n', "false\n", "0\n"]
    )
    def test_malformed_root_rejected(self, tmp_path: Path, root_yaml: str) -> None:
        path = _write_yaml(tmp_path, root_yaml)
        with pytest.raises(ConfigError, match="mapping"):
            load_config(path)

    @pytest.mark.parametrize("section_val", ["[]", "[1]", '"bad"', "123", "false"])
    @pytest.mark.parametrize("section_name", ["data", "models", "output"])
    def test_malformed_top_level_sections_rejected(
        self, tmp_path: Path, section_name: str, section_val: str
    ) -> None:
        path = _write_yaml(tmp_path, f"{section_name}: {section_val}\n")
        with pytest.raises(ConfigError, match="mapping"):
            load_config(path)

    @pytest.mark.parametrize("section_val", ["[]", "[1]", '"bad"', "123", "false"])
    def test_malformed_tfidf_logreg_section_rejected(
        self, tmp_path: Path, section_val: str
    ) -> None:
        path = _write_yaml(tmp_path, f"models:\n  tfidf_logreg: {section_val}\n")
        with pytest.raises(ConfigError, match="mapping"):
            load_config(path)


class TestNumericShapeAndStringValidation:
    @pytest.mark.parametrize("bad_val", ['"abc"', "null", "true", "false", "[0.2]"])
    def test_invalid_validation_size_type_rejected(self, tmp_path: Path, bad_val: str) -> None:
        path = _write_yaml(tmp_path, f"data:\n  validation_size: {bad_val}\n")
        with pytest.raises(ConfigError, match="validation_size"):
            load_config(path)

    @pytest.mark.parametrize("bad_val", ['"abc"', "null", "1.5", "true", "false"])
    def test_invalid_seed_type_rejected(self, tmp_path: Path, bad_val: str) -> None:
        path = _write_yaml(tmp_path, f"data:\n  seed: {bad_val}\n")
        with pytest.raises(ConfigError, match="seed"):
            load_config(path)

    @pytest.mark.parametrize("field_name", ["min_df", "max_iter", "random_state"])
    @pytest.mark.parametrize("bad_val", ['"abc"', "null", "1.5", "true", "false"])
    def test_invalid_tfidf_int_types_rejected(
        self, tmp_path: Path, field_name: str, bad_val: str
    ) -> None:
        path = _write_yaml(tmp_path, f"models:\n  tfidf_logreg:\n    {field_name}: {bad_val}\n")
        with pytest.raises(ConfigError, match=field_name):
            load_config(path)

    @pytest.mark.parametrize("field_name", ["max_df", "C"])
    @pytest.mark.parametrize("bad_val", ['"abc"', "null", "true", "false", "[1.0]"])
    def test_invalid_tfidf_float_types_rejected(
        self, tmp_path: Path, field_name: str, bad_val: str
    ) -> None:
        path = _write_yaml(tmp_path, f"models:\n  tfidf_logreg:\n    {field_name}: {bad_val}\n")
        with pytest.raises(ConfigError, match=field_name):
            load_config(path)

    @pytest.mark.parametrize(
        "bad_ngram",
        ['"12"', "[1]", "[1, 2, 3]", '["a", "b"]', "[1.5, 2]", "[true, 2]", "12", "null"],
    )
    def test_invalid_ngram_shape_and_types_rejected(self, tmp_path: Path, bad_ngram: str) -> None:
        path = _write_yaml(tmp_path, f"models:\n  tfidf_logreg:\n    ngram_range: {bad_ngram}\n")
        with pytest.raises(ConfigError, match="ngram_range"):
            load_config(path)

    @pytest.mark.parametrize(
        ("section", "field_name"),
        [
            ("data", "raw_dir"),
            ("output", "model_dir"),
            ("output", "reports_dir"),
        ],
    )
    @pytest.mark.parametrize("bad_str", ["null", '""', '"   "', "123", "[]"])
    def test_invalid_path_strings_rejected(
        self, tmp_path: Path, section: str, field_name: str, bad_str: str
    ) -> None:
        path = _write_yaml(tmp_path, f"{section}:\n  {field_name}: {bad_str}\n")
        with pytest.raises(ConfigError, match=field_name):
            load_config(path)


class TestNonFiniteAndRangeValidation:
    @pytest.mark.parametrize("non_finite", [".inf", "-.inf", ".nan", '"inf"', '"nan"'])
    def test_non_finite_c_rejected(self, tmp_path: Path, non_finite: str) -> None:
        path = _write_yaml(tmp_path, f"models:\n  tfidf_logreg:\n    C: {non_finite}\n")
        with pytest.raises(ConfigError, match="C"):
            load_config(path)

    @pytest.mark.parametrize("non_finite", [".inf", "-.inf", ".nan"])
    def test_non_finite_validation_size_and_max_df_rejected(
        self, tmp_path: Path, non_finite: str
    ) -> None:
        val_path = _write_yaml(tmp_path, f"data:\n  validation_size: {non_finite}\n", "val.yaml")
        with pytest.raises(ConfigError, match="validation_size"):
            load_config(val_path)

        max_df_path = _write_yaml(
            tmp_path,
            f"models:\n  tfidf_logreg:\n    max_df: {non_finite}\n",
            "maxdf.yaml",
        )
        with pytest.raises(ConfigError, match="max_df"):
            load_config(max_df_path)

    @pytest.mark.parametrize(
        ("yaml_text", "pattern"),
        [
            ("data:\n  validation_size: 0.0\n", "validation_size"),
            ("data:\n  validation_size: 1.0\n", "validation_size"),
            ("data:\n  validation_size: -0.1\n", "validation_size"),
            ("data:\n  validation_size: 1.2\n", "validation_size"),
            ("data:\n  seed: -1\n", "seed"),
            ("models:\n  tfidf_logreg:\n    analyzer: invalid\n", "analyzer"),
            ("models:\n  tfidf_logreg:\n    ngram_range: [0, 2]\n", "ngram_range"),
            ("models:\n  tfidf_logreg:\n    ngram_range: [3, 2]\n", "ngram_range"),
            ("models:\n  tfidf_logreg:\n    min_df: 0\n", "min_df"),
            ("models:\n  tfidf_logreg:\n    max_df: 0.0\n", "max_df"),
            ("models:\n  tfidf_logreg:\n    max_df: -0.5\n", "max_df"),
            ("models:\n  tfidf_logreg:\n    max_df: 1.5\n", "max_df"),
            ("models:\n  tfidf_logreg:\n    C: 0.0\n", "C"),
            ("models:\n  tfidf_logreg:\n    C: -1.0\n", "C"),
            ("models:\n  tfidf_logreg:\n    max_iter: 0\n", "max_iter"),
            ("models:\n  tfidf_logreg:\n    random_state: -1\n", "random_state"),
        ],
    )
    def test_known_invalid_ranges_rejected(
        self, tmp_path: Path, yaml_text: str, pattern: str
    ) -> None:
        path = _write_yaml(tmp_path, yaml_text)
        with pytest.raises(ConfigError, match=pattern):
            load_config(path)
