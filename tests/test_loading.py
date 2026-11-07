"""Tests for raw data loading and Vietnamese-safe normalization."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.conftest import make_records, write_vsfc_files

from vfi.data.loading import (
    CANONICAL_LABELS,
    DataLoadingError,
    load_dataset,
    load_split,
    normalize_text,
)


class TestNormalizeText:
    def test_nfc_composes_decomposed_diacritics(self) -> None:
        decomposed = "ho\u0300a" + " " + "tra\u0301i"  # o + combining grave, a + combining acute
        precomposed = "hòa trái"
        assert normalize_text(decomposed) == normalize_text(precomposed)
        assert normalize_text(decomposed) == "hòa trái"

    def test_preserves_diacritics_and_case(self) -> None:
        assert normalize_text("Giảng Viên Dạy RẤT TỐT") == "Giảng Viên Dạy RẤT TỐT"

    def test_collapses_whitespace(self) -> None:
        assert normalize_text("  môn \n học\tquan trọng  ") == "môn học quan trọng"

    def test_empty_stays_empty(self) -> None:
        assert normalize_text("   \n\t ") == ""


class TestLoadSplit:
    def test_roundtrip_and_label_mapping(self, tmp_path: Path) -> None:
        write_vsfc_files(tmp_path, {"train": make_records(2, split="train")})
        records = load_split(tmp_path, "train")
        assert len(records) == 6
        labels = {record.label for record in records}
        assert labels == set(CANONICAL_LABELS)

    def test_raw_integer_mapping_is_documented(self, tmp_path: Path) -> None:
        # Raw 0/1/2 must map to positive/negative/neutral (SEACrowd ClassLabel order).
        split_dir = tmp_path / "train"
        split_dir.mkdir()
        (split_dir / "sentences.txt").write_text("a\nb\nc\n", encoding="utf-8")
        (split_dir / "sentiments.txt").write_text("0\n1\n2\n", encoding="utf-8")
        records = load_split(tmp_path, "train")
        assert [record.label for record in records] == ["positive", "negative", "neutral"]

    def test_misaligned_files_raise(self, tmp_path: Path) -> None:
        split_dir = tmp_path / "train"
        split_dir.mkdir()
        (split_dir / "sentences.txt").write_text("a\nb\n", encoding="utf-8")
        (split_dir / "sentiments.txt").write_text("0\n", encoding="utf-8")
        with pytest.raises(DataLoadingError, match="misaligned"):
            load_split(tmp_path, "train")

    def test_empty_sentence_raises_with_line_number(self, tmp_path: Path) -> None:
        split_dir = tmp_path / "train"
        split_dir.mkdir()
        (split_dir / "sentences.txt").write_text("a\n\n", encoding="utf-8")
        (split_dir / "sentiments.txt").write_text("0\n1\n", encoding="utf-8")
        with pytest.raises(DataLoadingError, match="line 2"):
            load_split(tmp_path, "train")

    @pytest.mark.parametrize("label_line", ["9", "abc", ""])
    def test_invalid_labels_raise(self, tmp_path: Path, label_line: str) -> None:
        split_dir = tmp_path / "train"
        split_dir.mkdir()
        (split_dir / "sentences.txt").write_text("a\n", encoding="utf-8")
        (split_dir / "sentiments.txt").write_text(f"{label_line}\n", encoding="utf-8")
        with pytest.raises(DataLoadingError, match="sentiment label"):
            load_split(tmp_path, "train")

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DataLoadingError, match="missing"):
            load_split(tmp_path, "train")

    def test_unknown_split_name_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DataLoadingError, match="Unknown split"):
            load_split(tmp_path, "dev")

    def test_utf8_sig_bom_tolerated(self, tmp_path: Path) -> None:
        split_dir = tmp_path / "train"
        split_dir.mkdir()
        (split_dir / "sentences.txt").write_bytes("môn học\n".encode("utf-8-sig"))
        (split_dir / "sentiments.txt").write_text("2\n", encoding="utf-8")
        records = load_split(tmp_path, "train")
        assert records[0].text == "môn học"


class TestLoadDataset:
    def test_all_splits_loaded(self, tmp_path: Path) -> None:
        write_vsfc_files(
            tmp_path,
            {
                "train": make_records(2, split="train"),
                "validation": make_records(1, split="validation"),
                "test": make_records(1, split="test", offset=100),
            },
        )
        data = load_dataset(tmp_path)
        assert set(data) == {"train", "validation", "test"}
        assert len(data["train"]) == 6
        assert len(data["validation"]) == 3
        assert len(data["test"]) == 3
