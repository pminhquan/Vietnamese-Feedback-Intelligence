"""Shared deterministic synthetic fixtures for VFI tests.

No real dataset is downloaded or used in tests: every fixture is a small set
of synthetic Vietnamese-style sentences defined inline, so the suite stays
fast, offline, and license-safe. Templates deliberately reuse common tokens
("giảng viên", "môn học") so tiny TF-IDF vocabularies stay non-empty.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from vfi.data.loading import RAW_SENTIMENT_TO_LABEL, Record

POSITIVE_TEMPLATES = [
    "Giảng viên giảng dạy rất dễ hiểu và nhiệt tình.",
    "Môn học này rất hữu ích cho sinh viên.",
    "Nội dung môn học thú vị, tài liệu rõ ràng.",
]
NEGATIVE_TEMPLATES = [
    "Giảng viên giảng dạy quá nhanh, sinh viên không theo kịp.",
    "Môn học này không hữu ích, tài liệu cũ.",
    "Học phí tăng nhưng chất lượng phòng học không đổi.",
]
NEUTRAL_TEMPLATES = [
    "Môn học này được học vào thứ Hai hàng tuần.",
    "Giảng viên giảng dạy tại phòng A101 vào buổi sáng.",
    "Lớp học có ba mươi sinh viên đăng ký.",
]


def make_records(
    n_per_class: int = 4,
    split: str = "train",
    offset: int = 0,
) -> list[Record]:
    """Deterministic synthetic records with all three canonical labels."""
    records: list[Record] = []
    for index in range(n_per_class):
        suffix = f" (mã {index + offset})"
        for label, templates in (
            ("positive", POSITIVE_TEMPLATES),
            ("negative", NEGATIVE_TEMPLATES),
            ("neutral", NEUTRAL_TEMPLATES),
        ):
            records.append(Record(templates[index % len(templates)] + suffix, label, split))
    return records


def write_vsfc_files(root: Path, records_by_split: dict[str, list[Record]]) -> None:
    """Write records in the raw UIT-VSFC layout under ``root``."""
    raw_label_by_name = {name: raw for raw, name in RAW_SENTIMENT_TO_LABEL.items()}
    for split, records in records_by_split.items():
        split_dir = root / split
        split_dir.mkdir(parents=True)
        (split_dir / "sentences.txt").write_text(
            "".join(record.text + "\n" for record in records), encoding="utf-8"
        )
        (split_dir / "sentiments.txt").write_text(
            "".join(f"{raw_label_by_name[record.label]}\n" for record in records), encoding="utf-8"
        )


def write_test_config(
    path: Path,
    raw_dir: Path,
    model_dir: Path,
    reports_dir: Path,
    seed: int = 42,
) -> None:
    """Write a config YAML pointing at temporary test directories.

    ``min_df`` is lowered to 1 because synthetic corpora are tiny.
    """
    config = {
        "data": {
            "raw_dir": str(raw_dir),
            "use_official_splits": True,
            "seed": seed,
            # Test fixtures define their OWN label space, so the raw-label
            # gate is satisfied by an explicit, honestly-labeled declaration.
            "label_mapping_source": "synthetic_fixture",
            "label_mapping": "0:positive,1:negative,2:neutral",
            "label_mapping_evidence": "synthetic test fixture (labels defined by the test)",
        },
        "models": {
            "tfidf_logreg": {
                "analyzer": "word",
                "ngram_range": [1, 2],
                "min_df": 1,
                "max_df": 1.0,
                "C": 4.0,
                "random_state": seed,
            }
        },
        "output": {
            "model_dir": str(model_dir),
            "reports_dir": str(reports_dir),
        },
    }
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
