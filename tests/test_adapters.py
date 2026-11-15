"""Dataset adapter regressions using synthetic local inputs only."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from vfi.config import DataConfig
from vfi.data.adapters import AdapterError, load_dataset_for_config
from vfi.data.loading import DataLoadingError, Record, load_dataset, parse_label_mapping
from vfi.data.splitting import stratified_split
from vfi.data.validation import (
    NEAR_DUPLICATE_MAX_RECORDS,
    find_contradictory_labels,
    near_duplicate_candidates,
)

PRIVATE = "FAKE_PRIVATE_FEEDBACK_917027"
SYNTHETIC_MAPPING = "0:negative,1:neutral,2:positive"


def config(root: Path, adapter: str = "five_level_csv", **kwargs: object) -> DataConfig:
    return replace(
        DataConfig(raw_dir=str(root), dataset_adapter=adapter, synthetic_fixture=True),
        **kwargs,
    )


def write_csv(root: Path, split: str, rows: list[tuple[str, ...]]) -> Path:
    path = root / f"{split}.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)
    return path


def write_jsonl(root: Path, split: str, rows: list[dict]) -> None:
    path = root / f"{split}.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def custom_row(split: str, /, **changes: object) -> dict:
    return {
        "record_id": f"{PRIVATE}_{split}",
        "text": f"{PRIVATE} {split}",
        "sentiment_label": "negative",
        "source_type": "synthetic",
        "provenance_reference": PRIVATE,
        "annotation_method": "human",
        "annotation_status": "approved",
        "dataset_version": "v1",
        "split": split,
        **changes,
    }


def test_five_level_mapping_and_effective_split_provenance(tmp_path: Path) -> None:
    train = write_csv(
        tmp_path,
        "train",
        [("text", "label", "split"), *((f"Train {i}", str(i), "train") for i in range(1, 6))],
    )
    write_csv(tmp_path, "validation", [("text", "label"), (PRIVATE, "3")])
    data = config(tmp_path)
    result = load_dataset_for_config(data)
    assert [record.label for record in result.records_by_split["train"]] == [
        "negative",
        "negative",
        "neutral",
        "positive",
        "positive",
    ]
    assert result.records_by_split["validation"][0].split == "validation"
    assert PRIVATE not in json.dumps(result.provenance)
    previous = result.provenance["split_provenance"]["train"]
    train.write_bytes(train.read_bytes().replace(b"Train 1", b"Train  1"))
    current = load_dataset_for_config(data).provenance["split_provenance"]["train"]
    assert previous["fingerprint"] == current["fingerprint"]
    assert previous["source_content_sha256"] != current["source_content_sha256"]


@pytest.mark.parametrize(
    ("header", "row", "error"),
    [
        ("text,label", f"example,{PRIVATE}", "sentiment level"),
        ("text,label,split", f"example,1,{PRIVATE}", "physical file"),
        ("text,label,split", "example,1", "number of fields"),
        (f"text,label,{PRIVATE}", "example,1,train", "optional split columns"),
        ("text,label,label", "example,1,1", "optional split columns"),
        ("text,label", '"unfinished quote,1', "malformed CSV"),
    ],
)
def test_malformed_csv_has_privacy_safe_errors(
    tmp_path: Path, header: str, row: str, error: str
) -> None:
    (tmp_path / "train.csv").write_text(f"{header}\n{row}\n", encoding="utf-8")
    with pytest.raises(AdapterError, match=error) as caught:
        load_dataset_for_config(config(tmp_path))
    assert PRIVATE not in str(caught.value)


def test_neu_esc_toxic_stays_distinct_or_is_explicitly_excluded(tmp_path: Path) -> None:
    for split in ("train", "validation"):
        write_csv(
            tmp_path,
            split,
            [("text", "label"), (f"Toxic {split}", "Toxic"), (f"Negative {split}", "Negative")],
        )
    data = config(tmp_path, "neu_esc")
    with pytest.raises(AdapterError, match="explicit handling policy"):
        load_dataset_for_config(data)
    with pytest.raises(AdapterError, match="requires justification"):
        load_dataset_for_config(replace(data, toxic_policy="exclude"))
    exclude = load_dataset_for_config(
        replace(data, toxic_policy="exclude", toxic_justification=PRIVATE)
    )
    assert [row.label for row in exclude.records_by_split["train"]] == ["negative"]
    assert exclude.provenance["split_provenance"]["validation"]["toxic_excluded"] == 1
    assert PRIVATE not in json.dumps(exclude.provenance)
    keep = load_dataset_for_config(replace(data, toxic_policy="keep", toxic_justification="audit"))
    assert [row.label for row in keep.records_by_split["train"]] == ["toxic", "negative"]


def test_rights_gate_precedes_path_inspection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_stat(*args: object, **kwargs: object) -> None:
        pytest.fail("Path was inspected before the rights gate")

    with monkeypatch.context() as context:
        context.setattr(Path, "lstat", forbidden_stat)
        with pytest.raises(AdapterError, match="permitted use"):
            load_dataset_for_config(config(tmp_path, synthetic_fixture=False))


@pytest.mark.parametrize("adapter", ["five_level_csv", "custom_jsonl", "uit_vsfc", "neu_esc"])
def test_holdout_only_opened_on_explicit_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, adapter: str
) -> None:
    data = config(tmp_path, adapter)
    for split in ("train", "validation", "test"):
        if adapter == "custom_jsonl":
            write_jsonl(tmp_path, split, [custom_row(split)])
        elif adapter == "uit_vsfc":
            subdir = tmp_path / split
            subdir.mkdir()
            (subdir / "sentences.txt").write_text(f"Feedback {split}\n", encoding="utf-8")
            (subdir / "sentiments.txt").write_text("0\n", encoding="utf-8")
            data = replace(
                data,
                label_mapping=SYNTHETIC_MAPPING,
                label_mapping_evidence="synthetic assay",
                label_mapping_source="synthetic_fixture",
            )
        else:
            label = "1" if adapter == "five_level_csv" else "negative"
            write_csv(tmp_path, split, [("text", "label"), (f"Feedback {split}", label)])

    original_open = Path.open
    original_lstat = Path.lstat
    inspected: list[Path] = []
    opened: list[Path] = []

    def guarded_open(path: Path, *args: object, **kwargs: object):
        if path.is_relative_to(tmp_path):
            opened.append(path)
        return original_open(path, *args, **kwargs)

    def guarded_lstat(path: Path, *args: object, **kwargs: object):
        if path.is_relative_to(tmp_path):
            inspected.append(path)
        return original_lstat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(Path, "lstat", guarded_lstat)
    training = load_dataset_for_config(data)
    assert set(training.records_by_split) == {"train", "validation"}
    assert opened
    for path in [*opened, *inspected]:
        assert all(part.split(".")[0] != "test" for part in path.relative_to(tmp_path).parts)
    opened.clear()
    audited = load_dataset_for_config(data, include_test=True)
    assert audited.records_by_split["test"][0].split == "test"
    assert any("test" in path.parts or path.stem == "test" for path in opened)
    assert audited.provenance["identity"] == training.provenance["identity"]
    for split, recorded in training.provenance["split_provenance"].items():
        assert audited.provenance["split_provenance"][split] == recorded


def test_missing_test_holdout_is_explicit(tmp_path: Path) -> None:
    for split in ("train", "validation"):
        write_csv(tmp_path, split, [("text", "label"), (split, "1")])
    data = config(tmp_path)
    assert "test" not in load_dataset_for_config(data).records_by_split
    with pytest.raises(AdapterError, match="separate test holdout"):
        load_dataset_for_config(data, include_test=True)


def test_linked_split_sources_are_rejected(tmp_path: Path) -> None:
    test = write_csv(tmp_path, "test", [("text", "label"), (PRIVATE, "1")])
    try:
        os.link(test, tmp_path / "train.csv")
    except OSError:
        pytest.skip("Hard links are unavailable on this filesystem")
    with pytest.raises(AdapterError, match="separate regular file"):
        load_dataset_for_config(config(tmp_path))


def test_uit_mapping_gate_and_legacy_parser(tmp_path: Path) -> None:
    data = config(tmp_path, "uit_vsfc")
    with pytest.raises(AdapterError, match="explicit label mapping"):
        load_dataset_for_config(data)
    for split in ("train", "validation"):
        subdir = tmp_path / split
        subdir.mkdir()
        (subdir / "sentences.txt").write_text(f"Feedback {split}\n", encoding="utf-8")
        (subdir / "sentiments.txt").write_text("0\n", encoding="utf-8")
    mapped = load_dataset_for_config(
        replace(
            data,
            label_mapping=SYNTHETIC_MAPPING,
            label_mapping_evidence=PRIVATE,
            label_mapping_source="synthetic_fixture",
        )
    )
    assert mapped.records_by_split["train"][0].label == "negative"
    assert load_dataset(tmp_path, splits=("train",))["train"][0].label == "positive"
    assert PRIVATE not in json.dumps(mapped.provenance)


@pytest.mark.parametrize(
    "spec", ["0:negative,0:neutral,2:positive", "0:negative,1:negative,2:positive", PRIVATE]
)
def test_invalid_uit_mapping_never_reports_raw_values(spec: str) -> None:
    with pytest.raises(DataLoadingError, match="Invalid label mapping") as caught:
        parse_label_mapping(spec)
    assert PRIVATE not in str(caught.value)


def test_custom_approval_and_provenance_digest(tmp_path: Path) -> None:
    train = custom_row("train")
    draft = custom_row(
        "train",
        record_id="draft",
        sentiment_label="",
        annotation_method="",
        annotation_status="draft",
    )
    write_jsonl(tmp_path, "train", [train, draft])
    write_jsonl(tmp_path, "validation", [custom_row("validation")])
    data = config(tmp_path, "custom_jsonl")
    result = load_dataset_for_config(data)
    before = result.provenance["split_provenance"]["train"]
    assert before["source_type_counts"] == {"synthetic": 1}
    assert before["unapproved_records_in_source"] == 1
    assert PRIVATE not in json.dumps(result.provenance)
    write_jsonl(tmp_path, "train", [{**train, "provenance_reference": "new evidence"}])
    after = load_dataset_for_config(data).provenance["split_provenance"]["train"]
    assert before["fingerprint"] == after["fingerprint"]
    assert before["source_content_sha256"] != after["source_content_sha256"]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"annotation_method": "synthetic_generator"}, "approval requires a human annotation"),
        ({"dataset_version": "v2"}, "dataset_version does not match"),
        ({"source_type": "owner_collected"}, "fixture mode"),
        ({"split": "test"}, "match its physical file"),
    ],
)
def test_custom_adapters_refuse_inconsistent_rows(
    tmp_path: Path, changes: dict, message: str
) -> None:
    write_jsonl(tmp_path, "train", [custom_row("train", **changes)])
    with pytest.raises(AdapterError, match=message) as caught:
        load_dataset_for_config(config(tmp_path, "custom_jsonl"))
    assert PRIVATE not in str(caught.value)


def test_custom_reused_id_across_splits_is_rejected(tmp_path: Path) -> None:
    write_jsonl(tmp_path, "train", [custom_row("train", record_id=PRIVATE)])
    write_jsonl(tmp_path, "validation", [custom_row("validation", record_id=PRIVATE)])
    with pytest.raises(AdapterError, match="repeated across loaded splits") as caught:
        load_dataset_for_config(config(tmp_path, "custom_jsonl"))
    assert PRIVATE not in str(caught.value)


def test_grouped_validation_fallback_keeps_duplicates_and_fingerprints(tmp_path: Path) -> None:
    rows = [
        ("text", "label"),
        *((f"Training {label} {i}", label) for label in ("1", "3", "5") for i in range(4)),
    ]
    rows.append(("  Training  1  0  ", "1"))
    write_csv(tmp_path, "train", rows)
    data = config(tmp_path, use_official_splits=False, validation_size=0.25)
    result = load_dataset_for_config(data)
    train, validation = result.records_by_split["train"], result.records_by_split["validation"]
    assert len(train) + len(validation) == len(rows) - 1
    assert {row.text for row in train}.isdisjoint(row.text for row in validation)
    assert result == load_dataset_for_config(data)
    assert result.provenance["split_provenance"]["validation"]["source_split"] == "train"


def test_grouping_preserves_labels_of_conflicting_rows() -> None:
    rows = [
        Record(f"Feedback {label} {i}", label, "train")
        for label in ("negative", "neutral", "positive")
        for i in range(4)
    ]
    rows += [Record("Feedback negative 0", "positive", "train")]
    train, validation = stratified_split(rows, validation_size=0.25)
    assert {row.text for row in train}.isdisjoint(row.text for row in validation)
    assert find_contradictory_labels(("train", train), ("validation", validation)) == {
        "count": 1,
        "affected_records": 2,
    }


def test_near_duplicate_audit_is_bounded_and_advisory() -> None:
    pair = [Record(PRIVATE, "negative", "train"), Record(PRIVATE, "positive", "validation")]
    candidates = near_duplicate_candidates(pair)
    assert len(candidates) == 1
    assert candidates[0]["similarity"] == pytest.approx(1)
    assert PRIVATE not in json.dumps(candidates)
    prefix = [Record("", "neutral", "train")] * NEAR_DUPLICATE_MAX_RECORDS
    assert near_duplicate_candidates(prefix + pair) == []
