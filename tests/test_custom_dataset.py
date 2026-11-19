"""Regression tests for local dataset management using synthetic fixtures."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest
import yaml

from vfi import custom_dataset
from vfi.cli import main
from vfi.custom_dataset import (
    CustomDatasetError,
    CustomDatasetStore,
    add_dataset_parser,
    generate_synthetic_draft,
    make_record,
    read_records,
    record_fingerprint,
)
from vfi.data.loading import CANONICAL_LABELS

VERSION = "fixture-v1"
MARKER = "SYNTHETIC_PRIVATE_MARKER"


def test_draft_annotation_and_separate_approval(tmp_path: Path) -> None:
    store = CustomDatasetStore(tmp_path, VERSION)
    store.initialize()
    draft = generate_synthetic_draft(1, dataset_version=VERSION, run_id="TEST-FIXTURE")[0]
    store.append_records([draft])
    with pytest.raises(CustomDatasetError, match="human annotation"):
        store.approve(draft["record_id"])
    store.annotate(draft["record_id"], "negative")
    store.approve(draft["record_id"])
    assert store.load_records()[0]["annotation_status"] == "approved"
    store.annotate(draft["record_id"], "neutral")
    corrected = store.load_records()[0]
    assert corrected["annotation_status"] == "annotated"
    assert corrected["source_type"] == "synthetic"
    assert corrected["provenance_reference"] == draft["provenance_reference"]


def test_append_validates_entire_batch_without_leaking_id(tmp_path: Path) -> None:
    store = CustomDatasetStore(tmp_path, VERSION)
    row = generate_synthetic_draft(1, dataset_version=VERSION, run_id="TEST-FIXTURE")[0]
    store.append_records([row])
    before = store.records_path.read_bytes()
    with pytest.raises(CustomDatasetError, match="duplicate record_id") as caught:
        store.append_records([{**row, "record_id": "new"}, row])
    assert row["record_id"] not in str(caught.value)
    assert store.records_path.read_bytes() == before


def test_overlapping_appends_preserve_both_batches(tmp_path: Path, monkeypatch) -> None:
    store = CustomDatasetStore(tmp_path, VERSION)
    store.initialize()
    first = generate_synthetic_draft(1, dataset_version=VERSION, run_id="first")
    second = generate_synthetic_draft(1, dataset_version=VERSION, run_id="second")
    first_writing = Event()
    second_started = Event()
    second_writing = Event()
    write = custom_dataset._atomic_write

    def overlap(path: Path, text: str) -> None:
        if first[0]["record_id"] in text and second[0]["record_id"] not in text:
            first_writing.set()
            assert second_started.wait(5)
            # An unlocked second writer can publish while the first holds stale rows.
            second_writing.wait(1)
            write(path, text)
        else:
            write(path, text)
            second_writing.set()

    def append_second() -> int:
        second_started.set()
        return CustomDatasetStore(tmp_path, VERSION).append_records(second)

    monkeypatch.setattr(custom_dataset, "_atomic_write", overlap)
    with ThreadPoolExecutor(max_workers=2) as writers:
        first_result = writers.submit(store.append_records, first)
        assert first_writing.wait(5)
        second_result = writers.submit(append_second)
        assert first_result.result(timeout=15) == second_result.result(timeout=15) == 3
    assert store.load_records() == [*first, *second]


def test_validate_jsonl_version_and_machine_approval(tmp_path: Path) -> None:
    row = generate_synthetic_draft(1, dataset_version=VERSION, run_id="TEST-FIXTURE")[0]
    path = tmp_path / "fixture.jsonl"
    path.write_text(json.dumps({**row, "annotation_status": "approved"}) + "\n", encoding="utf-8")
    with pytest.raises(CustomDatasetError, match="approval requires a human"):
        read_records(path)
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(CustomDatasetError, match="dataset_version"):
        read_records(path, expected_version="fixture-v2")
    assert read_records(path) == [row]


def test_duplicate_json_keys_cannot_override_provenance(tmp_path: Path) -> None:
    row = generate_synthetic_draft(1, dataset_version=VERSION, run_id="fixture")[0]
    path = tmp_path / "records.jsonl"
    contents = json.dumps(row)[:-1] + ', "source_type": "owner_collected"}\n'
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(CustomDatasetError, match="Line 1: duplicate JSON object key") as caught:
        read_records(path)
    assert "source_type" not in str(caught.value)


@pytest.mark.parametrize(
    ("invalid", "error"),
    [(json.dumps(MARKER), "expected a JSON object"), ("", "unknown field")],
)
def test_error_is_safe_and_keeps_physical_line_number(
    tmp_path: Path, invalid: str, error: str
) -> None:
    if not invalid:
        row = generate_synthetic_draft(1, dataset_version=VERSION, run_id="TEST-FIXTURE")[0]
        invalid = json.dumps({**row, MARKER: MARKER, "record_id": MARKER})
    path = tmp_path / "fixture.jsonl"
    path.write_text("\n" + invalid + "\n", encoding="utf-8")
    with pytest.raises(CustomDatasetError, match=f"Line 2: {error}") as caught:
        read_records(path)
    assert MARKER not in str(caught.value)


def test_manifest_and_summary_omit_private_fields(tmp_path: Path) -> None:
    store = CustomDatasetStore(tmp_path, VERSION)
    row = make_record(
        record_id=MARKER,
        text=f"  {MARKER}   nội dung  ",
        source_type="synthetic",
        provenance_reference=MARKER,
        dataset_version=VERSION,
    )
    store.append_records(
        [
            row,
            {
                **row,
                "record_id": "other",
                "sentiment_label": "negative",
                "annotation_method": "synthetic_generator",
            },
        ]
    )
    assert store.load_records()[0]["text"] == f"{MARKER} nội dung"
    manifest = store.write_manifest()
    saved = manifest.read_text(encoding="utf-8")
    assert MARKER not in saved
    summary = json.loads(saved)
    assert summary["duplicate_text_count"] == 1
    assert summary["contradictory_text_count"] == 0
    with pytest.raises(CustomDatasetError, match="dataset_version"):
        CustomDatasetStore(tmp_path, "other-version").initialize()


def _fixture_store(root: Path) -> CustomDatasetStore:
    store = CustomDatasetStore(root, VERSION)
    generated = generate_synthetic_draft(10, dataset_version=VERSION, run_id="TEST-FIXTURE", seed=4)
    # Simulate human approval only for integration testing. Origin remains synthetic.
    store.append_records(
        [
            {**row, "annotation_status": "approved", "annotation_method": "human"}
            for row in [*generated, {**generated[0], "record_id": "same-text"}]
        ]
    )
    return store


def test_grouped_holdout_keeps_duplicates_and_fixture_source(tmp_path: Path) -> None:
    store = _fixture_store(tmp_path / "store")
    out = tmp_path / "split"
    with pytest.raises(CustomDatasetError, match="No approved records"):
        store.export_splits(out)
    assert not out.exists()
    manifest = store.export_splits(out, include_synthetic=True)
    assert manifest["data_kind"] == "synthetic_test_fixture"
    assert manifest == store.export_splits(tmp_path / "repeat", include_synthetic=True)
    assigned = {}
    total = 0
    for split in ("train", "validation", "test"):
        rows = read_records(out / f"{split}.jsonl", expected_version=VERSION)
        assert {row["sentiment_label"] for row in rows} == set(CANONICAL_LABELS)
        assert manifest["split_fingerprints"][split] == record_fingerprint(rows)
        for row in rows:
            assert row["split"] == split
            assert row["source_type"] == "synthetic"
            assert assigned.setdefault(row["text"], split) == split
            total += 1
    assert total == 31


def test_conflicting_approved_labels_fail_validation_and_export(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = _fixture_store(tmp_path / "store")
    row = store.load_records()[0]
    other_label = next(label for label in CANONICAL_LABELS if label != row["sentiment_label"])
    store.append_records([{**row, "record_id": "contradiction", "sentiment_label": other_label}])
    assert (
        main(["dataset", "validate", "--root", str(store.root), "--dataset-version", VERSION]) == 1
    )
    output = capsys.readouterr()
    assert row["text"] not in output.out + output.err
    verdict = json.loads(output.out)
    assert verdict["schema_valid"]
    assert verdict["benchmark_ready"] is False
    assert verdict["contradictory_approved_text_count"] == 1
    with pytest.raises(CustomDatasetError, match="contradictory"):
        store.export_splits(tmp_path / "split", include_synthetic=True)
    assert not (tmp_path / "split").exists()


def test_export_keeps_real_and_fixture_origins_separate(tmp_path: Path) -> None:
    store = _fixture_store(tmp_path / "store")
    rows = store.load_records()
    # These remain a test fixture. Simulate real records for the eligibility rule.
    store.append_records(
        [
            {
                **row,
                "record_id": "owner-" + row["record_id"],
                "source_type": "owner_collected",
                "provenance_reference": "TEST FIXTURE simulating authorized collection",
            }
            for row in rows
        ]
    )
    for include_synthetic, source in [(False, "owner_collected"), (True, "synthetic")]:
        out = tmp_path / source
        manifest = store.export_splits(out, include_synthetic=include_synthetic)
        assert manifest["approved_other_source_count"] == len(rows)
        assert set(
            row["source_type"]
            for split in ("train", "validation", "test")
            for row in read_records(out / f"{split}.jsonl")
        ) == {source}


def test_cli_entry_and_review_require_local_interactive_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    parser = argparse.ArgumentParser()
    add_dataset_parser(parser.add_subparsers(required=True))
    common = ["--root", str(tmp_path), "--dataset-version", VERSION]

    def run(command: str, *args: str) -> int:
        parsed = parser.parse_args(["dataset", command, *common, *args])
        return parsed.func(parsed)

    assert run("init") == 0
    assert run("add") == 1
    assert "interactive terminal" in capsys.readouterr().err
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    entries = iter((MARKER, "TEST FIXTURE origin", "neutral", "approve", "positive"))
    monkeypatch.setattr("builtins.input", lambda _: next(entries))
    assert run("add") == 0
    assert MARKER not in capsys.readouterr().out
    assert run("annotate") == 0
    assert run("approve") == 0
    stored = read_records(tmp_path / "records.jsonl")
    assert stored[0]["annotation_status"] == "approved"
    assert run("annotate", "--record-id", stored[0]["record_id"]) == 0
    assert read_records(tmp_path / "records.jsonl")[0]["annotation_status"] == "annotated"
    capsys.readouterr()
    assert run("report") == 0
    assert MARKER not in capsys.readouterr().out


@pytest.mark.parametrize(("command", "response"), [("approve", "approve"), ("annotate", "neutral")])
def test_interactive_review_rejects_a_changed_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
    response: str,
) -> None:
    store = CustomDatasetStore(tmp_path, VERSION)
    store.initialize()
    row = generate_synthetic_draft(1, dataset_version=VERSION, run_id="fixture")[0]
    record_id = row["record_id"]
    store.append_records([row])
    store.annotate(record_id, "negative")

    def change_before_confirmation(_prompt: str) -> str:
        CustomDatasetStore(tmp_path, VERSION).annotate(record_id, "positive")
        return response

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", change_before_confirmation)
    args = ["dataset", command, "--root", str(tmp_path), "--dataset-version", VERSION]
    assert main([*args, "--record-id", record_id]) == 1
    captured = capsys.readouterr()
    assert "Current label: negative" in captured.out
    assert "Record changed during review" in captured.err
    current = store.load_records()[0]
    assert (current["sentiment_label"], current["annotation_status"]) == ("positive", "annotated")


def test_custom_synthetic_fixture_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store_path = tmp_path / "store"
    split_path = tmp_path / "splits"
    common = ["--root", str(store_path), "--dataset-version", VERSION]
    assert main(["dataset", "init", *common]) == 0
    assert (
        main(
            [
                "dataset",
                "add-synthetic",
                *common,
                "--run-id",
                "TEST-FIXTURE",
                "--count-per-class",
                "10",
            ]
        )
        == 0
    )
    store = CustomDatasetStore(store_path, VERSION)
    # Simulate human label/approval actions for a test fixture. These texts
    # were generated in the test and their source remains explicitly synthetic.
    for row in store.load_records():
        store.annotate(row["record_id"], row["sentiment_label"])
        store.approve(row["record_id"])
    assert main(["dataset", "validate", *common]) == 0
    assert main(["dataset", "split", *common, "--out", str(split_path), "--include-synthetic"]) == 0
    reports_dir = tmp_path / "reports"
    model_dir = tmp_path / "models"
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "raw_dir": str(split_path),
                    "dataset_adapter": "custom_jsonl",
                    "dataset_id": "synthetic_test_fixture",
                    "dataset_version": VERSION,
                    "synthetic_fixture": True,
                },
                "models": {"tfidf_logreg": {"min_df": 1, "max_df": 1.0}},
                "output": {"model_dir": str(model_dir), "reports_dir": str(reports_dir)},
            }
        ),
        encoding="utf-8",
    )
    flags = ["--config", str(config)]
    assert main(["prepare-data", *flags]) == 0
    assert main(["evaluate", "--preflight-only", *flags]) == 0
    original_open = Path.open
    seen = set()

    def guard(path: Path, *args, **kwargs):
        assert path.name not in {"test.jsonl", "evaluation_preflight.json", "dataset_manifest.json"}
        if path.parent == split_path:
            seen.add(path.name)
        return original_open(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", guard)
        assert main(["train", *flags]) == 0
    assert seen == {"train.jsonl", "validation.jsonl"}
    validation = json.loads((reports_dir / "training_report.json").read_text(encoding="utf-8"))
    assert "macro_f1" in validation["validation_report"]["tfidf_logreg_selected"]
    assert main(["evaluate", "--model-id", "tfidf-logreg", *flags]) == 0
    evaluation = json.loads((reports_dir / "eval_tfidf-logreg.json").read_text(encoding="utf-8"))
    assert evaluation["result_scope"] == "synthetic_fixture"
    assert evaluation["test_metrics"]["n_samples"] > 0
    assert set(evaluation["training_provenance"]["dataset_fingerprints"]) == {"train", "validation"}
    capsys.readouterr()
    assert (
        main(
            [
                "predict",
                "--model-dir",
                str(model_dir),
                "--text",
                "Bài tập khó hiểu và thiếu hướng dẫn.",
            ]
        )
        == 0
    )
    prediction = json.loads(capsys.readouterr().out)
    assert set(prediction["probabilities"]) == set(CANONICAL_LABELS)
    holdout = split_path / "test.jsonl"
    holdout_rows = read_records(holdout)
    holdout_rows[0]["provenance_reference"] = MARKER
    holdout.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in holdout_rows),
        encoding="utf-8",
    )
    saved_benchmark = (reports_dir / "eval_tfidf-logreg.json").read_bytes()
    assert main(["evaluate", "--model-id", "tfidf-logreg", *flags]) == 1
    mismatch = capsys.readouterr()
    assert "Adapter provenance differs" in mismatch.err
    assert MARKER not in mismatch.out + mismatch.err
    assert (reports_dir / "eval_tfidf-logreg.json").read_bytes() == saved_benchmark
