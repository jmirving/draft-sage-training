#!/usr/bin/env python3
"""Run training jobs from planned rows in experiment CSV logs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT_DIR.parent / ".tmp" / "prodata-2025-plus-2026-01-clean"
DEFAULT_DATASET_LABEL = "Clean 2025 + Jan 2026"
DEFAULT_ELIGIBILITY_PATH = ROOT_DIR / "data" / "eligibility" / "champion-eligibility-20260213-refresh.json"
DEFAULT_CHAMPION_PRIORS_DIR = ROOT_DIR / "data" / "weights" / "champion-priors"
DEFAULT_ROLE_PRIORS_DIR = ROOT_DIR / "data" / "weights" / "role-priors"
TRAIN_SCRIPT = ROOT_DIR / "scripts" / "train.py"
PUBLISH_SCRIPT = ROOT_DIR / "scripts" / "publish_training_data.py"

REQUIRED_COLUMNS = [
    "test_batch",
    "test_group",
    "run_id",
    "status",
    "dataset_label",
    "variant_name",
    "league_embeddings",
    "team_embeddings",
    "eligibility",
    "draft_frequency_bias",
    "draft_frequency_bias_strength",
    "role_distribution_bias",
    "role_distribution_bias_strength",
    "accuracy",
    "loss",
    "notes",
]


@dataclass(frozen=True)
class PlannedRow:
    row_number: int
    values: dict[str, str]

    @property
    def selector_id(self) -> str:
        value = self.values.get("run_id", "").strip()
        return value or f"row-{self.row_number}"


@dataclass
class RunSpec:
    row: PlannedRow
    output_dir: Path
    log_path: Path
    command: list[str]


@dataclass
class RunResult:
    spec: RunSpec
    return_code: int
    produced_run_id: str | None
    accuracy: float | None
    loss: float | None
    summary_path: Path | None
    index_path: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute training runs defined in EXPERIMENT_TEST_LOG CSV rows.",
    )
    parser.add_argument(
        "--csv",
        help="Path to experiment CSV. Defaults to today's project-brain daily log.",
    )
    parser.add_argument(
        "--status",
        action="append",
        default=None,
        help="Row status to include (repeatable, default: planned).",
    )
    parser.add_argument(
        "--run-id",
        action="append",
        help="Only include specific CSV run_id values (repeatable).",
    )
    parser.add_argument(
        "--test-group",
        action="append",
        help="Only include test_group values (repeatable).",
    )
    parser.add_argument(
        "--test-batch",
        action="append",
        help="Only include test_batch values (repeatable).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit selected rows after filtering.",
    )
    parser.add_argument(
        "--output-root",
        help="Root directory for run outputs/logs (default: timestamped .tmp path).",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help="Max concurrent training processes.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run selected jobs. Without this flag, only prints the plan.",
    )
    parser.add_argument(
        "--backfill-csv",
        action="store_true",
        help="After execution, update CSV rows with actual run_id/status/accuracy/loss.",
    )
    parser.add_argument(
        "--publish-after-run",
        action="store_true",
        help="Run publish_training_data.py after successful jobs.",
    )
    parser.add_argument(
        "--publish-data-dir",
        help="Target hosted data dir for publish step.",
    )
    parser.add_argument(
        "--hosted-index",
        help="Hosted experiment-index to include in additive publish (default: <publish-data-dir>/experiment-index.json).",
    )
    parser.add_argument(
        "--publish-commit",
        action="store_true",
        help="Pass --commit to publish helper.",
    )
    parser.add_argument(
        "--publish-push",
        action="store_true",
        help="Pass --push to publish helper.",
    )
    parser.add_argument(
        "--publish-message",
        help="Commit message for publish helper.",
    )
    parser.add_argument(
        "--inspection-keep",
        type=int,
        default=10,
        help="inspection-keep value for publish helper.",
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Training input directory.",
    )
    parser.add_argument(
        "--allow-dataset-override",
        action="store_true",
        help="Allow non-default input-dir / dataset-label values.",
    )
    parser.add_argument(
        "--champion-mapping-path",
        help="Path to champion mapping JSON (auto-detected by default).",
    )
    parser.add_argument(
        "--champion-eligibility-path",
        default=str(DEFAULT_ELIGIBILITY_PATH),
        help="Eligibility JSON path when CSV row sets eligibility=On.",
    )
    parser.add_argument(
        "--champion-priors-dir",
        default=str(DEFAULT_CHAMPION_PRIORS_DIR),
        help="Champion priors dir when draft_frequency_bias=On.",
    )
    parser.add_argument(
        "--role-priors-dir",
        default=str(DEFAULT_ROLE_PRIORS_DIR),
        help="Role priors dir when role_distribution_bias=On.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--split-strategy", default="seriesid", choices=["random", "gameid", "seriesid"])
    parser.add_argument(
        "--python-bin",
        default=str(ROOT_DIR / ".venv" / "bin" / "python"),
        help="Python executable used to run scripts/train.py.",
    )
    return parser.parse_args()


def slug(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
    normalized = normalized.strip("_")
    return normalized or "run"


def parse_bool(value: str, *, row: PlannedRow, column: str) -> bool:
    parsed = value.strip().lower()
    if parsed in {"on", "true", "1", "yes"}:
        return True
    if parsed in {"off", "false", "0", "no", ""}:
        return False
    raise ValueError(
        f"Row {row.row_number} has invalid boolean in column '{column}': {value!r}"
    )


def parse_required_float(value: str, *, row: PlannedRow, column: str) -> float:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(
            f"Row {row.row_number} requires numeric '{column}' when the matching bias is On."
        )
    try:
        return float(cleaned)
    except ValueError as exc:
        raise ValueError(
            f"Row {row.row_number} has invalid float in column '{column}': {value!r}"
        ) from exc


def ensure_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def ensure_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} not found: {path}")


def default_csv_path() -> Path:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return ROOT_DIR.parent / "project-brain" / "daily-briefs" / f"EXPERIMENT_TEST_LOG_{today}.csv"


def discover_champion_mapping() -> Path:
    candidates = [
        ROOT_DIR / "data" / "ddragon" / "artifacts" / "champion-mapping" / "latest.json",
        ROOT_DIR.parent / ".tmp" / "champion-mapping" / "latest.json",
        ROOT_DIR.parent
        / "lol-ddragon-context-artifact-builder"
        / "data"
        / "ddragon"
        / "artifacts"
        / "champion-mapping"
        / "latest.json",
        ROOT_DIR.parent
        / "lol-ddragon-snapshot-cron"
        / "data"
        / "ddragon"
        / "artifacts"
        / "champion-mapping"
        / "latest.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "No champion mapping found. Set --champion-mapping-path explicitly."
    )


def normalize_filters(values: Iterable[str] | None) -> set[str]:
    return {value.strip() for value in (values or []) if value and value.strip()}


def load_rows(csv_path: Path) -> list[PlannedRow]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        missing = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")
        rows: list[PlannedRow] = []
        for row_number, row in enumerate(reader, start=2):
            normalized = {key: (value or "").strip() for key, value in row.items()}
            rows.append(PlannedRow(row_number=row_number, values=normalized))
        return rows


def select_rows(rows: list[PlannedRow], args: argparse.Namespace) -> list[PlannedRow]:
    statuses = normalize_filters(args.status) or {"planned"}
    selected_run_ids = normalize_filters(args.run_id)
    selected_groups = normalize_filters(args.test_group)
    selected_batches = normalize_filters(args.test_batch)

    chosen: list[PlannedRow] = []
    for row in rows:
        status = row.values.get("status", "")
        run_id = row.values.get("run_id", "")
        group = row.values.get("test_group", "")
        batch = row.values.get("test_batch", "")

        if statuses and status not in statuses:
            continue
        if selected_run_ids and run_id not in selected_run_ids:
            continue
        if selected_groups and group not in selected_groups:
            continue
        if selected_batches and batch not in selected_batches:
            continue
        chosen.append(row)

    if args.limit is not None:
        chosen = chosen[: args.limit]
    return chosen


def derive_output_root(args: argparse.Namespace, csv_path: Path) -> Path:
    if args.output_root:
        return Path(args.output_root).resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return (ROOT_DIR.parent / ".tmp" / f"training-from-csv-{csv_path.stem}-{timestamp}").resolve()


def ensure_unique_output_dir(base: Path, used: set[Path]) -> Path:
    if base not in used and not base.exists():
        used.add(base)
        return base
    suffix = 2
    while True:
        candidate = base.with_name(f"{base.name}-{suffix}")
        if candidate not in used and not candidate.exists():
            used.add(candidate)
            return candidate
        suffix += 1


def build_run_spec(
    row: PlannedRow,
    *,
    args: argparse.Namespace,
    output_root: Path,
    used_output_dirs: set[Path],
    champion_mapping_path: Path,
    input_dir: Path,
) -> RunSpec:
    values = row.values
    display_name = values.get("variant_name") or row.selector_id
    row_key = values.get("run_id") or display_name
    run_slug = slug(row_key)
    output_dir = ensure_unique_output_dir(output_root / run_slug, used_output_dirs)
    log_path = output_root / "logs" / f"{run_slug}.log"

    command: list[str] = [
        args.python_bin,
        "-u",
        str(TRAIN_SCRIPT),
        "--output-dir",
        str(output_dir),
        "--display-name",
        display_name,
        "--description",
        f"CSV planned run from {row.selector_id} ({output_root.name})",
        "--input-dir",
        str(input_dir),
        "--champion-mapping-path",
        str(champion_mapping_path),
        "--epochs",
        str(args.epochs),
        "--seed",
        str(args.seed),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--split-strategy",
        args.split_strategy,
        "--category",
        values.get("test_group") or "csv-driven",
        "--dataset-label",
        values.get("dataset_label") or DEFAULT_DATASET_LABEL,
    ]

    use_league = parse_bool(values.get("league_embeddings", ""), row=row, column="league_embeddings")
    use_team = parse_bool(values.get("team_embeddings", ""), row=row, column="team_embeddings")
    use_eligibility = parse_bool(values.get("eligibility", ""), row=row, column="eligibility")
    use_dfbias = parse_bool(
        values.get("draft_frequency_bias", ""), row=row, column="draft_frequency_bias"
    )
    use_rolebias = parse_bool(
        values.get("role_distribution_bias", ""), row=row, column="role_distribution_bias"
    )

    if not use_league:
        command.append("--no-league-embeddings")
    if not use_team:
        command.append("--no-team-embeddings")

    if use_eligibility:
        eligibility_path = Path(args.champion_eligibility_path).resolve()
        ensure_file(eligibility_path, "Champion eligibility file")
        command.extend(["--champion-eligibility-path", str(eligibility_path)])

    if use_dfbias:
        df_strength = parse_required_float(
            values.get("draft_frequency_bias_strength", ""),
            row=row,
            column="draft_frequency_bias_strength",
        )
        df_dir = Path(args.champion_priors_dir).resolve()
        ensure_dir(df_dir, "Champion priors directory")
        command.extend(["--champion-priors-dir", str(df_dir)])
        command.extend(["--champion-priors-strength", str(df_strength)])

    if use_rolebias:
        role_strength = parse_required_float(
            values.get("role_distribution_bias_strength", ""),
            row=row,
            column="role_distribution_bias_strength",
        )
        role_dir = Path(args.role_priors_dir).resolve()
        ensure_dir(role_dir, "Role priors directory")
        command.extend(["--role-priors-dir", str(role_dir)])
        command.extend(["--role-priors-strength", str(role_strength)])

    return RunSpec(row=row, output_dir=output_dir, log_path=log_path, command=command)


def read_run_result(output_dir: Path) -> tuple[str | None, float | None, float | None, Path | None, Path | None]:
    index_path = output_dir / "experiment-index.json"
    if not index_path.is_file():
        return None, None, None, None, None
    with index_path.open(encoding="utf-8") as handle:
        index_payload = json.load(handle)
    runs = index_payload.get("runs", [])
    if not isinstance(runs, list) or not runs:
        return None, None, None, None, index_path

    latest = sorted(runs, key=lambda row: row.get("run_id") or "")[-1]
    run_id = latest.get("run_id")
    summary_rel = latest.get("summary_path")
    if not run_id:
        return None, None, None, None, index_path
    summary_path = None
    if isinstance(summary_rel, str) and summary_rel:
        summary_path = (output_dir / summary_rel).resolve()
    if summary_path is None or not summary_path.is_file():
        fallback = output_dir / str(run_id) / "summary.json"
        if fallback.is_file():
            summary_path = fallback

    if summary_path is None or not summary_path.is_file():
        return str(run_id), None, None, None, index_path

    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    metrics = summary.get("metrics") or {}
    accuracy = metrics.get("accuracy")
    loss = metrics.get("loss")
    return str(run_id), accuracy, loss, summary_path, index_path


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def print_plan(specs: list[RunSpec], output_root: Path) -> None:
    print(f"Output root: {output_root}")
    print(f"Selected rows: {len(specs)}")
    for spec in specs:
        row = spec.row
        print(
            f"- row={row.row_number} run_id={row.selector_id} group={row.values.get('test_group')} "
            f"variant={row.values.get('variant_name')}"
        )
        print(f"  output_dir={spec.output_dir}")
        print(f"  log_file={spec.log_path}")
        print(f"  cmd={' '.join(spec.command)}")


def execute_specs(specs: list[RunSpec], max_parallel: int) -> list[RunResult]:
    if max_parallel < 1:
        raise ValueError("--max-parallel must be >= 1")

    output_root = specs[0].output_dir.parent if specs else None
    if output_root is not None:
        (output_root / "logs").mkdir(parents=True, exist_ok=True)
    results: list[RunResult] = []
    active: list[tuple[RunSpec, subprocess.Popen[bytes], object]] = []

    def wait_oldest() -> None:
        spec, process, log_handle = active.pop(0)
        return_code = process.wait()
        log_handle.close()
        run_id, accuracy, loss, summary_path, index_path = read_run_result(spec.output_dir)
        results.append(
            RunResult(
                spec=spec,
                return_code=return_code,
                produced_run_id=run_id,
                accuracy=accuracy if isinstance(accuracy, (int, float)) else None,
                loss=loss if isinstance(loss, (int, float)) else None,
                summary_path=summary_path,
                index_path=index_path,
            )
        )
        status = "DONE" if return_code == 0 else "FAILED"
        print(f"{timestamp()} {status} row={spec.row.row_number} selector={spec.row.selector_id} code={return_code}")

    for spec in specs:
        spec.output_dir.mkdir(parents=True, exist_ok=True)
        spec.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = spec.log_path.open("w", encoding="utf-8")
        print(f"{timestamp()} START row={spec.row.row_number} selector={spec.row.selector_id}")
        process = subprocess.Popen(spec.command, stdout=log_handle, stderr=subprocess.STDOUT)
        active.append((spec, process, log_handle))
        while len(active) >= max_parallel:
            wait_oldest()

    while active:
        wait_oldest()
    return results


def backfill_csv(csv_path: Path, results: list[RunResult]) -> None:
    by_row = {result.spec.row.row_number: result for result in results}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        rows = list(reader)

    for index, row in enumerate(rows, start=2):
        result = by_row.get(index)
        if result is None:
            continue
        if result.return_code == 0:
            row["status"] = "completed"
            if result.produced_run_id:
                row["run_id"] = result.produced_run_id
            if result.accuracy is not None:
                row["accuracy"] = f"{result.accuracy:.10f}"
            if result.loss is not None:
                row["loss"] = f"{result.loss:.10f}"
            note = row.get("notes", "").strip()
            suffix = "Auto-backfilled by run_training_from_csv.py"
            row["notes"] = suffix if not note else f"{note} | {suffix}"
        else:
            row["status"] = "failed"
            note = row.get("notes", "").strip()
            suffix = f"Run failed (exit={result.return_code})"
            row["notes"] = suffix if not note else f"{note} | {suffix}"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_publish(args: argparse.Namespace, results: list[RunResult]) -> None:
    success_indexes = [result.index_path for result in results if result.return_code == 0 and result.index_path]
    unique_indexes: list[Path] = []
    seen: set[Path] = set()
    for index_path in success_indexes:
        resolved = index_path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_indexes.append(resolved)

    if not unique_indexes:
        print("No successful run indexes found; skipping publish.")
        return

    if not args.publish_data_dir:
        raise ValueError("--publish-after-run requires --publish-data-dir")
    publish_data_dir = Path(args.publish_data_dir).resolve()
    publish_command = [args.python_bin, "-u", str(PUBLISH_SCRIPT), "--data-dir", str(publish_data_dir)]

    hosted_index = Path(args.hosted_index).resolve() if args.hosted_index else publish_data_dir / "experiment-index.json"
    if hosted_index.is_file():
        publish_command.extend(["--index", str(hosted_index)])
    else:
        print(f"Hosted index missing, continuing without it: {hosted_index}")

    for index_path in unique_indexes:
        publish_command.extend(["--index", str(index_path)])

    publish_command.extend(["--inspection-keep", str(args.inspection_keep)])
    if args.publish_commit:
        publish_command.append("--commit")
    if args.publish_push:
        publish_command.append("--push")
    if args.publish_message:
        publish_command.extend(["--message", args.publish_message])

    print(f"{timestamp()} PUBLISH cmd={' '.join(publish_command)}")
    subprocess.run(publish_command, check=True)


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv).resolve() if args.csv else default_csv_path().resolve()
    ensure_file(csv_path, "Experiment CSV")
    ensure_file(TRAIN_SCRIPT, "Training script")

    input_dir = Path(args.input_dir).resolve()
    ensure_dir(input_dir, "Input dir")
    default_input = DEFAULT_INPUT_DIR.resolve()
    if input_dir != default_input and not args.allow_dataset_override:
        raise ValueError(
            f"Non-default input dir requires --allow-dataset-override. input={input_dir} default={default_input}"
        )

    champion_mapping = Path(args.champion_mapping_path).resolve() if args.champion_mapping_path else discover_champion_mapping()
    ensure_file(champion_mapping, "Champion mapping")

    rows = load_rows(csv_path)
    selected_rows = select_rows(rows, args)
    if not selected_rows:
        raise ValueError("No rows selected from CSV. Adjust filters (--status/--run-id/--test-group/--test-batch).")

    for row in selected_rows:
        dataset_label = row.values.get("dataset_label", "")
        if input_dir == default_input and dataset_label and dataset_label != DEFAULT_DATASET_LABEL:
            if not args.allow_dataset_override:
                raise ValueError(
                    "CSV row dataset_label differs from canonical default while using default input dir. "
                    "Use --allow-dataset-override for intentional deviations."
                )

    output_root = derive_output_root(args, csv_path)
    used_output_dirs: set[Path] = set()
    specs = [
        build_run_spec(
            row,
            args=args,
            output_root=output_root,
            used_output_dirs=used_output_dirs,
            champion_mapping_path=champion_mapping,
            input_dir=input_dir,
        )
        for row in selected_rows
    ]
    print_plan(specs, output_root)
    if not args.execute:
        print("Dry-run only. Add --execute to run.")
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    results = execute_specs(specs, max_parallel=args.max_parallel)
    failed = [result for result in results if result.return_code != 0]

    print("Run results:")
    for result in results:
        status = "completed" if result.return_code == 0 else "failed"
        print(
            f"- row={result.spec.row.row_number} selector={result.spec.row.selector_id} status={status} "
            f"run_id={result.produced_run_id} accuracy={result.accuracy} loss={result.loss} "
            f"log={result.spec.log_path}"
        )

    if args.backfill_csv:
        backfill_csv(csv_path, results)
        print(f"CSV backfilled: {csv_path}")

    if args.publish_after_run:
        run_publish(args, results)

    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
