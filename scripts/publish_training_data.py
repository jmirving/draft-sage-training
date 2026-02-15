#!/usr/bin/env python3
"""Publish training outputs into the static training data host."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


@dataclass
class RunBundle:
    run_id: str
    index_path: Path
    run_entry: dict
    summary_path: Path
    summary: dict
    timestamp: float


def default_data_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root.parent / "draft-sage-training-data" / "public" / "training"


def default_index_candidates(workspace_root: Path) -> list[Path]:
    return [
        workspace_root / ".tmp" / "training-clean-2025-all" / "experiment-index.json",
        workspace_root / ".tmp" / "training-clean-2025-seriesid-index" / "experiment-index.json",
        workspace_root / ".tmp" / "training-autopublish" / "experiment-index.json",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish experiment index + run metadata into draft-sage-training-data.",
    )
    parser.add_argument(
        "--index",
        action="append",
        help="Path to an experiment-index.json file (repeatable).",
    )
    parser.add_argument(
        "--use-default-indexes",
        action="store_true",
        help="Include the default workspace index list when publishing.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(default_data_dir()),
        help="Target data host training directory (default: ../draft-sage-training-data/public/training).",
    )
    parser.add_argument(
        "--inspection-keep",
        type=int,
        default=10,
        help="Number of newest inspection bundles to keep on the data host.",
    )
    parser.add_argument(
        "--commit",
        action="store_true",
        help="Commit changes in the data host repo after publishing.",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push data host repo changes after committing.",
    )
    parser.add_argument(
        "--message",
        help="Commit message override when using --commit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log actions without writing files.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Verbosity for log messages.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return data


def write_json(path: Path, payload: dict, dry_run: bool) -> None:
    if dry_run:
        logging.info("Would write %s", path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_timestamp(value: str | None) -> float | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.timestamp()
    except ValueError:
        return None


def parse_run_id_timestamp(run_id: str) -> float | None:
    try:
        parsed = datetime.strptime(run_id, "%Y%m%d_%H%M%S")
        return parsed.replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


def run_timestamp(summary: dict, run_id: str) -> float:
    for field in ("updated_at", "created_at"):
        parsed = parse_timestamp(summary.get(field))
        if parsed is not None:
            return parsed
    fallback = parse_run_id_timestamp(run_id)
    return fallback if fallback is not None else 0.0


def sanitize_dataset(dataset: dict | None) -> dict | None:
    if not isinstance(dataset, dict):
        return dataset
    cleaned = dict(dataset)
    cleaned.pop("input_dir", None)
    for key in ("champion_eligibility", "champion_priors", "role_priors"):
        value = cleaned.get(key)
        if isinstance(value, dict):
            value = dict(value)
            value.pop("path", None)
            value.pop("dir", None)
            cleaned[key] = value
    return cleaned


def normalize_manifest_path(path_value: str) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return str(Path("manifests") / path.name)
    safe_parts = [part for part in path.parts if part not in (".", "..")]
    if not safe_parts:
        return str(Path("manifests") / path.name)
    return str(Path(*safe_parts))


def resolve_source_path(base_dir: Path, path_value: str | None, workspace_root: Path) -> Path | None:
    if not path_value:
        return None
    candidate = Path(path_value)
    if candidate.is_absolute():
        if candidate.exists():
            return candidate
        if path_value.startswith("/.tmp/"):
            fallback = workspace_root / ".tmp" / path_value.removeprefix("/.tmp/")
            return fallback
        return candidate
    return (base_dir / candidate).resolve()


def copy_file(source: Path | None, dest: Path, dry_run: bool) -> bool:
    if source is None:
        return False
    if not source.exists():
        logging.warning("Missing source file: %s", source)
        return False
    try:
        if source.resolve() == dest.resolve():
            return True
    except OSError:
        pass
    if dry_run:
        logging.info("Would copy %s -> %s", source, dest)
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    return True


def select_inspection_runs(runs: list[RunBundle], keep: int) -> set[str]:
    if keep <= 0:
        return set()
    ranked = sorted(runs, key=lambda run: run.timestamp, reverse=True)
    return {run.run_id for run in ranked[:keep]}


def build_baseline_pointers(index_payloads: list[dict]) -> dict:
    chosen = {}
    best_timestamp = -1.0
    for payload in index_payloads:
        if not isinstance(payload, dict):
            continue
        has_pointers = payload.get("true_baseline_run_id") or payload.get("baseline_to_beat_run_id")
        if not has_pointers:
            continue
        candidate_ts = parse_timestamp(payload.get("baseline_updated_at")) or 0.0
        if candidate_ts >= best_timestamp:
            best_timestamp = candidate_ts
            chosen = {
                "true_baseline_run_id": payload.get("true_baseline_run_id"),
                "baseline_to_beat_run_id": payload.get("baseline_to_beat_run_id"),
                "baseline_updated_at": payload.get("baseline_updated_at"),
            }
    return chosen


def find_git_root(path: Path) -> Path | None:
    current = path.resolve()
    for parent in [current, *current.parents]:
        if (parent / ".git").exists():
            return parent
    return None


def run_git(command: list[str], cwd: Path, dry_run: bool) -> subprocess.CompletedProcess | None:
    if dry_run:
        logging.info("Would run: %s (cwd=%s)", " ".join(command), cwd)
        return None
    return subprocess.run(command, cwd=str(cwd), check=True, capture_output=True, text=True)


def has_git_changes(cwd: Path, dry_run: bool) -> bool:
    if dry_run:
        return True
    result = run_git(["git_local", "status", "--porcelain"], cwd, dry_run=False)
    return bool(result.stdout.strip())


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    raw_indexes = args.index or []
    index_paths = [Path(path).resolve() for path in raw_indexes if path]
    data_dir = Path(args.data_dir).resolve()
    workspace_root = Path(__file__).resolve().parents[2]
    if args.use_default_indexes or not index_paths:
        for candidate in default_index_candidates(workspace_root):
            if candidate.exists():
                index_paths.append(candidate.resolve())
    # De-dupe while preserving order.
    seen = set()
    index_paths = [p for p in index_paths if not (p in seen or seen.add(p))]
    if not index_paths:
        raise ValueError("No experiment-index.json files found to publish.")
    runs: list[RunBundle] = []
    index_payloads: list[dict] = []

    for index_path in index_paths:
        payload = load_json(index_path)
        index_payloads.append(payload)
        runs_list = payload.get("runs", [])
        if not isinstance(runs_list, list):
            raise ValueError(f"Index runs must be a list: {index_path}")
        for run_entry in runs_list:
            if not isinstance(run_entry, dict):
                continue
            summary_path_value = run_entry.get("summary_path")
            if not summary_path_value:
                logging.warning("Run %s missing summary_path in %s", run_entry.get("run_id"), index_path)
                continue
            summary_path = resolve_source_path(index_path.parent, summary_path_value, workspace_root)
            if summary_path is None or not summary_path.exists():
                logging.warning("Missing summary file: %s", summary_path)
                continue
            summary = load_json(summary_path)
            run_id = summary.get("run_id") or run_entry.get("run_id")
            if not run_id:
                logging.warning("Summary missing run_id: %s", summary_path)
                continue
            runs.append(
                RunBundle(
                    run_id=str(run_id),
                    index_path=index_path,
                    run_entry=dict(run_entry),
                    summary_path=summary_path,
                    summary=summary,
                    timestamp=run_timestamp(summary, str(run_id)),
                )
            )

    if not runs:
        raise ValueError("No runs discovered from provided indexes.")

    inspection_keep = select_inspection_runs(runs, args.inspection_keep)

    combined_runs: dict[str, dict] = {}
    for bundle in runs:
        run_id = bundle.run_id
        run_entry = dict(bundle.run_entry)
        summary = dict(bundle.summary)
        run_dir = bundle.summary_path.parent

        run_entry["run_id"] = run_id
        run_entry["summary_path"] = f"runs/{run_id}/summary.json"
        run_entry["status"] = summary.get("status") or run_entry.get("status")
        if not run_entry.get("metrics"):
            run_entry["metrics"] = summary.get("metrics")
        dataset = sanitize_dataset(summary.get("dataset"))
        if isinstance(dataset, dict) and dataset.get("manifest_path"):
            manifest_path = dataset.get("manifest_path")
            if isinstance(manifest_path, str):
                normalized_manifest = normalize_manifest_path(manifest_path)
                manifest_source = resolve_source_path(bundle.index_path.parent, manifest_path, workspace_root)
                manifest_target = data_dir / normalized_manifest
                if copy_file(manifest_source, manifest_target, args.dry_run):
                    dataset["manifest_path"] = normalized_manifest
                else:
                    logging.warning("Manifest missing for run %s: %s", run_id, manifest_path)
        summary["dataset"] = dataset
        if dataset is not None:
            run_entry["dataset"] = dataset

        combined_runs[run_id] = run_entry

        paths = dict(summary.get("paths") or {})
        paths.pop("model", None)

        inspection_path_value = paths.get("inspection_samples")
        inspection_source = (
            resolve_source_path(run_dir, inspection_path_value, workspace_root)
            if inspection_path_value
            else None
        )

        target_run_dir = data_dir / "runs" / run_id
        if run_id in inspection_keep and inspection_source and inspection_source.exists():
            inspection_target = data_dir / "runs" / run_id / Path(inspection_path_value).name
            copy_file(inspection_source, inspection_target, args.dry_run)
            summary["inspection_status"] = "available"
            summary["inspection_error"] = None
            paths["inspection_samples"] = Path(inspection_path_value).name
        else:
            if inspection_path_value:
                stale_path = target_run_dir / Path(inspection_path_value).name
                if stale_path.exists():
                    if args.dry_run:
                        logging.info("Would remove pruned inspection bundle: %s", stale_path)
                    else:
                        stale_path.unlink()
            paths.pop("inspection_samples", None)
            summary["inspection_status"] = "missing"
            reason = "Inspection bundle pruned on publish."
            if inspection_path_value and inspection_source and not inspection_source.exists():
                reason = "Inspection bundle missing from run outputs."
            if run_id not in inspection_keep:
                reason = "Inspection bundle pruned on publish."
            summary["inspection_error"] = reason

        summary["paths"] = paths

        write_json(target_run_dir / "summary.json", summary, args.dry_run)

        for key in ("config", "metrics"):
            source = resolve_source_path(run_dir, paths.get(key), workspace_root)
            if source:
                copy_file(source, target_run_dir / Path(paths.get(key)).name, args.dry_run)

    combined_index = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "runs": sorted(combined_runs.values(), key=lambda item: item.get("run_id") or ""),
    }
    baseline = build_baseline_pointers(index_payloads)
    combined_index.update({k: v for k, v in baseline.items() if v})

    write_json(data_dir / "experiment-index.json", combined_index, args.dry_run)

    if args.commit or args.push:
        git_root = find_git_root(data_dir)
        if not git_root:
            raise RuntimeError(f"Unable to locate git root for {data_dir}")

        if args.push:
            args.commit = True

        if has_git_changes(git_root, args.dry_run):
            run_git(["git_local", "add", "-A"], git_root, args.dry_run)
            message = args.message or "Publish training data"
            run_git(["git_local", "commit", "-m", message], git_root, args.dry_run)
            if args.push:
                run_git(["git_net", "pull", "--rebase"], git_root, args.dry_run)
                run_git(["git_net", "push"], git_root, args.dry_run)
        else:
            logging.info("No data host changes detected; skipping commit/push.")

    logging.info("Publish complete -> %s", data_dir)


if __name__ == "__main__":
    main()
