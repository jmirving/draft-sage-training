from __future__ import annotations

import logging
import zlib
from datetime import datetime, timezone
from typing import Iterable

import numpy as np
import torch

from draft_sage_training.dataset import DraftDataset
from draft_sage_training.utils.draft_order import DRAFT_ORDER

DEFAULT_INSPECTION_SAMPLE_SIZE = 200
DEFAULT_INSPECTION_TOP_K = 5


def derive_inspection_seed(run_id: str) -> int:
    return zlib.crc32(run_id.encode("utf-8")) & 0xFFFFFFFF


def stratified_sample_indices(
    samples: list[dict],
    candidate_indices: Iterable[int],
    sample_size: int,
    seed: int,
) -> list[int]:
    num_slots = len(DRAFT_ORDER)
    buckets: dict[int, list[int]] = {slot: [] for slot in range(1, num_slots + 1)}
    for idx in candidate_indices:
        idx_value = int(idx)
        row = samples[idx_value]
        event_index = row.get("event_index")
        if event_index is None:
            continue
        slot = int(event_index) + 1
        if slot < 1 or slot > num_slots:
            continue
        buckets[slot].append(idx_value)

    rng = np.random.default_rng(seed)
    per_slot = sample_size // num_slots if num_slots else sample_size
    selected: list[int] = []
    for slot in range(1, num_slots + 1):
        bucket = buckets[slot]
        if not bucket:
            continue
        if len(bucket) <= per_slot:
            selected.extend(bucket)
        else:
            selected.extend(rng.choice(bucket, size=per_slot, replace=False).tolist())

    remaining = sample_size - len(selected)
    if remaining > 0:
        remaining_pool = [
            idx
            for slot in range(1, num_slots + 1)
            for idx in buckets[slot]
            if idx not in selected
        ]
        if remaining_pool:
            pick = rng.choice(
                remaining_pool,
                size=min(remaining, len(remaining_pool)),
                replace=False,
            )
            selected.extend(pick.tolist())

    return selected


def _champion_name(dataset: DraftDataset, champion_index: int) -> str:
    if not champion_index:
        return "MISSING"
    return dataset.idx2champion.get(champion_index, "UNKNOWN")


def _is_nan(value) -> bool:
    return isinstance(value, (float, np.floating)) and np.isnan(value)


def _to_int(value):
    if value is None or _is_nan(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_str(value):
    if value is None or _is_nan(value):
        return None
    return str(value)


def _format_draft_sequence(dataset: DraftDataset, draft_sequence: list[int]) -> list[str | int]:
    formatted: list[str | int] = []
    for champ_index in draft_sequence:
        if champ_index == 0:
            formatted.append(0)
        else:
            formatted.append(_champion_name(dataset, champ_index))
    return formatted


def build_inspection_bundle(
    *,
    run_id: str,
    model: torch.nn.Module,
    dataset: DraftDataset,
    candidate_indices: Iterable[int],
    device: torch.device,
    sample_size: int = DEFAULT_INSPECTION_SAMPLE_SIZE,
    top_k: int = DEFAULT_INSPECTION_TOP_K,
    seed: int | None = None,
) -> dict:
    sample_seed = derive_inspection_seed(run_id) if seed is None else seed
    selected_indices = stratified_sample_indices(
        dataset.samples,
        candidate_indices=candidate_indices,
        sample_size=sample_size,
        seed=sample_seed,
    )
    if not selected_indices:
        logging.warning("No inspection samples available after stratification.")

    model.eval()
    samples_payload = []
    with torch.no_grad():
        for idx in selected_indices:
            row = dataset.samples[idx]
            item = dataset[idx]
            features = {
                "draft_sequence": item["draft_sequence"].unsqueeze(0).to(device),
                "patch_index": item["patch_index"].unsqueeze(0).to(device),
                "action_type": item["action_type"].unsqueeze(0).to(device),
                "side": item["side"].unsqueeze(0).to(device),
                "event_index": item["event_index"].unsqueeze(0).to(device),
                "league_index": item["league_index"].unsqueeze(0).to(device),
                "team_index": item["team_index"].unsqueeze(0).to(device),
            }
            if "champion_priors" in item:
                features["champion_priors"] = item["champion_priors"].unsqueeze(0).to(device)
            if "role_priors" in item:
                features["role_priors"] = item["role_priors"].unsqueeze(0).to(device)

            outputs = model(features)
            output_mask = item["output_mask"].unsqueeze(0).to(device)
            masked_outputs = outputs.masked_fill(output_mask == 0, -1e9)
            probs = torch.softmax(masked_outputs, dim=1)
            k = min(top_k, probs.size(1))
            top_probs, top_indices = torch.topk(probs, k=k, dim=1)

            top_k_payload = []
            for score, index in zip(top_probs[0].tolist(), top_indices[0].tolist()):
                champion_index = index + 1
                top_k_payload.append(
                    {
                        "champion": _champion_name(dataset, champion_index),
                        "score": float(score),
                    }
                )

            target_champion = _champion_name(dataset, row.get("target", 0))
            draft_sequence = _format_draft_sequence(dataset, row.get("draft_sequence", []))

            samples_payload.append(
                {
                    "gameid": _to_str(row.get("gameid")),
                    "seriesid": _to_str(row.get("seriesid")),
                    "game_number": _to_int(row.get("game_number")),
                    "league": _to_str(row.get("league") or row.get("league_key")),
                    "patch": _to_str(row.get("patch")),
                    "side": row.get("side"),
                    "action_type": row.get("action_type"),
                    "slot": int(row.get("event_index", 0)) + 1,
                    "draft_sequence": draft_sequence,
                    "series_used_champions": row.get("series_used_champions", []),
                    "target_champion": target_champion,
                    "top_k": top_k_payload,
                }
            )

    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sample_method": "stratified_by_slot",
        "sample_seed": sample_seed,
        "sample_size": len(samples_payload),
        "samples": samples_payload,
    }
