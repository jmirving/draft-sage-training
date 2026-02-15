from __future__ import annotations

import re
from typing import Iterable


DEFAULT_ROLE_ORDER = ["top", "jungle", "mid", "bot", "support"]
PATCH_MAJOR_MINOR_PATTERN = re.compile(r"^\s*(\d+)(?:\.(\d+))?")


def parse_patch_major_minor(patch_value: object) -> tuple[int, int] | None:
    if patch_value is None:
        return None
    match = PATCH_MAJOR_MINOR_PATTERN.match(str(patch_value))
    if not match:
        return None
    major = int(match.group(1))
    minor_token = match.group(2)
    minor = int(minor_token) if minor_token is not None else 0
    return (major, minor)


def build_causal_patch_weights(
    patches: list[str],
    target_index: int,
    *,
    latest_major_weight: float = 4.0,
    patch_recency_decay: float = 0.9,
    older_major_decay: float = 0.35,
) -> dict[str, float]:
    if target_index < 0 or target_index >= len(patches):
        raise IndexError(f"target_index out of range for patches: {target_index}")
    if latest_major_weight <= 0:
        raise ValueError("latest_major_weight must be > 0.")
    if patch_recency_decay <= 0:
        raise ValueError("patch_recency_decay must be > 0.")
    if older_major_decay <= 0:
        raise ValueError("older_major_decay must be > 0.")

    target_patch = patches[target_index]
    target_parts = parse_patch_major_minor(target_patch)
    target_major = target_parts[0] if target_parts is not None else None

    weights: dict[str, float] = {}
    for source_index in range(target_index + 1):
        source_patch = patches[source_index]
        source_parts = parse_patch_major_minor(source_patch)
        source_major = source_parts[0] if source_parts is not None else None

        distance = target_index - source_index
        recency_weight = patch_recency_decay ** distance

        major_weight = 1.0
        if target_major is not None and source_major is not None:
            if source_major == target_major:
                major_weight = latest_major_weight
            elif source_major < target_major:
                major_weight = older_major_decay ** (target_major - source_major)
            else:
                major_weight = 0.0

        weights[source_patch] = major_weight * recency_weight

    return weights


def validate_role_priors_payload(
    payload: dict,
    champion_names: Iterable[str],
    *,
    allow_missing: bool = False,
    tolerance: float = 1e-6,
) -> list[str]:
    if not isinstance(payload, dict):
        raise ValueError("Role priors payload must be a dict.")

    weight_type = payload.get("weight_type")
    if weight_type and weight_type != "role-priors":
        raise ValueError(f"Unexpected weight_type in role priors payload: {weight_type}")

    roles = payload.get("roles") or DEFAULT_ROLE_ORDER
    if roles != DEFAULT_ROLE_ORDER:
        raise ValueError(f"Role priors must use roles {DEFAULT_ROLE_ORDER}")

    weights = payload.get("weights")
    if not isinstance(weights, dict):
        raise ValueError("Role priors payload missing weights map.")

    champion_set = {str(name) for name in champion_names}
    weight_keys = {str(name) for name in weights.keys()}
    unknown = sorted(weight_keys - champion_set)
    if unknown:
        raise ValueError(f"Unknown champions in role priors: {unknown}")

    if not allow_missing:
        missing = sorted(champion_set - weight_keys)
        if missing:
            raise ValueError(f"Missing champions in role priors: {missing}")

    for champion, role_weights in weights.items():
        if not isinstance(role_weights, dict):
            raise ValueError(f"Role priors for {champion} must be a dict.")

        role_keys = set(role_weights.keys())
        if role_keys != set(DEFAULT_ROLE_ORDER):
            raise ValueError(
                f"Role priors for {champion} must include roles {DEFAULT_ROLE_ORDER}."
            )

        total = 0.0
        for role in DEFAULT_ROLE_ORDER:
            value = role_weights.get(role)
            if value is None:
                raise ValueError(f"Role priors for {champion} missing role {role}.")
            weight = float(value)
            if weight < 0:
                raise ValueError(f"Negative weight for {champion} role {role}.")
            total += weight

        if abs(total - 1.0) > tolerance:
            raise ValueError(
                f"Role priors for {champion} must sum to 1 (got {total})."
            )

    return list(roles)
