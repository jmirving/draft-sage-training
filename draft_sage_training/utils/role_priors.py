from __future__ import annotations

from typing import Iterable


DEFAULT_ROLE_ORDER = ["top", "jungle", "mid", "bot", "support"]


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
