DRAFT_ORDER = [
    ("blue", "ban", 1), ("red", "ban", 1),
    ("blue", "ban", 2), ("red", "ban", 2),
    ("blue", "ban", 3), ("red", "ban", 3),
    ("blue", "pick", 1), ("red", "pick", 1),
    ("red", "pick", 2), ("blue", "pick", 2),
    ("blue", "pick", 3), ("red", "pick", 3),
    ("red", "ban", 4), ("blue", "ban", 4),
    ("red", "ban", 5), ("blue", "ban", 5),
    ("red", "pick", 4), ("blue", "pick", 4),
    ("blue", "pick", 5), ("red", "pick", 5),
]


def draft_order_for_first_pick(first_pick_side: str | None) -> list[tuple[str, str, int]]:
    normalized = (first_pick_side or "").strip().lower()
    if normalized != "red":
        return DRAFT_ORDER
    return [
        ("red" if side == "blue" else "blue", action_type, action_number)
        for side, action_type, action_number in DRAFT_ORDER
    ]
