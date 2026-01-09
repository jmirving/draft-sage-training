# Moved from draft-sage pbai.utils.champ_enum.

import json
import os
from enum import Enum


def create_champ_enum(champions_path: str | None = None):
    if champions_path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        champions_path = os.path.join(here, "../../resources/champions.json")
    with open(champions_path, encoding="utf-8-sig") as handle:
        data = json.load(handle)
    champ_dict = {value["name"].upper(): int(value["key"]) for value in data["data"].values()}
    champ_dict["MISSING"] = -1
    return Enum("ChampEnum", champ_dict)
