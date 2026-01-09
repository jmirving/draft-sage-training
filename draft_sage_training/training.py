from __future__ import annotations

import logging

from draft_sage_training.config import TrainingConfig


def train(config: TrainingConfig) -> int:
    logging.info("Training scaffold invoked; pipeline not implemented yet.")
    logging.info("Config: %s", config)
    return 1
