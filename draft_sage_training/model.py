from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class DraftMLP(nn.Module):
    """MLP for draft data using draft sequence plus patch embedding."""

    def __init__(self, feature_dims: Dict[str, int], hidden_size: int = 256, output_size: int = 2):
        super().__init__()
        self.feature_dims = feature_dims

        self.champion_embedding = nn.Embedding(feature_dims["num_champions"], 16)
        self.patch_embedding = nn.Embedding(feature_dims["num_patches"], 4)

        draft_input_size = feature_dims["draft_sequence"] * 16 + 4
        self.draft_encoder = nn.Sequential(
            nn.Linear(draft_input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        draft_sequence = features["draft_sequence"]
        patch_index = features["patch_index"]
        champion_priors = features.get("champion_priors")

        draft_embedded = self.champion_embedding(draft_sequence)
        draft_flat = draft_embedded.view(draft_embedded.size(0), -1)
        patch_embedded = self.patch_embedding(patch_index)
        combined = torch.cat([draft_flat, patch_embedded], dim=1)

        draft_encoded = self.draft_encoder(combined)
        logits = self.classifier(draft_encoded)
        if champion_priors is not None:
            # Add per-champion priors as a logit bias (A/B switchable feature).
            logits = logits + champion_priors
        return logits
