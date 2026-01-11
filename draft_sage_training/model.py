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
        self.action_embedding = nn.Embedding(feature_dims["num_actions"], 2)
        self.side_embedding = nn.Embedding(feature_dims["num_sides"], 2)
        self.event_embedding = nn.Embedding(feature_dims["num_events"], 4)
        self.league_embedding = nn.Embedding(feature_dims["num_leagues"], 4)
        self.team_embedding = nn.Embedding(feature_dims["num_teams"], 8)

        draft_input_size = feature_dims["draft_sequence"] * 16 + 4 + 2 + 2 + 4 + 4 + 8
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
        action_type = features["action_type"]
        side = features["side"]
        event_index = features["event_index"]
        league_index = features["league_index"]
        team_index = features["team_index"]
        champion_priors = features.get("champion_priors")

        draft_embedded = self.champion_embedding(draft_sequence)
        draft_flat = draft_embedded.view(draft_embedded.size(0), -1)
        patch_embedded = self.patch_embedding(patch_index)
        action_embedded = self.action_embedding(action_type)
        side_embedded = self.side_embedding(side)
        event_embedded = self.event_embedding(event_index)
        league_embedded = self.league_embedding(league_index)
        team_embedded = self.team_embedding(team_index)
        combined = torch.cat(
            [
                draft_flat,
                patch_embedded,
                action_embedded,
                side_embedded,
                event_embedded,
                league_embedded,
                team_embedded,
            ],
            dim=1,
        )

        draft_encoded = self.draft_encoder(combined)
        logits = self.classifier(draft_encoded)
        if champion_priors is not None:
            # Add per-champion priors as a logit bias (A/B switchable feature).
            logits = logits + champion_priors
        return logits
