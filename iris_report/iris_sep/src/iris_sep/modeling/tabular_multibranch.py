"""Compact schema-aligned primary IRIS-SEP model.

The model consumes one 24-hour aggregate vector per modality. It does not
pretend that min/max/mean columns are temporal steps. Future-target columns are
excluded by the data adapter before this interface.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping, Optional

import torch
from torch import Tensor, nn


MODALITIES = ("magnetic", "eruption", "particle_context")


@dataclass(frozen=True)
class TabularModelConfig:
    magnetic_features: int
    eruption_features: int
    particle_context_features: int
    branch_hidden: int = 32
    embedding_dim: int = 16
    shared_dim: int = 32
    dropout: float = 0.10
    missing_modality_dropout: float = 0.15
    magnetic_feature_names: tuple[str, ...] = ()
    eruption_feature_names: tuple[str, ...] = ()
    particle_context_feature_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("magnetic_features", "eruption_features", "particle_context_features", "branch_hidden", "embedding_dim", "shared_dim"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 <= self.dropout < 1 or not 0 <= self.missing_modality_dropout < 1:
            raise ValueError("dropout probabilities must be in [0,1)")
        named = self.feature_names
        supplied = [bool(names) for names in named.values()]
        if any(supplied) and not all(supplied):
            raise ValueError("feature names must bind every modality or none")
        for name, names in named.items():
            if names and (len(names) != self.feature_counts[name] or len(set(names)) != len(names) or any(not item for item in names)):
                raise ValueError(f"{name} feature names must be unique, nonempty, and match its width")

    @property
    def feature_counts(self) -> dict[str, int]:
        return {
            "magnetic": self.magnetic_features,
            "eruption": self.eruption_features,
            "particle_context": self.particle_context_features,
        }

    @property
    def feature_names(self) -> dict[str, tuple[str, ...]]:
        return {
            "magnetic": self.magnetic_feature_names,
            "eruption": self.eruption_feature_names,
            "particle_context": self.particle_context_feature_names,
        }

    @property
    def schema_bound(self) -> bool:
        return all(self.feature_names.values())

    @property
    def feature_schema_sha256(self) -> str | None:
        if not self.schema_bound:
            return None
        payload = {name: list(values) for name, values in self.feature_names.items()}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class BranchInput:
    values: Tensor
    observed_mask: Tensor


@dataclass(frozen=True)
class PrimaryForecastOutput:
    primary_logit: Tensor
    gate_weights: Tensor
    modality_available: Tensor
    all_missing: Tensor
    shared_embedding: Tensor


class _Branch(nn.Module):
    def __init__(self, features: int, hidden: int, embedding: int, dropout: float) -> None:
        super().__init__()
        self.features = features
        self.network = nn.Sequential(
            nn.Linear(features * 2, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embedding),
            nn.LayerNorm(embedding),
            nn.SiLU(),
        )

    def forward(self, branch: BranchInput) -> tuple[Tensor, Tensor]:
        if branch.values.ndim != 2 or branch.values.shape[1] != self.features:
            raise ValueError(f"values must be [batch,{self.features}]")
        if branch.observed_mask.shape != branch.values.shape:
            raise ValueError("feature mask must exactly match values")
        parameter = next(self.parameters())
        values = branch.values.to(device=parameter.device, dtype=parameter.dtype)
        mask = branch.observed_mask.to(device=parameter.device)
        if mask.dtype != torch.bool:
            if mask.is_complex() or not torch.isfinite(mask).all() or ((mask != 0) & (mask != 1)).any():
                raise ValueError("feature mask must be boolean or contain only exact binary values {0,1}")
        mask = mask.bool()
        if not torch.isfinite(values).all():
            raise ValueError("values must be finite after train-only transformation")
        prepared = torch.cat((values * mask.to(values.dtype), mask.to(values.dtype)), dim=1)
        return self.network(prepared), mask.any(dim=1)


def _missing_modality_keep_mask(
    available: Tensor,
    probability: float,
    generator: Optional[torch.Generator],
) -> Tensor:
    keep = available & (torch.rand(available.shape, device=available.device, generator=generator) >= probability)
    needs_fallback = available.any(dim=1) & ~keep.any(dim=1)
    for row in torch.nonzero(needs_fallback, as_tuple=False).flatten().tolist():
        choices = torch.nonzero(available[row], as_tuple=False).flatten()
        selected = torch.randint(len(choices), (1,), device=available.device, generator=generator)
        keep[row, choices[selected].item()] = True
    return keep


class IRISSEPTabularModel(nn.Module):
    """Three aggregate-feature branches, gated fusion, one primary head."""

    def __init__(self, config: TabularModelConfig) -> None:
        super().__init__()
        self.config = config
        self.branches = nn.ModuleDict({
            name: _Branch(config.feature_counts[name], config.branch_hidden, config.embedding_dim, config.dropout)
            for name in MODALITIES
        })
        gate_input = len(MODALITIES) * config.embedding_dim + len(MODALITIES)
        self.gate = nn.Sequential(
            nn.Linear(gate_input, 16), nn.SiLU(), nn.Linear(16, len(MODALITIES))
        )
        self.shared = nn.Sequential(
            nn.Linear(config.embedding_dim + len(MODALITIES), config.shared_dim),
            nn.LayerNorm(config.shared_dim), nn.SiLU(), nn.Dropout(config.dropout),
        )
        self.primary_head = nn.Linear(config.shared_dim, 1)

    @property
    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(
        self,
        inputs: Mapping[str, BranchInput],
        *,
        apply_missing_modality_dropout: Optional[bool] = None,
        generator: Optional[torch.Generator] = None,
    ) -> PrimaryForecastOutput:
        if set(inputs) != set(MODALITIES):
            raise ValueError(f"inputs must contain exactly {MODALITIES}")
        embeddings = []
        availability = []
        batch_size: int | None = None
        for name in MODALITIES:
            embedding, available = self.branches[name](inputs[name])
            if batch_size is None:
                batch_size = embedding.shape[0]
            elif embedding.shape[0] != batch_size:
                raise ValueError("all modality branches must share the batch dimension")
            embeddings.append(embedding)
            availability.append(available)
        stacked = torch.stack(embeddings, dim=1)
        available = torch.stack(availability, dim=1)
        should_drop = self.training if apply_missing_modality_dropout is None else apply_missing_modality_dropout
        effective = (
            _missing_modality_keep_mask(available, self.config.missing_modality_dropout, generator)
            if should_drop and self.config.missing_modality_dropout > 0
            else available
        )
        masked = stacked * effective.to(stacked.dtype).unsqueeze(-1)
        gate_input = torch.cat((masked.flatten(start_dim=1), effective.to(stacked.dtype)), dim=1)
        logits = self.gate(gate_input).masked_fill(~effective, -1e4)
        weights = torch.softmax(logits, dim=1) * effective.to(stacked.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        shared = self.shared(torch.cat((fused, effective.to(fused.dtype)), dim=1))
        return PrimaryForecastOutput(
            primary_logit=self.primary_head(shared).squeeze(-1),
            gate_weights=weights,
            modality_available=effective,
            all_missing=~effective.any(dim=1),
            shared_embedding=shared,
        )
