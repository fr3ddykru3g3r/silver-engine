"""Compact causal multimodal prototype for IRIS-SEP.

This module is intentionally data-source agnostic.  It only defines the model
contract used by the later SEP-PRISM integration:

* every modality is an ``[batch, time, feature]`` causal sequence;
* an observation mask and time-since-observation (in hours) are supplied with
  each sequence;
* availability is explicit, so delayed or absent feeds can be represented;
* no labels, cohorts, test identities, or benchmark outcomes are loaded here.

The architecture is sized for a Colab T4.  A causal convolution never pads on
the right, so a representation at time ``t`` cannot use an input after ``t``.
The output heads return logits for binary tasks, log-flux quantiles, and an
hourly onset-hazard logit sequence.  This file contains no scientific result;
the smoke test uses generated tensors only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping, MutableMapping, Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


MODALITY_NAMES: tuple[str, ...] = ("magnetic", "eruption", "particle")


def _validate_positive_int(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class IRISSEPConfig:
    """Shape and capacity contract for :class:`IRISSEPModel`.

    ``*_input_features`` are supplied by the data adapter after the benchmark
    feature manifest is frozen.  The model does not infer them from a dataset.
    ``time_since_observation`` is measured in hours and is log-normalized using
    ``max_time_since_hours`` before it is appended to each expert input.
    """

    magnetic_input_features: int
    eruption_input_features: int
    particle_input_features: int
    lookback_steps: int = 24
    hidden_channels: int = 32
    embedding_dim: int = 48
    shared_dim: int = 64
    temporal_layers: int = 2
    kernel_size: int = 3
    dropout: float = 0.10
    onset_horizon_bins: int = 24
    max_time_since_hours: float = 72.0
    missing_modality_dropout: float = 0.15
    quantile_levels: tuple[float, ...] = (0.10, 0.50, 0.90)

    def __post_init__(self) -> None:
        for name in (
            "magnetic_input_features",
            "eruption_input_features",
            "particle_input_features",
            "lookback_steps",
            "hidden_channels",
            "embedding_dim",
            "shared_dim",
            "temporal_layers",
            "onset_horizon_bins",
        ):
            _validate_positive_int(name, getattr(self, name))
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.max_time_since_hours <= 0.0 or not math.isfinite(self.max_time_since_hours):
            raise ValueError("max_time_since_hours must be finite and positive")
        if not 0.0 <= self.missing_modality_dropout < 1.0:
            raise ValueError("missing_modality_dropout must be in [0, 1)")
        levels = tuple(float(q) for q in self.quantile_levels)
        if len(levels) != 3 or any(not math.isfinite(q) or not 0.0 < q < 1.0 for q in levels):
            raise ValueError("quantile_levels must contain three values in (0, 1)")
        if tuple(sorted(levels)) != levels:
            raise ValueError("quantile_levels must be sorted")
        object.__setattr__(self, "quantile_levels", levels)

    @property
    def input_features(self) -> dict[str, int]:
        return {
            "magnetic": self.magnetic_input_features,
            "eruption": self.eruption_input_features,
            "particle": self.particle_input_features,
        }

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable configuration metadata for checkpoints."""

        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> "IRISSEPConfig":
        payload = dict(values)
        if "quantile_levels" in payload:
            payload["quantile_levels"] = tuple(payload["quantile_levels"])  # type: ignore[arg-type]
        return cls(**payload)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ModalityInput:
    """One causal modality sequence.

    Attributes:
        values: Float-like tensor with shape ``[B, T, F]``.
        observed_mask: Boolean/float tensor with shape ``[B, T]`` or
            feature-level shape ``[B, T, F]``.  A one means observed.
        time_since_observation_hours: Non-negative tensor with shape ``[B, T]``
            or ``[B, T, 1]``.  Values are hours, not minutes or seconds.
        available: Optional sample-level boolean tensor ``[B]``.  When omitted,
            availability is derived from the observation mask.  Supplying this
            field is recommended when a feed is delayed but its mask semantics
            need to remain distinguishable from an all-zero measurement.
    """

    values: Tensor
    observed_mask: Tensor
    time_since_observation_hours: Tensor
    available: Optional[Tensor] = None


@dataclass(frozen=True)
class IRISSEPInputs:
    """Named input bundle to make integration interfaces unambiguous."""

    magnetic: ModalityInput
    eruption: ModalityInput
    particle: ModalityInput

    def as_mapping(self) -> dict[str, ModalityInput]:
        return {
            "magnetic": self.magnetic,
            "eruption": self.eruption,
            "particle": self.particle,
        }


@dataclass
class ForecastOutput:
    """Model outputs and diagnostics.

    ``peak_flux_log_quantiles`` are ordered estimates in log-pfu space at the
    configured quantile levels.  ``onset_hazard_logits`` has one value per
    configured future hour/bin.  Binary heads are logits so the training code
    can choose its class weighting without hidden threshold decisions.
    """

    primary_logit: Tensor
    high_energy_logit: Tensor
    peak_flux_log_quantiles: Tensor
    onset_hazard_logits: Tensor
    flare_activity_logit: Tensor
    cme_activity_logit: Tensor
    shared_embedding: Tensor
    modality_embeddings: Tensor
    gate_weights: Tensor
    modality_available: Tensor
    quantile_levels: tuple[float, ...] = (0.10, 0.50, 0.90)

    @property
    def primary_probability(self) -> Tensor:
        return torch.sigmoid(self.primary_logit)

    @property
    def high_energy_probability(self) -> Tensor:
        return torch.sigmoid(self.high_energy_logit)

    @property
    def onset_hazard_probability(self) -> Tensor:
        return torch.sigmoid(self.onset_hazard_logits)

    def as_dict(self, *, include_probabilities: bool = False) -> dict[str, Tensor]:
        result = {
            "primary_logit": self.primary_logit,
            "high_energy_logit": self.high_energy_logit,
            "peak_flux_log_quantiles": self.peak_flux_log_quantiles,
            "onset_hazard_logits": self.onset_hazard_logits,
            "flare_activity_logit": self.flare_activity_logit,
            "cme_activity_logit": self.cme_activity_logit,
            "shared_embedding": self.shared_embedding,
            "modality_embeddings": self.modality_embeddings,
            "gate_weights": self.gate_weights,
            "modality_available": self.modality_available,
        }
        if include_probabilities:
            result.update(
                {
                    "primary_probability": self.primary_probability,
                    "high_energy_probability": self.high_energy_probability,
                    "onset_hazard_probability": self.onset_hazard_probability,
                }
            )
        return result


class CausalConv1d(nn.Conv1d):
    """A left-padded 1-D convolution with no access to future timesteps."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        kernel_size = kwargs.get("kernel_size")
        dilation = kwargs.get("dilation", 1)
        if kernel_size is None and len(args) >= 3:
            kernel_size = args[2]
        if isinstance(kernel_size, tuple):
            if len(kernel_size) != 1:
                raise ValueError("CausalConv1d expects a scalar kernel_size")
            kernel_size = kernel_size[0]
        if not isinstance(kernel_size, int) or kernel_size <= 0:
            raise ValueError("CausalConv1d requires a positive kernel_size")
        if isinstance(dilation, tuple):
            if len(dilation) != 1:
                raise ValueError("CausalConv1d expects a scalar dilation")
            dilation = dilation[0]
        if not isinstance(dilation, int) or dilation <= 0:
            raise ValueError("CausalConv1d requires a positive dilation")
        # Conv1d's own padding is kept at zero; padding is applied on the left
        # in forward, which makes the temporal boundary obvious and testable.
        kwargs["padding"] = 0
        super().__init__(*args, **kwargs)
        self.left_padding = dilation * (kernel_size - 1)

    def forward(self, input: Tensor) -> Tensor:  # noqa: A002 - torch API name
        if input.ndim != 3:
            raise ValueError("CausalConv1d expects [B, C, T]")
        if self.left_padding:
            input = F.pad(input, (self.left_padding, 0))
        return super().forward(input)


class _PerTimestepLayerNorm(nn.Module):
    """LayerNorm across channels at each time, never across the time axis."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, sequence: Tensor) -> Tensor:
        return self.norm(sequence.transpose(1, 2)).transpose(1, 2)


class _ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation=dilation)
        self.norm1 = _PerTimestepLayerNorm(channels)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation=dilation)
        self.norm2 = _PerTimestepLayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence: Tensor) -> Tensor:
        residual = sequence
        hidden = self.conv1(sequence)
        hidden = F.silu(self.norm1(hidden))
        hidden = self.dropout(hidden)
        hidden = self.norm2(self.conv2(hidden))
        return F.silu(residual + hidden)


class CausalTemporalExpert(nn.Module):
    """One compact modality expert with explicit mask/time channels."""

    def __init__(
        self,
        input_features: int,
        *,
        lookback_steps: int,
        hidden_channels: int,
        embedding_dim: int,
        temporal_layers: int,
        kernel_size: int,
        dropout: float,
        max_time_since_hours: float,
    ) -> None:
        super().__init__()
        _validate_positive_int("input_features", input_features)
        _validate_positive_int("lookback_steps", lookback_steps)
        _validate_positive_int("hidden_channels", hidden_channels)
        _validate_positive_int("embedding_dim", embedding_dim)
        _validate_positive_int("temporal_layers", temporal_layers)
        self.input_features = input_features
        self.lookback_steps = lookback_steps
        self.max_time_since_hours = float(max_time_since_hours)
        self.input_projection = nn.Conv1d(input_features + 2, hidden_channels, kernel_size=1)
        self.temporal_blocks = nn.ModuleList(
            [
                _ResidualTemporalBlock(
                    hidden_channels,
                    kernel_size,
                    dilation=2**layer,
                    dropout=dropout,
                )
                for layer in range(temporal_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_channels * 2)
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_channels * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
        )

    @staticmethod
    def _as_float_tensor(value: Tensor, *, name: str, device: torch.device, dtype: torch.dtype) -> Tensor:
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if not (value.is_floating_point() or value.is_complex()):
            value = value.float()
        if value.is_complex():
            raise TypeError(f"{name} must be real-valued")
        value = value.to(device=device, dtype=dtype)
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values; impute/scale in the train-only adapter")
        return value

    def _prepare_input(
        self,
        modality: ModalityInput,
        *,
        batch_size: Optional[int] = None,
        sequence_length: Optional[int] = None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        values = self._as_float_tensor(modality.values, name="values", device=device, dtype=dtype)
        if values.ndim != 3:
            raise ValueError("values must have shape [B, T, F]")
        batch, steps, features = values.shape
        if steps != self.lookback_steps:
            raise ValueError(
                f"values has {steps} time steps; expected configured "
                f"lookback_steps={self.lookback_steps}"
            )
        if features != self.input_features:
            raise ValueError(
                f"values has {features} features; expected {self.input_features} for this modality"
            )
        if batch_size is not None and batch != batch_size:
            raise ValueError("all modalities must have the same batch size")
        if sequence_length is not None and steps != sequence_length:
            raise ValueError("all modalities must have the same sequence length")

        mask = self._as_float_tensor(
            modality.observed_mask,
            name="observed_mask",
            device=device,
            dtype=dtype,
        )
        if mask.ndim == 2:
            if mask.shape != (batch, steps):
                raise ValueError("2-D observed_mask must have shape [B, T]")
            mask_features = mask.unsqueeze(-1)
        elif mask.ndim == 3:
            if mask.shape[:2] != (batch, steps) or mask.shape[-1] not in (1, features):
                raise ValueError("3-D observed_mask must have shape [B, T, 1] or [B, T, F]")
            mask_features = mask if mask.shape[-1] == features else mask.expand(-1, -1, features)
        else:
            raise ValueError("observed_mask must have shape [B, T] or [B, T, F]")
        if (mask_features < 0.0).any() or (mask_features > 1.0).any():
            raise ValueError("observed_mask values must lie in [0, 1]")
        mask_features = mask_features.clamp(0.0, 1.0)
        # The summary mask is both a pooling weight and the scalar mask channel.
        mask_summary = mask_features.mean(dim=-1, keepdim=True)

        time_since = self._as_float_tensor(
            modality.time_since_observation_hours,
            name="time_since_observation_hours",
            device=device,
            dtype=dtype,
        )
        if time_since.ndim == 3 and time_since.shape[-1] == 1:
            time_since = time_since[..., 0]
        if time_since.ndim != 2 or time_since.shape != (batch, steps):
            raise ValueError("time_since_observation_hours must have shape [B, T] or [B, T, 1]")
        if (time_since < 0).any():
            raise ValueError("time_since_observation_hours must be non-negative")
        # Normalization preserves ordering and makes the unit contract explicit.
        normalized_time = torch.log1p(time_since) / math.log1p(self.max_time_since_hours)
        normalized_time = normalized_time.clamp(0.0, 1.0).unsqueeze(-1)

        # Invalid feature values cannot contribute to the embedding.  This is
        # masking, not learned imputation; the data adapter remains responsible
        # for train-only imputation/scaling of genuinely absent numeric values.
        masked_values = values * mask_features
        prepared = torch.cat((masked_values, mask_summary, normalized_time), dim=-1)
        return prepared, mask_summary

    def forward(self, modality: ModalityInput) -> Tensor:
        parameter = next(self.parameters())
        prepared, mask_summary = self._prepare_input(
            modality,
            device=parameter.device,
            dtype=parameter.dtype,
        )
        sequence = prepared.transpose(1, 2)
        sequence = F.silu(self.input_projection(sequence))
        for block in self.temporal_blocks:
            sequence = block(sequence)
        sequence = sequence.transpose(1, 2)  # [B, T, C]

        weights = mask_summary
        denominator = weights.sum(dim=1, keepdim=False).clamp_min(1.0)
        masked_mean = (sequence * weights).sum(dim=1) / denominator

        # The final state is causal even if the last observation is missing: it
        # contains only the prefix through the issue time.  Zero it for a fully
        # absent sequence so all-missing input has a neutral embedding.
        last_state = sequence[:, -1]
        has_observation = (weights.sum(dim=1).squeeze(-1) > 0).to(sequence.dtype).unsqueeze(-1)
        last_state = last_state * has_observation
        pooled = self.output_norm(torch.cat((masked_mean, last_state), dim=-1))
        return self.output_projection(pooled)


def sample_modality_keep_mask(
    available: Tensor,
    dropout_probability: float,
    *,
    generator: Optional[torch.Generator] = None,
    ensure_one_available: bool = True,
) -> Tensor:
    """Sample a train-time keep mask without dropping every available feed.

    ``available`` is ``[B, 3]``.  The result has the same shape and is boolean.
    This helper is separate from the model so a trainer can log the exact
    missing-modality policy or provide a deterministic generator in a replay.
    """

    if available.ndim != 2 or available.shape[1] != len(MODALITY_NAMES):
        raise ValueError(f"available must have shape [B, {len(MODALITY_NAMES)}]")
    if not 0.0 <= dropout_probability < 1.0:
        raise ValueError("dropout_probability must be in [0, 1)")
    available = available.bool()
    keep = torch.rand(available.shape, device=available.device, generator=generator) >= dropout_probability
    keep = keep & available
    if ensure_one_available:
        needs_fallback = available.any(dim=1) & ~keep.any(dim=1)
        if needs_fallback.any():
            # Selecting the first available modality is deterministic and does
            # not inspect labels or any benchmark role.
            first_available = available.to(torch.int64).argmax(dim=1)
            rows = torch.nonzero(needs_fallback, as_tuple=False).squeeze(-1)
            keep[rows, first_available[rows]] = True
    return keep


class IRISSEPModel(nn.Module):
    """Three-expert causal IRIS-SEP prototype with gated late fusion."""

    def __init__(self, config: IRISSEPConfig) -> None:
        super().__init__()
        self.config = config
        self.experts = nn.ModuleDict(
            {
                name: CausalTemporalExpert(
                    config.input_features[name],
                    lookback_steps=config.lookback_steps,
                    hidden_channels=config.hidden_channels,
                    embedding_dim=config.embedding_dim,
                    temporal_layers=config.temporal_layers,
                    kernel_size=config.kernel_size,
                    dropout=config.dropout,
                    max_time_since_hours=config.max_time_since_hours,
                )
                for name in MODALITY_NAMES
            }
        )
        gate_hidden = max(16, config.embedding_dim // 2)
        gate_input_dim = len(MODALITY_NAMES) * config.embedding_dim + len(MODALITY_NAMES)
        self.gate = nn.Sequential(
            nn.Linear(gate_input_dim, gate_hidden),
            nn.LayerNorm(gate_hidden),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(gate_hidden, len(MODALITY_NAMES)),
        )
        self.shared = nn.Sequential(
            nn.Linear(config.embedding_dim + len(MODALITY_NAMES), config.shared_dim),
            nn.LayerNorm(config.shared_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        )

        self.primary_head = nn.Linear(config.shared_dim, 1)
        self.high_energy_head = nn.Linear(config.shared_dim, 1)
        self.peak_flux_head = nn.Linear(config.shared_dim, len(config.quantile_levels))
        self.onset_hazard_head = nn.Linear(config.shared_dim, config.onset_horizon_bins)
        self.flare_activity_head = nn.Linear(config.shared_dim, 1)
        self.cme_activity_head = nn.Linear(config.shared_dim, 1)

    @property
    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    @staticmethod
    def _coerce_input_bundle(
        inputs: IRISSEPInputs | Mapping[str, ModalityInput],
    ) -> dict[str, ModalityInput]:
        if isinstance(inputs, IRISSEPInputs):
            mapping = inputs.as_mapping()
        elif isinstance(inputs, Mapping):
            mapping = dict(inputs)
        else:
            raise TypeError("inputs must be IRISSEPInputs or a mapping of modality names")
        missing = [name for name in MODALITY_NAMES if name not in mapping]
        extra = [name for name in mapping if name not in MODALITY_NAMES]
        if missing:
            raise ValueError(f"missing modality inputs: {', '.join(missing)}")
        if extra:
            raise ValueError(f"unknown modality inputs: {', '.join(extra)}")
        for name in MODALITY_NAMES:
            if not isinstance(mapping[name], ModalityInput):
                raise TypeError(f"{name} must be a ModalityInput")
        return {name: mapping[name] for name in MODALITY_NAMES}

    @staticmethod
    def _availability_from_input(modality: ModalityInput, *, batch_size: int, device: torch.device) -> Tensor:
        if modality.available is not None:
            available = modality.available
            if not isinstance(available, Tensor):
                raise TypeError("available must be a torch.Tensor")
            if available.ndim != 1 or available.shape[0] != batch_size:
                raise ValueError("available must have shape [B]")
            if available.is_floating_point() and not torch.isfinite(available).all():
                raise ValueError("available contains non-finite values")
            return available.to(device=device).bool()
        mask = modality.observed_mask
        if not isinstance(mask, Tensor) or mask.ndim not in (2, 3) or mask.shape[0] != batch_size:
            raise ValueError("observed_mask must have batch dimension [B]")
        mask = mask.to(device=device)
        if mask.is_floating_point() and not torch.isfinite(mask).all():
            raise ValueError("observed_mask contains non-finite values")
        return (mask.detach().float().reshape(batch_size, -1).sum(dim=1) > 0)

    def forward(
        self,
        inputs: IRISSEPInputs | Mapping[str, ModalityInput],
        *,
        modality_available: Optional[Tensor] = None,
        modality_keep_mask: Optional[Tensor] = None,
        apply_missing_modality_dropout: Optional[bool] = None,
        generator: Optional[torch.Generator] = None,
    ) -> ForecastOutput:
        """Run an issue-time forecast from three causal input histories.

        Args:
            inputs: Named ``ModalityInput`` values for magnetic, eruption, and
                particle histories.
            modality_available: Optional explicit ``[B, 3]`` feed-availability
                override.  If omitted it is derived from each input.
            modality_keep_mask: Optional explicit ``[B, 3]`` mask used by a
                trainer for deterministic missing-modality training.  A false
                value removes that expert from fusion for the current batch.
            apply_missing_modality_dropout: When ``None``, use the configured
                probability only in ``model.training`` mode.  Set ``False`` for
                deterministic validation/replay; set ``True`` to force the hook.
            generator: Optional torch RNG generator for reproducible dropout.
        """

        bundle = self._coerce_input_bundle(inputs)
        first_values = bundle[MODALITY_NAMES[0]].values
        if not isinstance(first_values, Tensor) or first_values.ndim != 3:
            raise ValueError("values must have shape [B, T, F]")
        batch_size, sequence_length, _ = first_values.shape
        embeddings: list[Tensor] = []
        for name in MODALITY_NAMES:
            embedding = self.experts[name](bundle[name])
            if embedding.shape[0] != batch_size:
                raise ValueError("all modalities must have the same batch size")
            # The expert already validates temporal dimensions; this check keeps
            # the integration error close to the call site.
            if bundle[name].values.shape[1] != sequence_length:
                raise ValueError("all modalities must have the same sequence length")
            embeddings.append(embedding)
        stacked = torch.stack(embeddings, dim=1)  # [B, 3, E]

        parameter = next(self.parameters())
        if modality_available is None:
            availability = torch.stack(
                [
                    self._availability_from_input(
                        bundle[name], batch_size=batch_size, device=parameter.device
                    )
                    for name in MODALITY_NAMES
                ],
                dim=1,
            )
        else:
            if not isinstance(modality_available, Tensor):
                raise TypeError("modality_available must be a torch.Tensor")
            if modality_available.shape != (batch_size, len(MODALITY_NAMES)):
                raise ValueError(f"modality_available must have shape [B, {len(MODALITY_NAMES)}]")
            if modality_available.is_floating_point() and not torch.isfinite(modality_available).all():
                raise ValueError("modality_available contains non-finite values")
            availability = modality_available.to(device=parameter.device).bool()

        if modality_keep_mask is not None:
            if not isinstance(modality_keep_mask, Tensor):
                raise TypeError("modality_keep_mask must be a torch.Tensor")
            if modality_keep_mask.shape != availability.shape:
                raise ValueError(f"modality_keep_mask must have shape {tuple(availability.shape)}")
            if modality_keep_mask.is_floating_point() and not torch.isfinite(modality_keep_mask).all():
                raise ValueError("modality_keep_mask contains non-finite values")
            keep_mask = modality_keep_mask.to(device=parameter.device).bool()
        else:
            should_drop = self.training if apply_missing_modality_dropout is None else apply_missing_modality_dropout
            if should_drop and self.config.missing_modality_dropout > 0.0:
                keep_mask = sample_modality_keep_mask(
                    availability,
                    self.config.missing_modality_dropout,
                    generator=generator,
                    ensure_one_available=True,
                )
            else:
                keep_mask = torch.ones_like(availability)
        effective_available = availability & keep_mask

        masked_embeddings = stacked * effective_available.to(stacked.dtype).unsqueeze(-1)
        gate_input = torch.cat((masked_embeddings.flatten(start_dim=1), effective_available.to(stacked.dtype)), dim=1)
        gate_logits = self.gate(gate_input)
        # A large finite mask avoids NaNs when a row has no available feed.  We
        # explicitly renormalize after masking, yielding exact zeros for absent
        # modalities and an all-zero fusion vector for an all-missing row.
        masked_gate_logits = gate_logits.masked_fill(~effective_available, -1e4)
        gate_weights = torch.softmax(masked_gate_logits, dim=1)
        gate_weights = gate_weights * effective_available.to(gate_weights.dtype)
        gate_weights = gate_weights / gate_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        fused = (stacked * gate_weights.unsqueeze(-1)).sum(dim=1)
        shared_input = torch.cat((fused, effective_available.to(fused.dtype)), dim=1)
        shared = self.shared(shared_input)

        primary_logit = self.primary_head(shared).squeeze(-1)
        high_energy_logit = self.high_energy_head(shared).squeeze(-1)
        raw_quantiles = self.peak_flux_head(shared)
        # Monotonic quantiles avoid impossible p10 > p50 > p90 orderings while
        # leaving the target's log-pfu units unconstrained in the real line.
        peak_flux_log_quantiles = torch.cat(
            (
                raw_quantiles[:, :1],
                raw_quantiles[:, :1] + F.softplus(raw_quantiles[:, 1:2]),
                raw_quantiles[:, :1]
                + F.softplus(raw_quantiles[:, 1:2])
                + F.softplus(raw_quantiles[:, 2:3]),
            ),
            dim=1,
        )
        return ForecastOutput(
            primary_logit=primary_logit,
            high_energy_logit=high_energy_logit,
            peak_flux_log_quantiles=peak_flux_log_quantiles,
            onset_hazard_logits=self.onset_hazard_head(shared),
            flare_activity_logit=self.flare_activity_head(shared).squeeze(-1),
            cme_activity_logit=self.cme_activity_head(shared).squeeze(-1),
            shared_embedding=shared,
            modality_embeddings=stacked,
            gate_weights=gate_weights,
            modality_available=effective_available,
            quantile_levels=self.config.quantile_levels,
        )


def pinball_loss(prediction: Tensor, target: Tensor, quantiles: Sequence[float]) -> Tensor:
    """Quantile loss helper for log-flux targets; no threshold is selected here."""

    if prediction.ndim != 2 or prediction.shape[1] != len(quantiles):
        raise ValueError("prediction must have shape [B, number_of_quantiles]")
    if target.ndim == 1:
        target = target.unsqueeze(-1)
    if target.shape != (prediction.shape[0], 1):
        raise ValueError("target must have shape [B] or [B, 1]")
    q = torch.as_tensor(tuple(float(value) for value in quantiles), device=prediction.device, dtype=prediction.dtype)
    error = target - prediction
    return torch.maximum(q * error, (q - 1.0) * error).mean()


def compute_task_losses(
    output: ForecastOutput,
    targets: Mapping[str, Tensor],
    *,
    weights: Optional[Mapping[str, float]] = None,
) -> dict[str, Tensor]:
    """Compute transparent per-head losses for an integration trainer.

    This function only validates tensor shapes and computes differentiable
    losses.  It does not choose class weights, thresholds, calibration, or any
    benchmark split; those remain outside this prototype and must be fitted
    under the frozen benchmark contract.
    """

    required = {"primary", "high_energy", "peak_flux_log", "onset_hazard", "flare_activity", "cme_activity"}
    missing = sorted(required - set(targets))
    if missing:
        raise ValueError(f"targets missing keys: {', '.join(missing)}")
    batch_size = output.primary_logit.shape[0]
    losses: MutableMapping[str, Tensor] = {}

    def binary(name: str, logit: Tensor) -> None:
        target = targets[name]
        if target.shape != (batch_size,):
            raise ValueError(f"target {name} must have shape [B]")
        losses[name] = F.binary_cross_entropy_with_logits(
            logit,
            target.to(device=logit.device, dtype=logit.dtype),
        )

    binary("primary", output.primary_logit)
    binary("high_energy", output.high_energy_logit)
    binary("flare_activity", output.flare_activity_logit)
    binary("cme_activity", output.cme_activity_logit)
    losses["peak_flux_log"] = pinball_loss(
        output.peak_flux_log_quantiles,
        targets["peak_flux_log"].to(
            device=output.peak_flux_log_quantiles.device,
            dtype=output.peak_flux_log_quantiles.dtype,
        ),
        output.quantile_levels,
    )
    onset_target = targets["onset_hazard"]
    if onset_target.shape != output.onset_hazard_logits.shape:
        raise ValueError("target onset_hazard must match [B, onset_horizon_bins]")
    losses["onset_hazard"] = F.binary_cross_entropy_with_logits(
        output.onset_hazard_logits,
        onset_target.to(device=output.onset_hazard_logits.device, dtype=output.onset_hazard_logits.dtype),
    )
    if weights is None:
        weights = {}
    total = sum(
        (float(weights.get(name, 1.0)) * value for name, value in losses.items()),
        torch.zeros((), device=output.primary_logit.device),
    )
    losses["total"] = total
    return dict(losses)


__all__ = [
    "CausalConv1d",
    "CausalTemporalExpert",
    "ForecastOutput",
    "IRISSEPConfig",
    "IRISSEPInputs",
    "IRISSEPModel",
    "MODALITY_NAMES",
    "ModalityInput",
    "compute_task_losses",
    "pinball_loss",
    "sample_modality_keep_mask",
]
