# Neo-Spectrum-DiT
#
# Scheduler, forecaster and adaptive controller of ComfyUI-Spectrum-MOSS v0.8.1
# (https://github.com/Rkkss/ComfyUI-Spectrum-MOSS, nodes.py), kept verbatim where it
# does not touch ComfyUI so that Forge Neo makes the same actual/forecast decisions
# as the validated node. One deliberate change: Manual is capped at two forecasts in a
# row under ER-SDE (`_runtime_config`). Based on the method from arXiv:2603.01623.
#
# SPDX-License-Identifier: GPL-3.0-only

"""Model- and webui-independent half of the port: torch and the standard library only."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import logging
import math
from typing import Any, Optional

import torch


logger = logging.getLogger("Spectrum-DiT")

_PREDICT_CHUNK = 1 << 22

# Validated Adaptive policy constants (MOSS-Image v0.6).
_ADAPTIVE_HISTORY_POINTS = 8
_ADAPTIVE_CHEBYSHEV_DEGREE = 4
_ADAPTIVE_RIDGE_LAMBDA = 0.1
_ADAPTIVE_BLEND_WEIGHT = 0.5
_ADAPTIVE_GUARD_HORIZON = 1
_ADAPTIVE_BASELINE_ALPHA = 0.12
_ADAPTIVE_MEDIUM_RATIO = 1.18
_ADAPTIVE_HIGH_RATIO = 1.35
_ADAPTIVE_SAFE_REFRESHES_TO_EXPAND = 2
_ADAPTIVE_REPAIR_ACTUAL_STEPS = 1
_ADAPTIVE_SAMPLE_TOKENS = 384
_ADAPTIVE_HIGH_COOLDOWN_REFRESHES = 1


def _log_debug(enabled: bool, msg: str) -> None:
    if enabled:
        logger.info(msg)


def _log_warning(msg: str) -> None:
    logger.warning(msg)


@dataclass(frozen=True)
class SpectrumConfig:
    """Runtime configuration shared by the Manual and Adaptive nodes.

    Manual mode exposes the core Spectrum predictor/scheduler controls.
    Adaptive mode uses the validated MOSS-Image v0.6 policy: an early detail
    guard plus a local p95/p99 drift controller that adjusts forecast horizon.
    """

    mode: str = "manual"
    warmup_steps: int = 8
    tail_actual_steps: int = 1
    window_size: float = 2.0
    flex_window: float = 0.75
    max_consecutive_forecasts: int = 2  # Manual mode only.
    history_points: int = 8
    chebyshev_degree: int = 4
    ridge_lambda: float = 0.1
    blend_weight: float = 0.5
    cache_device: str = "main_device"
    force_actual_on_control: bool = True

    # Adaptive scheduling controls. Other adaptive constants are fixed above.
    detail_guard_steps: int = 8
    post_guard_horizon_cap: int = 6
    error_tolerance: float = 1.0

    debug: bool = False
    warn_unverified_sampler: bool = True

    # Internal sampler-specific runtime policy; not exposed as node controls.
    # ER-SDE uses synchronized branch lifecycle/decisions in both modes. Adaptive
    # additionally enables the validated H1->H2 controller.
    er_sde_sync: bool = False
    er_sde_h2: bool = False

    @property
    def adaptive(self) -> bool:
        return self.mode == "adaptive"

    def validate(self) -> None:
        if self.mode not in {"manual", "adaptive"}:
            raise ValueError("mode must be 'manual' or 'adaptive'")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if self.tail_actual_steps < 0:
            raise ValueError("tail_actual_steps must be >= 0")
        if self.window_size < 1.0:
            raise ValueError("window_size must be >= 1.0")
        if self.flex_window < 0.0:
            raise ValueError("flex_window must be >= 0.0")
        if self.max_consecutive_forecasts < 0:
            raise ValueError("max_consecutive_forecasts must be >= 0")
        if self.history_points < 2:
            raise ValueError("history_points must be >= 2")
        if self.chebyshev_degree < 1:
            raise ValueError("chebyshev_degree must be >= 1")
        if self.ridge_lambda < 0.0:
            raise ValueError("ridge_lambda must be >= 0")
        if not 0.0 <= self.blend_weight <= 1.0:
            raise ValueError("blend_weight must be in [0, 1]")
        if self.cache_device not in {"main_device", "offload_device", "cpu"}:
            raise ValueError("cache_device must be main_device, offload_device, or cpu")

        if self.adaptive:
            if self.detail_guard_steps < 0:
                raise ValueError("detail_guard_steps must be >= 0")
            if self.post_guard_horizon_cap < 1:
                raise ValueError("post_guard_horizon_cap must be >= 1")
            if self.error_tolerance <= 0.0:
                raise ValueError("error_tolerance must be > 0")

    @property
    def window_growth_cap(self) -> float:
        return max(1.0, self.window_size, float(self.history_points))


def _normalize_step_position(step_index: int, total_steps: int) -> float:
    if total_steps <= 1:
        return 0.0
    coord = 2.0 * float(step_index) / float(total_steps - 1) - 1.0
    return max(-1.0, min(1.0, coord))


def _chebyshev_basis(x: torch.Tensor, degree: int) -> torch.Tensor:
    x = x.to(dtype=torch.float32, device="cpu")
    cols = [torch.ones_like(x)]
    if degree >= 1:
        cols.append(x)
    for _ in range(2, degree + 1):
        cols.append(2.0 * x * cols[-1] - cols[-2])
    return torch.stack(cols[: degree + 1], dim=-1)


@dataclass
class _Anchor:
    coord: float
    feature: torch.Tensor


class HistoryWeightChebyshevForecaster:
    """Ridge-regularized Chebyshev fit represented as weights over anchors.

    Instead of materializing a huge coefficient matrix over the hidden-state
    feature dimension, solve the tiny polynomial system on CPU and turn the
    prediction into a weighted sum of the stored real-step hidden states.
    """

    def __init__(
        self,
        degree: int,
        ridge_lambda: float,
        max_history: int,
        blend_weight: float,
    ) -> None:
        self.degree = int(degree)
        self.ridge_lambda = float(ridge_lambda)
        self.max_history = max(2, int(max_history))
        self.blend_weight = float(min(1.0, max(0.0, blend_weight)))
        self._anchors: list[_Anchor] = []
        self._feature_shape: Optional[torch.Size] = None
        self._feature_dtype: Optional[torch.dtype] = None
        self._storage_device: Optional[torch.device] = None
        self._chol: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self._anchors.clear()
        self._feature_shape = None
        self._feature_dtype = None
        self._storage_device = None
        self._chol = None

    def release_features(self) -> None:
        self._anchors.clear()
        self._chol = None

    def ready(self) -> bool:
        return self._feature_shape is not None and len(self._anchors) >= self.degree + 1

    @property
    def history_size(self) -> int:
        return len(self._anchors)

    @property
    def feature_shape(self) -> Optional[torch.Size]:
        return self._feature_shape

    def update(self, coord: float, feature: torch.Tensor) -> None:
        if self._feature_shape is None:
            self._feature_shape = torch.Size(feature.shape)
            self._feature_dtype = feature.dtype
            self._storage_device = feature.device
        else:
            if tuple(feature.shape) != tuple(self._feature_shape):
                raise ValueError(
                    f"feature shape changed {tuple(self._feature_shape)} -> {tuple(feature.shape)}"
                )
            if feature.dtype != self._feature_dtype:
                raise ValueError(
                    f"feature dtype changed {self._feature_dtype} -> {feature.dtype}"
                )
            if feature.device != self._storage_device:
                raise ValueError(
                    f"feature device changed {self._storage_device} -> {feature.device}"
                )

        self._anchors.append(_Anchor(float(coord), feature))
        if len(self._anchors) > self.max_history:
            self._anchors.pop(0)
        self._chol = None

    def _design(self) -> torch.Tensor:
        coords = torch.tensor([a.coord for a in self._anchors], dtype=torch.float32)
        return _chebyshev_basis(coords, self.degree)

    def _factorize(self) -> torch.Tensor:
        if self._chol is not None:
            return self._chol
        design = self._design()
        gram = design.T @ design
        p = gram.shape[0]
        eye = torch.eye(p, dtype=torch.float32)
        lhs = gram + self.ridge_lambda * eye
        try:
            chol = torch.linalg.cholesky(lhs)
        except RuntimeError:
            scale = max(float(gram.diagonal().abs().mean().item()), 1.0)
            chol = None
            for m in (1e-8, 1e-7, 1e-6, 1e-5):
                try:
                    chol = torch.linalg.cholesky(lhs + scale * m * eye)
                    break
                except RuntimeError:
                    pass
            if chol is None:
                raise
        self._chol = chol
        return chol

    def _spectral_weights(self, coord: float) -> torch.Tensor:
        design = self._design()  # K x P
        chol = self._factorize()  # P x P
        phi = _chebyshev_basis(torch.tensor([float(coord)]), self.degree)  # 1 x P
        solved = torch.cholesky_solve(design.T, chol)  # P x K
        return (phi @ solved).reshape(-1)

    def _linear_weights(self, coord: float) -> torch.Tensor:
        k = len(self._anchors)
        w = torch.zeros(k, dtype=torch.float32)
        if k <= 1:
            if k == 1:
                w[0] = 1.0
            return w
        prev = self._anchors[-2].coord
        last = self._anchors[-1].coord
        spacing = last - prev
        if abs(spacing) <= 1e-12:
            w[-1] = 1.0
            return w
        ratio = (float(coord) - last) / spacing
        w[-2] = -ratio
        w[-1] = 1.0 + ratio
        return w

    def combined_weights(self, coord: float) -> torch.Tensor:
        if self.blend_weight <= 1e-9:
            return self._linear_weights(coord)
        spectral = self._spectral_weights(coord)
        if self.blend_weight >= 1.0 - 1e-9:
            return spectral
        linear = self._linear_weights(coord)
        return self.blend_weight * spectral + (1.0 - self.blend_weight) * linear

    def sampled_token_error_stats(self, coord: float, actual: torch.Tensor, sample_tokens: int) -> dict[str, float]:
        """Measure spatially local forecast error on sampled image tokens.

        MOSS hidden states are shaped [B, image_tokens, D].  A global scalar L2
        can hide a badly drifting hand/eye token region because that region is a
        tiny fraction of the image.  Here we predict only a deterministic set of
        spatial tokens and compute a relative L2 per token, then report tail
        percentiles (p95/p99).
        """
        if not self.ready():
            raise RuntimeError("forecaster is not ready")
        if actual.ndim != 3:
            raise ValueError(f"expected [B,T,D] feature, got shape={tuple(actual.shape)}")

        weights = self.combined_weights(coord).tolist()
        b, tokens, dim = map(int, actual.shape)
        if tokens <= 0 or dim <= 0:
            return {"median": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}

        count = min(tokens, max(32, int(sample_tokens)))
        device = actual.device
        if count >= tokens:
            tidx = torch.arange(tokens, device=device, dtype=torch.int64)
        elif count <= 1:
            tidx = torch.zeros(1, device=device, dtype=torch.int64)
        else:
            tidx = torch.arange(count, device=device, dtype=torch.int64)
            tidx = (tidx * (tokens - 1)) // (count - 1)
            tidx.clamp_(0, tokens - 1)

        target = actual.index_select(1, tidx).to(torch.float32)
        pred = torch.zeros_like(target, dtype=torch.float32)
        for weight, anchor in zip(weights, self._anchors):
            feat = anchor.feature
            if feat.ndim != 3 or tuple(feat.shape) != tuple(actual.shape):
                return {"median": float("inf"), "p95": float("inf"), "p99": float("inf"), "mean": float("inf")}
            sampled = feat.to(device=device).index_select(1, tidx).to(torch.float32)
            if weight != 0.0:
                pred.add_(sampled, alpha=float(weight))

        diff_norm = torch.linalg.vector_norm(pred - target, dim=-1)
        target_norm = torch.linalg.vector_norm(target, dim=-1).clamp_min_(1e-6)
        rel = (diff_norm / target_norm).reshape(-1)
        rel = torch.nan_to_num(rel, nan=float("inf"), posinf=float("inf"), neginf=float("inf"))
        finite = torch.isfinite(rel)
        if not bool(finite.any()):
            return {"median": float("inf"), "p95": float("inf"), "p99": float("inf"), "mean": float("inf")}
        vals = rel[finite].detach().to("cpu", dtype=torch.float32)
        q = torch.quantile(vals, torch.tensor([0.50, 0.95, 0.99], dtype=torch.float32))
        return {
            "median": float(q[0].item()),
            "p95": float(q[1].item()),
            "p99": float(q[2].item()),
            "mean": float(vals.mean().item()),
        }

    def predict(self, coord: float) -> torch.Tensor:
        if not self.ready():
            raise RuntimeError("forecaster is not ready")
        assert self._feature_shape is not None
        assert self._feature_dtype is not None
        assert self._storage_device is not None

        weights = self.combined_weights(coord).tolist()
        out = torch.empty(
            self._feature_shape,
            dtype=self._feature_dtype,
            device=self._storage_device,
        )
        out_flat = out.view(-1)
        flats = [a.feature.contiguous().view(-1) for a in self._anchors]
        numel = out_flat.numel()

        finfo = torch.finfo(self._feature_dtype) if self._feature_dtype.is_floating_point else None
        for start in range(0, numel, _PREDICT_CHUNK):
            stop = min(start + _PREDICT_CHUNK, numel)
            acc = torch.zeros(stop - start, dtype=torch.float32, device=out.device)
            for weight, flat in zip(weights, flats):
                if weight != 0.0:
                    acc.add_(flat[start:stop].to(torch.float32), alpha=float(weight))
            if finfo is not None:
                torch.nan_to_num_(acc, nan=0.0, posinf=float(finfo.max), neginf=float(finfo.min))
                acc.clamp_(min=float(finfo.min), max=float(finfo.max))
            out_flat[start:stop] = acc.to(self._feature_dtype)
        return out


def _initial_adaptive_horizon(config: SpectrumConfig) -> int:
    # Native ER-SDE starts conservatively at H1 and must earn H2 through the
    # existing error controller. Other samplers keep the legacy Adaptive policy.
    if config.er_sde_h2:
        return 1
    return max(1, config.post_guard_horizon_cap)


@dataclass
class SpectrumBranchState:
    config: SpectrumConfig
    forecaster: HistoryWeightChebyshevForecaster = field(init=False)
    consecutive_forecasts: int = 0
    current_window: float = 2.0
    actual_count: int = 0
    forecast_count: int = 0

    # Adaptive local-drift controller.
    baseline_p95: Optional[float] = None
    baseline_p99: Optional[float] = None
    baseline_observations: int = 0
    adaptive_horizon: int = 1
    safe_refreshes: int = 0
    repair_actual_steps_remaining: int = 0
    last_p95: Optional[float] = None
    last_p99: Optional[float] = None
    last_local_ratio: Optional[float] = None
    last_validation_tier: str = "learning"
    last_streak: int = 0
    horizon_upgrades: int = 0
    horizon_downgrades: int = 0
    repair_events: int = 0

    # Temporary guard after a local anomaly, counted in validation refreshes.
    cooldown_refreshes_remaining: int = 0
    cooldown_horizon_cap: Optional[int] = None

    def __post_init__(self) -> None:
        self.current_window = float(self.config.window_size)
        self.adaptive_horizon = (
            _initial_adaptive_horizon(self.config)
            if self.config.adaptive
            else max(1, self.config.max_consecutive_forecasts)
        )
        self.forecaster = HistoryWeightChebyshevForecaster(
            degree=self.config.chebyshev_degree,
            ridge_lambda=self.config.ridge_lambda,
            max_history=self.config.history_points,
            blend_weight=self.config.blend_weight,
        )

    def reset(self) -> None:
        self.forecaster.reset()
        self.consecutive_forecasts = 0
        self.current_window = float(self.config.window_size)
        self.actual_count = 0
        self.forecast_count = 0
        self.baseline_p95 = None
        self.baseline_p99 = None
        self.baseline_observations = 0
        self.adaptive_horizon = (
            _initial_adaptive_horizon(self.config)
            if self.config.adaptive
            else max(1, self.config.max_consecutive_forecasts)
        )
        self.safe_refreshes = 0
        self.repair_actual_steps_remaining = 0
        self.last_p95 = None
        self.last_p99 = None
        self.last_local_ratio = None
        self.last_validation_tier = "learning"
        self.last_streak = 0
        self.horizon_upgrades = 0
        self.horizon_downgrades = 0
        self.repair_events = 0
        self.cooldown_refreshes_remaining = 0
        self.cooldown_horizon_cap = None


@dataclass
class SpectrumRootState:
    config: SpectrumConfig
    branches: dict[Any, SpectrumBranchState] = field(default_factory=dict)
    last_sigmas_id: Optional[int] = None
    last_total_steps: Optional[int] = None
    last_step_index: Optional[int] = None
    completed_branches: set[Any] = field(default_factory=set)
    summary_logged: bool = False

    # Forge Neo: the sampler is known when the wrapper is installed, so ER-SDE is a
    # flag here instead of the node's context variable.
    er_sde: bool = False
    # Forge Neo: index of this run's first sigma in `sampling_sigmas`, which holds the
    # full schedule even when img2img or the hires pass samples only its tail.
    step_offset: Optional[int] = None

    # Active patch signature for each CFG/SLG cond branch while Adaptive ER-SDE
    # synchronization is in use. Partial SLG windows can replace a branch key
    # mid-run. Historical branches stay in ``branches`` for statistics only.
    active_branch_keys: dict[Any, Any] = field(default_factory=dict)

    # One global Adaptive ER-SDE A/F decision is cached per solver step so
    # CFG/SLG branches cannot choose different modes.
    global_decision_step: Optional[int] = None
    global_decision_actual: Optional[bool] = None
    global_decision_reason: Optional[str] = None

    def reset_run(self) -> None:
        self.branches.clear()
        self.last_sigmas_id = None
        self.last_total_steps = None
        self.last_step_index = None
        self.completed_branches.clear()
        self.summary_logged = False
        self.active_branch_keys.clear()
        self.global_decision_step = None
        self.global_decision_actual = None
        self.global_decision_reason = None
        self.step_offset = None

    def branch(self, key: Any) -> SpectrumBranchState:
        if key not in self.branches:
            self.branches[key] = SpectrumBranchState(_runtime_config(self.config, self.er_sde))
        return self.branches[key]

    def activate_er_sde_branch(self, cond_key: Any, branch_key: Any, debug: bool = False) -> None:
        """Make ``branch_key`` the sole active ER-SDE branch for ``cond_key``.

        SkipLayerGuidanceDiTSimple changes its DiT patch signature when a partial
        guidance window starts/ends. That intentionally creates a fresh history;
        the superseded state must be retired from the global scheduler.
        """
        previous = self.active_branch_keys.get(cond_key)
        if previous == branch_key:
            return
        if previous is not None:
            old_branch = self.branches.get(previous)
            if old_branch is not None:
                old_branch.forecaster.release_features()
            _log_debug(
                debug,
                f"ER-SDE branch transition cond={cond_key}: "
                f"retire={previous} activate={branch_key}",
            )
        self.active_branch_keys[cond_key] = branch_key

    def active_branch_items(self) -> list[tuple[Any, SpectrumBranchState]]:
        items: list[tuple[Any, SpectrumBranchState]] = []
        for key in self.active_branch_keys.values():
            branch = self.branches.get(key)
            if branch is not None:
                items.append((key, branch))
        return items

    def active_branch_key_set(self) -> set[Any]:
        return {key for key in self.active_branch_keys.values() if key in self.branches}


def _find_step_index(sigmas: torch.Tensor, timestep: torch.Tensor) -> int:
    """Map the model timestep back to the sampler sigma index.

    Native MOSS is ModelType.FLOW, whose ModelSamplingDiscreteFlow passes
    ``timestep = sigma * multiplier`` (normally multiplier=1000) into the
    diffusion model.  Some other flow wrappers pass sigma directly.  Try both
    representations and keep the one that actually lands on the sigma schedule.
    """
    if not isinstance(sigmas, torch.Tensor) or sigmas.numel() < 1:
        return -1
    if not isinstance(timestep, torch.Tensor) or timestep.numel() < 1:
        return -1

    raw = timestep.reshape(-1)[0].float()
    s = sigmas.to(device=raw.device, dtype=torch.float32)
    candidates = [raw, raw / 1000.0]

    best_index = -1
    best_error = float("inf")
    scale = max(float(s.abs().max().item()), 1e-6)
    for target in candidates:
        diffs = (s - target).abs()
        idx = int(diffs.argmin().item())
        err = float(diffs[idx].item()) / scale
        if err < best_error:
            best_error = err
            best_index = idx

    return best_index


def _patch_signature(transformer_options: dict) -> tuple[str, ...]:
    """Separate histories when DiT replacement patches (e.g. SLG) change."""
    try:
        dit = transformer_options.get("patches_replace", {}).get("dit", {})
        if not isinstance(dit, dict) or not dit:
            return ()
        return tuple(sorted(repr(key) for key in dit.keys()))
    except Exception:
        return ()


def _runtime_config(config: SpectrumConfig, er_sde: bool) -> SpectrumConfig:
    """Apply sampler-specific runtime policy without changing the public node UI."""
    if not er_sde:
        return config
    if config.adaptive:
        return replace(
            config,
            window_size=2.0,
            flex_window=0.0,
            post_guard_horizon_cap=min(config.post_guard_horizon_cap, 2),
            history_points=_ADAPTIVE_HISTORY_POINTS,
            chebyshev_degree=_ADAPTIVE_CHEBYSHEV_DEGREE,
            ridge_lambda=_ADAPTIVE_RIDGE_LAMBDA,
            blend_weight=_ADAPTIVE_BLEND_WEIGHT,
            er_sde_sync=True,
            er_sde_h2=True,
        )

    # Manual stays manual under ER-SDE: keep every user-selected predictor and
    # cadence control. Only add solver-safe branch synchronization/lifecycle.
    #
    # Neo-Spectrum-DiT departs from the node here: Manual is capped at two forecasts
    # in a row. The bridge extrapolates F1 and F2 from real anchors; from F3 on it holds
    # the last real result while ER SDE keeps adding noise, which ghosts (seen live on
    # 2026-10-22 with eight in a row).
    return replace(
        config,
        er_sde_sync=True,
        er_sde_h2=False,
        max_consecutive_forecasts=min(config.max_consecutive_forecasts, 2),
    )


def _adaptive_phase_horizon_cap(config: SpectrumConfig, step_index: int) -> int:
    """Hard horizon cap for the validated MOSS-Image adaptive policy.

    Warmup is handled separately. The following detail guard is intentionally
    conservative, then the scheduler may expand up to the post-guard cap.
    """
    guard_end = config.warmup_steps + max(0, config.detail_guard_steps)
    if step_index < guard_end:
        return _ADAPTIVE_GUARD_HORIZON
    return max(1, config.post_guard_horizon_cap)


def _choose_actual(
    state: SpectrumBranchState,
    step_index: int,
    total_steps: int,
    control_present: bool,
    config: SpectrumConfig,
) -> tuple[bool, str]:
    if step_index < 0:
        return True, "unknown_step"
    if step_index < config.warmup_steps:
        return True, "warmup"
    if config.tail_actual_steps > 0 and step_index >= max(0, total_steps - config.tail_actual_steps):
        return True, "tail"
    if control_present and config.force_actual_on_control:
        return True, "control"
    if not state.forecaster.ready():
        return True, "insufficient_history"

    if config.adaptive:
        if state.repair_actual_steps_remaining > 0:
            return True, "local_repair"

        # Adaptive mode interprets window_size as a forecast budget before the
        # next real refresh. Local drift and phase protection can only reduce it.
        window_cap = max(1, int(math.floor(state.current_window)))
        phase_cap = _adaptive_phase_horizon_cap(config, step_index)
        safety_cap = min(max(1, state.adaptive_horizon), phase_cap)
        if state.cooldown_refreshes_remaining > 0 and state.cooldown_horizon_cap is not None:
            safety_cap = min(safety_cap, max(1, state.cooldown_horizon_cap))
        effective_cap = min(window_cap, safety_cap)
        if state.consecutive_forecasts >= effective_cap:
            return True, "window" if window_cap <= safety_cap else "local_guard"
        return False, "forecast"

    # Manual mode uses the original Spectrum-style expanding-window scheduler.
    if state.consecutive_forecasts >= config.max_consecutive_forecasts:
        return True, "horizon_refresh"

    interval = max(1, int(math.floor(state.current_window)))
    if (state.consecutive_forecasts + 1) % interval == 0:
        return True, "window"
    return False, "forecast"


def _choose_actual_global_er_sde_adaptive(
    root: SpectrumRootState,
    step_index: int,
    total_steps: int,
    control_present: bool,
    config: SpectrumConfig,
) -> tuple[bool, str]:
    """Choose one conservative A/F mode for every Adaptive ER-SDE branch.

    Per-branch error estimation remains independent, but the least-permissive
    active CFG/SLG branch owns the shared solver-step decision.
    """
    if step_index < 0:
        return True, "unknown_step"
    if step_index < config.warmup_steps:
        return True, "warmup"
    if config.tail_actual_steps > 0 and step_index >= max(0, total_steps - config.tail_actual_steps):
        return True, "tail"
    if control_present and config.force_actual_on_control:
        return True, "control"

    # Only currently active branch signatures participate. A partial SLG window
    # can leave historical branch states in root.branches; including those here
    # makes stale error/readiness state permanently throttle the shared cadence.
    branches = [b for _, b in root.active_branch_items()]
    if not branches or any(not b.forecaster.ready() for b in branches):
        return True, "insufficient_history"
    if any(b.repair_actual_steps_remaining > 0 for b in branches):
        return True, "local_repair"

    phase_cap = _adaptive_phase_horizon_cap(config, step_index)
    effective_caps: list[int] = []
    window_caps: list[int] = []
    safety_caps: list[int] = []
    for b in branches:
        window_cap = max(1, int(math.floor(b.current_window)))
        safety_cap = min(max(1, b.adaptive_horizon), phase_cap)
        if b.cooldown_refreshes_remaining > 0 and b.cooldown_horizon_cap is not None:
            safety_cap = min(safety_cap, max(1, b.cooldown_horizon_cap))
        window_caps.append(window_cap)
        safety_caps.append(safety_cap)
        effective_caps.append(min(window_cap, safety_cap))

    # Conservative shared cap: the least permissive branch owns the step.
    effective_cap = min(effective_caps)
    max_streak = max(b.consecutive_forecasts for b in branches)
    if max_streak >= effective_cap:
        window_cap = min(window_caps)
        safety_cap = min(safety_caps)
        return True, "window" if window_cap <= safety_cap else "local_guard"
    return False, "forecast"


def _choose_actual_global_er_sde_manual(
    root: SpectrumRootState,
    step_index: int,
    total_steps: int,
    control_present: bool,
    config: SpectrumConfig,
) -> tuple[bool, str]:
    """Choose one ER-SDE A/F mode while preserving Manual cadence controls.

    Manual does not use the Adaptive error controller. The user-selected warmup,
    tail, window/flex growth and max-consecutive-forecast limits remain authoritative;
    the least-ready active CFG/SLG branch only synchronizes the shared solver step.
    """
    if step_index < 0:
        return True, "unknown_step"
    if step_index < config.warmup_steps:
        return True, "warmup"
    if config.tail_actual_steps > 0 and step_index >= max(0, total_steps - config.tail_actual_steps):
        return True, "tail"
    if control_present and config.force_actual_on_control:
        return True, "control"

    branches = [b for _, b in root.active_branch_items()]
    if not branches or any(not b.forecaster.ready() for b in branches):
        return True, "insufficient_history"

    if config.max_consecutive_forecasts <= 0:
        return True, "horizon_refresh"
    if any(b.consecutive_forecasts >= config.max_consecutive_forecasts for b in branches):
        return True, "horizon_refresh"

    # Preserve the original Manual expanding-window scheduler. If active branch
    # histories differ around a partial SLG transition, the most conservative
    # branch wins the shared ER-SDE decision.
    intervals = [max(1, int(math.floor(b.current_window))) for b in branches]
    if any((b.consecutive_forecasts + 1) % interval == 0 for b, interval in zip(branches, intervals)):
        return True, "window"
    return False, "forecast"

def _note_decision(state: SpectrumBranchState, actual: bool, reason: str, config: SpectrumConfig) -> None:
    if actual:
        state.consecutive_forecasts = 0
        state.actual_count += 1
        if reason == "window":
            state.current_window = min(
                state.current_window + config.flex_window,
                config.window_growth_cap,
            )
        if reason == "local_repair":
            state.repair_actual_steps_remaining = max(0, state.repair_actual_steps_remaining - 1)
    else:
        state.consecutive_forecasts += 1
        state.forecast_count += 1


def _adaptive_validate_and_update(
    config: SpectrumConfig,
    branch: SpectrumBranchState,
    coord: float,
    actual_feature: Optional[torch.Tensor],
    allow_action: bool,
    allow_baseline_update: bool,
    preceding_forecast_streak: int,
) -> None:
    """Update the local p95/p99 drift controller after an actual refresh."""
    if not config.adaptive or actual_feature is None or not branch.forecaster.ready():
        return
    try:
        stats = branch.forecaster.sampled_token_error_stats(
            coord, actual_feature, _ADAPTIVE_SAMPLE_TOKENS
        )
    except Exception as exc:
        _log_debug(config.debug, f"local validation failed ({type(exc).__name__}: {exc})")
        return

    p95 = stats["p95"]
    p99 = stats["p99"]
    branch.last_p95 = p95
    branch.last_p99 = p99
    branch.last_streak = int(preceding_forecast_streak)

    baseline_eligible = allow_baseline_update and preceding_forecast_streak <= 1
    if branch.baseline_p95 is None or branch.baseline_p99 is None:
        branch.baseline_p95 = p95
        branch.baseline_p99 = p99
        branch.baseline_observations = 1
        ratio = 1.0
        tier = "learning"
    else:
        r95 = p95 / max(branch.baseline_p95, 1e-8)
        r99 = p99 / max(branch.baseline_p99, 1e-8)
        ratio = max(r95, r99)
        medium_threshold = _ADAPTIVE_MEDIUM_RATIO * config.error_tolerance
        high_threshold = _ADAPTIVE_HIGH_RATIO * config.error_tolerance
        if branch.baseline_observations < 3:
            tier = "learning"
        elif ratio >= high_threshold:
            tier = "high"
        elif ratio >= medium_threshold:
            tier = "medium"
        else:
            tier = "normal"

        # Repair/tail observations do not update the baseline. A very clean
        # repair sample can otherwise make later normal refreshes look anomalous.
        if baseline_eligible and tier in {"learning", "normal"}:
            a = _ADAPTIVE_BASELINE_ALPHA
            branch.baseline_p95 = (1.0 - a) * branch.baseline_p95 + a * p95
            branch.baseline_p99 = (1.0 - a) * branch.baseline_p99 + a * p99
            branch.baseline_observations += 1

    branch.last_local_ratio = ratio
    branch.last_validation_tier = tier
    horizon_before = branch.adaptive_horizon
    repair_before = branch.repair_actual_steps_remaining

    if allow_action and branch.baseline_observations >= 3:
        if tier == "high":
            observed_cap = max(1, preceding_forecast_streak - 1)
            branch.adaptive_horizon = max(
                1,
                min(branch.adaptive_horizon - 1, observed_cap),
            )
            branch.safe_refreshes = 0
            branch.current_window = float(config.window_size)
            branch.repair_actual_steps_remaining = max(
                branch.repair_actual_steps_remaining, _ADAPTIVE_REPAIR_ACTUAL_STEPS
            )
            if repair_before <= 0 and branch.repair_actual_steps_remaining > 0:
                branch.repair_events += 1
            branch.cooldown_refreshes_remaining = max(
                branch.cooldown_refreshes_remaining, _ADAPTIVE_HIGH_COOLDOWN_REFRESHES
            )
            branch.cooldown_horizon_cap = max(
                1, min(2, branch.adaptive_horizon)
            )
        elif tier == "medium":
            if preceding_forecast_streak >= 2:
                branch.adaptive_horizon = max(1, branch.adaptive_horizon - 1)
            branch.safe_refreshes = 0
            branch.cooldown_horizon_cap = max(1, min(3, branch.adaptive_horizon))
        else:
            if branch.cooldown_refreshes_remaining > 0:
                branch.cooldown_refreshes_remaining -= 1
                if branch.cooldown_refreshes_remaining <= 0:
                    branch.cooldown_horizon_cap = None
            elif preceding_forecast_streak >= max(1, branch.adaptive_horizon - 1):
                branch.safe_refreshes += 1
                if branch.safe_refreshes >= _ADAPTIVE_SAFE_REFRESHES_TO_EXPAND:
                    max_h = max(1, config.post_guard_horizon_cap)
                    branch.adaptive_horizon = min(max_h, branch.adaptive_horizon + 1)
                    branch.safe_refreshes = 0

    if branch.adaptive_horizon > horizon_before:
        branch.horizon_upgrades += 1
        _log_debug(
            config.debug,
            f"adaptive horizon upgrade {horizon_before}->{branch.adaptive_horizon} "
            f"tier={tier} ratio={ratio:.3f}",
        )
    elif branch.adaptive_horizon < horizon_before:
        branch.horizon_downgrades += 1
        _log_debug(
            config.debug,
            f"adaptive horizon downgrade {horizon_before}->{branch.adaptive_horizon} "
            f"tier={tier} ratio={ratio:.3f}",
        )

    _log_debug(
        config.debug,
        f"local tier={tier} p95={p95:.5f} p99={p99:.5f} "
        f"base95={branch.baseline_p95 if branch.baseline_p95 is not None else float('nan'):.5f} "
        f"base99={branch.baseline_p99 if branch.baseline_p99 is not None else float('nan'):.5f} "
        f"ratio={ratio:.3f} tol={config.error_tolerance:.2f} streak={preceding_forecast_streak} "
        f"horizon={branch.adaptive_horizon} repair={branch.repair_actual_steps_remaining} "
        f"cooldown={branch.cooldown_refreshes_remaining} "
        f"coolcap={branch.cooldown_horizon_cap if branch.cooldown_horizon_cap is not None else 'none'}",
    )

def _adaptive_phase_name(config: SpectrumConfig, step_index: int, total_steps: int) -> str:
    if step_index < config.warmup_steps:
        return "warmup"
    tail_start = max(0, total_steps - config.tail_actual_steps) if config.tail_actual_steps > 0 else total_steps
    if step_index >= tail_start:
        return "tail"
    guard_end = config.warmup_steps + max(0, config.detail_guard_steps)
    if step_index < guard_end:
        return "guard"
    return "post_guard"
