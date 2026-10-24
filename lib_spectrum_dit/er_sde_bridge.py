# Neo-Spectrum-DiT
#
# ER-SDE solver-space bridge of ComfyUI-Spectrum-MOSS v0.8.0
# (https://github.com/Rkkss/ComfyUI-Spectrum-MOSS, er_sde_bridge.py), ported to Forge Neo.
#
# SPDX-License-Identifier: GPL-3.0-only

"""ER-SDE solver-space bridge.

Spectrum still decides which transformer calls are actual and which are forecast. The
bridge leaves ER-SDE's RNG and noise untouched, records the exact guided denoised output
on actual steps, and replaces forecast observations with a causal two-anchor dense output
in ER-SDE's lambda coordinates. Mixed actual/forecast CFG branches (for example at a
partial SLG boundary) never reach ER-SDE raw; they use the same clean-anchor dense output.

What changed for Forge Neo: the node copies the ComfyUI sampler and swaps in a
`noise_scaler` only to *capture* the lambda coordinates while ER-SDE runs. Forge Neo's
`sample_er_sde` computes them from the sigmas before its first step
(`offset_first_sigma_for_snr`, then `sigma_to_half_log_snr(...).neg().exp()`), so
`set_schedule` computes the same values up front and the sampler is never touched. The
guided denoised result arrives through a `sampler_post_cfg_function`, the Forge
counterpart of the node's `PREDICT_NOISE` wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Callable, Optional

import torch


logger = logging.getLogger("Spectrum-DiT")

SUPPORTED_ER_SDE_SAMPLER = "sample_er_sde"
VERIFIED_SAMPLERS = frozenset({"sample_euler", "sample_er_sde"})


def _debug(enabled: bool, message: str) -> None:
    if enabled:
        logger.info(message)


def _warning(message: str) -> None:
    logger.warning(message)


@dataclass(slots=True)
class _DenoisedAnchor:
    step_id: int
    value: torch.Tensor


@dataclass(slots=True)
class _StepDecision:
    step_id: int
    total_steps: int
    modes: list[bool]  # True = actual, False = forecast


class ERSDESolverBridge:
    """Solver-space bridge for Forge Neo's `sample_er_sde`.

    `lambdas_fn(sigmas) -> lambdas` turns the schedule this run samples into ER-SDE's
    lambda coordinates; Forge's script binds it to the model's predictor.
    """

    def __init__(self, *, debug: bool, enabled: bool, lambdas_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None) -> None:
        self.debug = bool(debug)
        self.bridge_enabled = bool(enabled)
        self.bypass_spectrum = False
        self._forced_bypass = False
        self._lambdas_fn = lambdas_fn
        self._solver_coordinates: dict[int, torch.Tensor] = {}
        self._anchors: list[_DenoisedAnchor] = []
        self._decisions: dict[int, _StepDecision] = {}
        self._latest_decision_step: Optional[int] = None
        self._totals: dict[int, int] = {}
        self.actual_observations = 0
        self.forecast_observations = 0
        self.hold_predictions = 0
        self.lambda_predictions = 0
        self.bridge_fallbacks = 0
        self.mixed_observations = 0
        self.mixed_dense_predictions = 0
        self.mixed_raw_fallbacks = 0

    def reset_runtime(self) -> None:
        self._solver_coordinates.clear()
        self._anchors.clear()
        self._decisions.clear()
        self._latest_decision_step = None
        self._totals.clear()
        self.actual_observations = 0
        self.forecast_observations = 0
        self.hold_predictions = 0
        self.lambda_predictions = 0
        self.bridge_fallbacks = 0
        self.mixed_observations = 0
        self.mixed_dense_predictions = 0
        self.mixed_raw_fallbacks = 0
        self.bypass_spectrum = self._forced_bypass

    def force_bypass(self, message: str) -> None:
        """A structural conflict: every step of every run in this pass stays actual."""
        self.bridge_enabled = False
        self._forced_bypass = True
        self.bypass_spectrum = True
        _warning(message)

    def set_schedule(self, sigmas: torch.Tensor) -> None:
        """Start a run: `sigmas` is the schedule the sampler runs, step 0 first.

        Called by the Spectrum wrapper when it detects a new run. A schedule whose lambdas
        cannot be computed makes the whole run fall back to actual steps, the same
        fail-closed rule the node applies to a structural bridge conflict.
        """
        self.reset_runtime()
        if not self.bridge_enabled:
            return
        try:
            lambdas = self._lambdas_fn(sigmas)
            if lambdas.numel() != sigmas.numel() or not bool(torch.isfinite(lambdas[:-1]).all()):
                raise ValueError(f"{lambdas.numel()} lambdas for {sigmas.numel()} sigmas")
        except Exception as exc:
            self.bypass_spectrum = True
            _warning(f"ER-SDE lambda coordinates unavailable ({type(exc).__name__}: {exc}); using actual steps.")
            return
        for i in range(int(lambdas.numel())):
            self._solver_coordinates[i] = lambdas[i].detach().clone()

    def summary(self) -> None:
        if not (self.debug and self.bridge_enabled):
            return
        retained_bytes = sum(a.value.numel() * a.value.element_size() for a in self._anchors)
        _debug(
            True,
            "ER-SDE bridge summary: "
            f"actual_observations={self.actual_observations} "
            f"forecast_observations={self.forecast_observations} "
            f"latest_hold={self.hold_predictions} "
            f"lambda_dense={self.lambda_predictions} "
            f"mixed={self.mixed_observations} "
            f"mixed_dense={self.mixed_dense_predictions} "
            f"mixed_raw_fallback={self.mixed_raw_fallbacks} "
            f"fallbacks={self.bridge_fallbacks} "
            f"retained_anchor_bytes={retained_bytes}",
        )

    def release(self) -> None:
        self._anchors.clear()
        self._decisions.clear()
        self._totals.clear()
        self._latest_decision_step = None

    def note_branch_decision(
        self,
        step_id: int,
        total_steps: int,
        actual: bool,
        _reason: str,
        _branch_key: Any,
    ) -> None:
        if not self.bridge_enabled or step_id < 0:
            return
        self._totals[step_id] = int(total_steps)
        item = self._decisions.get(step_id)
        if item is None:
            item = _StepDecision(step_id=step_id, total_steps=total_steps, modes=[])
            self._decisions[step_id] = item
        item.modes.append(bool(actual))
        self._latest_decision_step = step_id

    def _consume_mode(self) -> tuple[int, Optional[bool]] | None:
        step_id = self._latest_decision_step
        if step_id is None:
            return None
        item = self._decisions.pop(step_id, None)
        self._latest_decision_step = None
        if item is None or not item.modes:
            return None
        if any(mode != item.modes[0] for mode in item.modes[1:]):
            self.mixed_observations += 1
            _debug(
                self.debug,
                f"ER-SDE bridge step {step_id + 1}: mixed actual/forecast branches -> dense output",
            )
            return step_id, None
        return step_id, item.modes[0]

    @staticmethod
    def _same_tensor_contract(a: torch.Tensor, b: torch.Tensor) -> bool:
        return a.shape == b.shape and a.device == b.device and a.dtype == b.dtype

    def _observe_actual(self, step_id: int, denoised: torch.Tensor) -> None:
        retained = denoised.detach().clone(memory_format=torch.contiguous_format)
        self._anchors.append(_DenoisedAnchor(step_id=int(step_id), value=retained))
        if len(self._anchors) > 2:
            del self._anchors[:-2]
        self.actual_observations += 1
        _debug(
            self.debug,
            f"ER-SDE bridge step {step_id + 1}: actual solver anchor; "
            f"anchors={[a.step_id + 1 for a in self._anchors]}",
        )

    def _latest_hold(self, step_id: int, raw: torch.Tensor, mode: str) -> torch.Tensor:
        latest = self._anchors[-1]
        if not self._same_tensor_contract(latest.value, raw):
            self.bridge_fallbacks += 1
            _warning(f"ER-SDE bridge tensor contract changed at step {step_id + 1}.")
            return raw
        self.hold_predictions += 1
        _debug(
            self.debug,
            f"ER-SDE bridge step {step_id + 1}: mode={mode} anchor={latest.step_id + 1}",
        )
        return latest.value.clone(memory_format=torch.contiguous_format)

    def _predict_dense(self, step_id: int, raw: torch.Tensor, *, mixed: bool = False) -> torch.Tensor:
        if not mixed:
            self.forecast_observations += 1
        if not self._anchors:
            self.bridge_fallbacks += 1
            if mixed:
                self.mixed_raw_fallbacks += 1
            _warning(f"ER-SDE bridge has no clean anchor at step {step_id + 1}.")
            return raw

        latest = self._anchors[-1]
        mode_prefix = "mixed_" if mixed else ""
        if latest.step_id != step_id - 1:
            return self._latest_hold(step_id, raw, mode_prefix + "latest_actual_hold_nonadjacent")
        if len(self._anchors) < 2:
            return self._latest_hold(step_id, raw, mode_prefix + "latest_actual_hold")

        previous = self._anchors[-2]
        if (
            not self._same_tensor_contract(previous.value, latest.value)
            or not self._same_tensor_contract(latest.value, raw)
        ):
            if mixed and not self._same_tensor_contract(latest.value, raw):
                self.mixed_raw_fallbacks += 1
            return self._latest_hold(step_id, raw, mode_prefix + "latest_actual_hold_contract_guard")

        previous_coord = self._solver_coordinates.get(previous.step_id)
        latest_coord = self._solver_coordinates.get(latest.step_id)
        target_coord = self._solver_coordinates.get(step_id)
        if previous_coord is None or latest_coord is None or target_coord is None:
            return self._latest_hold(step_id, raw, mode_prefix + "latest_actual_hold_missing_lambda")

        denominator = latest_coord - previous_coord
        advance = target_coord - latest_coord
        alpha = advance / denominator
        scale = torch.maximum(
            torch.maximum(previous_coord.abs(), latest_coord.abs()),
            torch.ones_like(latest_coord),
        )
        valid = (
            torch.isfinite(alpha)
            & torch.isfinite(denominator)
            & (denominator.abs() > 1e-12 * scale)
            & (denominator * advance >= 0.0)
            & (alpha >= -1e-6)
            & (alpha <= 1.0 + 1e-6)
        )
        weight = torch.where(valid, alpha.clamp(0.0, 1.0), torch.zeros_like(alpha))
        weight = weight.to(device=latest.value.device, dtype=latest.value.dtype)
        predicted = torch.lerp(latest.value, previous.value, -weight)
        self.lambda_predictions += 1
        if mixed:
            self.mixed_dense_predictions += 1

        mode_name = "mixed_lambda_bounded_extrapolation" if mixed else "lambda_bounded_extrapolation"
        _debug(
            self.debug,
            f"ER-SDE bridge step {step_id + 1}: mode={mode_name} "
            f"anchors=({previous.step_id + 1},{latest.step_id + 1})",
        )
        return predicted

    def consume_guided_denoised(self, denoised: torch.Tensor) -> torch.Tensor:
        """Bridge the guided denoised result before ER-SDE sees it."""
        if not self.bridge_enabled or self.bypass_spectrum or not torch.is_tensor(denoised):
            return denoised
        decision = self._consume_mode()
        if decision is None:
            return denoised
        step_id, actual = decision
        total_steps = self._totals.pop(step_id, None)

        if actual is True:
            self._observe_actual(step_id, denoised)
            bridged = denoised
        elif actual is False:
            bridged = self._predict_dense(step_id, denoised)
        else:
            bridged = self._predict_dense(step_id, denoised, mixed=True)

        # The run's last solver step: nothing is forecast after it, so report and let the
        # anchors go here, after this step's observation rather than before it.
        if total_steps is not None and step_id >= total_steps - 1:
            self.summary()
            self._anchors.clear()
        return bridged

    def post_cfg_function(self, args: dict) -> torch.Tensor:
        """`UnetPatcher.set_model_sampler_post_cfg_function` entry point."""
        return self.consume_guided_denoised(args["denoised"])
