# Neo-Spectrum-DiT
#
# Forge Neo adapter for the ComfyUI-Spectrum-MOSS v0.8.0 diffusion-model wrapper
# (https://github.com/Rkkss/ComfyUI-Spectrum-MOSS, nodes.py).
#
# SPDX-License-Identifier: GPL-3.0-only

"""Where the node's `DIFFUSION_MODEL` wrapper becomes a Forge `model_function_wrapper`.

The node wraps the diffusion model itself; Forge Neo's nearest hook is one level up, around
`KModel.apply_model` (`backend/sampling/sampling_function.py`, the `model_function_wrapper`
call). So an actual step calls the model function as usual, with the model adapter's hooks
capturing the image tokens' hidden state (models.py), and a forecast step does what
`apply_model` would have done around the transformer: `predictor.calculate_input`, the
model's output head on the forecast hidden state at `predictor.timestep(sigma)`, then
`predictor.calculate_denoised`.

The step index comes from `transformer_options["sampling_sigmas"]`, rebased to the first
sigma this run sees, because img2img and the hires pass publish the full schedule while the
sampler runs only its tail.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch

from backend import memory_management
from backend.logging import setup_logger

from .core import (
    SpectrumBranchState,
    SpectrumConfig,
    SpectrumRootState,
    _adaptive_validate_and_update,
    _choose_actual,
    _find_step_index,
    _log_debug,
    _log_warning,
    _normalize_step_position,
    _note_decision,
    _patch_signature,
    logger,
)
from .er_sde_bridge import ERSDESolverBridge
from .models import SUPPORTED, adapter_for

setup_logger(logger)

WRAPPER_TAG = "_neo_spectrum_dit"
BUILTIN_SPECTRUM_WRAPPER = "spectrum_unet_wrapper"


def _resolve_cache_target(hidden: torch.Tensor, cache_device: str) -> tuple[torch.device, torch.dtype]:
    if cache_device == "cpu":
        return torch.device("cpu"), hidden.dtype
    if cache_device == "offload_device":
        try:
            return torch.device(memory_management.unet_offload_device()), hidden.dtype
        except Exception:
            return torch.device("cpu"), hidden.dtype
    return hidden.device, hidden.dtype


def _copy_feature(hidden: torch.Tensor, cache_device: str) -> Optional[torch.Tensor]:
    """A copy, not a view: residual streams are updated in place, and a view would also keep
    the text tokens of the whole activation alive."""
    target_device, target_dtype = _resolve_cache_target(hidden, cache_device)
    try:
        return hidden.detach().to(
            device=target_device,
            dtype=target_dtype,
            non_blocking=(target_device.type == hidden.device.type),
            copy=True,
            memory_format=torch.contiguous_format,
        )
    except RuntimeError:
        # VRAM/offload target failed: prefer a slower CPU anchor to breaking the run.
        try:
            return hidden.detach().to("cpu", copy=True, memory_format=torch.contiguous_format)
        except RuntimeError:
            return None


def _capture_hooks(adapter: Any, core: Any, config: SpectrumConfig, captured: dict) -> list:
    """Install the adapter's hooks; each piece of hidden state they hand over is copied into
    `captured["pieces"]`. Return the handles."""
    pieces = captured.setdefault("pieces", [])

    def store(hidden: torch.Tensor) -> None:
        copied = _copy_feature(hidden, config.cache_device)
        if copied is not None:
            pieces.append(copied)

    return adapter.hooks(core, store)


def _captured_feature(captured: dict) -> Optional[torch.Tensor]:
    """The call's hidden state: one piece, or the per-sample pieces joined along the batch."""
    pieces = captured.get("pieces") or []
    if not pieces:
        return None
    if len(pieces) == 1:
        return pieces[0]
    try:
        return torch.cat([p.to(pieces[0].device) for p in pieces], dim=0)
    except RuntimeError:
        return None


def forecast_denoised(kmodel: Any, adapter: Any, core: Any, x: torch.Tensor, sigma: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
    """`KModel.apply_model` (`backend/modules/k_model.py`) with the transformer replaced by the head."""
    predictor = kmodel.predictor
    xc = predictor.calculate_input(sigma, x).to(kmodel.computation_dtype)
    t = predictor.timestep(sigma).float()
    model_output = adapter.head(core, xc, t, feature).float()
    return predictor.calculate_denoised(sigma, model_output, x)


@dataclass
class RunReport:
    """What the last finished run did, for the one console line and the tests."""

    total_steps: int = 0
    actual_calls: int = 0
    forecast_calls: int = 0
    branches: int = 0
    decisions: list = None
    model: str = ""


def make_wrapper(
    adapter: Any,
    core: Any,
    kmodel: Any,
    config: SpectrumConfig,
    bridge: ERSDESolverBridge,
    previous: Optional[Callable] = None,
    on_run_complete: Optional[Callable[[RunReport], None]] = None,
):
    root = SpectrumRootState(config, er_sde=bridge.bridge_enabled)
    decisions: list = []
    # Last step each branch was called at. The node waits for *every* branch to reach the
    # final step, which never happens when SLG ends early (Forge Neo's default range is
    # 0 to 0.5): its branches stop at the midpoint, so the anchors were never released.
    last_seen: dict = {}

    def call_model(model_function: Callable, kwargs: dict) -> torch.Tensor:
        if previous is not None:
            return previous(model_function, kwargs)
        return model_function(kwargs["input"], kwargs["timestep"], **kwargs["c"])

    def run_actual(model_function, kwargs, branch: SpectrumBranchState, reason, step_index, total_steps, coord, branch_key, x) -> torch.Tensor:
        runtime_config = branch.config
        captured: dict = {}
        handles = _capture_hooks(adapter, core, runtime_config, captured)
        try:
            out = call_model(model_function, kwargs)
        finally:
            for handle in handles:
                handle.remove()

        feature = _captured_feature(captured)
        image_tokens = int(x.shape[-2]) * int(x.shape[-1])
        if feature is not None and (feature.ndim != 3 or int(feature.shape[0]) != int(x.shape[0]) or int(feature.shape[1]) != image_tokens):
            _log_debug(runtime_config.debug, f"captured feature {tuple(feature.shape)} does not match the input; staying on real steps")
            feature = None

        if feature is not None:
            tail_start = max(0, total_steps - runtime_config.tail_actual_steps) if runtime_config.tail_actual_steps > 0 else total_steps
            adaptive_action_start = runtime_config.warmup_steps
            if runtime_config.adaptive:
                # Learn baselines throughout the protected band, but only let the
                # controller change horizon after the detail guard.
                adaptive_action_start += max(0, runtime_config.detail_guard_steps)
            allow_adaptive_action = step_index >= adaptive_action_start and step_index < tail_start and reason != "local_repair"
            preceding_forecast_streak = branch.consecutive_forecasts
            allow_baseline_update = reason not in {"local_repair", "tail"}
            _adaptive_validate_and_update(runtime_config, branch, coord, feature, allow_adaptive_action, allow_baseline_update, preceding_forecast_streak)
            try:
                branch.forecaster.update(coord, feature)
            except ValueError:
                branch.reset()
                try:
                    branch.forecaster.update(coord, feature)
                except ValueError:
                    _log_warning("could not record last-block hidden state after reset")
        else:
            _log_debug(runtime_config.debug, "the hidden-state hook captured nothing; staying on real steps")

        _log_debug(
            runtime_config.debug,
            f"step {step_index + 1}/{total_steps} branch={branch_key} "
            f"mode=actual reason={reason} history={branch.forecaster.history_size} "
            f"p95={branch.last_p95 if branch.last_p95 is not None else 'n/a'} "
            f"p99={branch.last_p99 if branch.last_p99 is not None else 'n/a'} "
            f"ratio={branch.last_local_ratio if branch.last_local_ratio is not None else 'n/a'} "
            f"tier={branch.last_validation_tier} horizon={branch.adaptive_horizon} "
            f"cooldown={branch.cooldown_refreshes_remaining}",
        )
        return out

    def run_forecast(branch: SpectrumBranchState, step_index, total_steps, coord, branch_key, x, sigma) -> torch.Tensor:
        runtime_config = branch.config
        feature = branch.forecaster.predict(coord)
        out = forecast_denoised(kmodel, adapter, core, x, sigma, feature)
        _log_debug(
            runtime_config.debug,
            f"step {step_index + 1}/{total_steps} branch={branch_key} "
            f"mode=forecast history={branch.forecaster.history_size} "
            f"ratio={branch.last_local_ratio if branch.last_local_ratio is not None else 'n/a'} "
            f"tier={branch.last_validation_tier} horizon={branch.adaptive_horizon} "
            f"cooldown={branch.cooldown_refreshes_remaining}",
        )
        return out

    def spectrum_dit_wrapper(model_function: Callable, kwargs: dict) -> torch.Tensor:
        x = kwargs["input"]
        sigma = kwargs["timestep"]
        c = kwargs.get("c") or {}
        transformer_options = c.get("transformer_options")
        if not isinstance(transformer_options, dict):
            transformer_options = {}

        sigmas = transformer_options.get("sampling_sigmas")
        if not isinstance(sigmas, torch.Tensor) or sigmas.numel() < 2 or not torch.is_tensor(x) or x.ndim != 4:
            return call_model(model_function, kwargs)

        raw_index = _find_step_index(sigmas, sigma)

        # New-run detection, on the index into the published schedule. The schedule tensor
        # is normally new per sampling pass; a backwards step also catches a reused one.
        if (
            (root.last_sigmas_id is not None and root.last_sigmas_id != id(sigmas))
            or (root.last_total_steps is not None and root.last_total_steps != int(sigmas.numel()) - 1)
            or (root.last_step_index is not None and raw_index >= 0 and raw_index < root.last_step_index)
        ):
            root.reset_run()
            decisions.clear()
            last_seen.clear()

        if root.step_offset is None:
            root.step_offset = max(0, raw_index)
            bridge.set_schedule(sigmas[root.step_offset :])

        if bridge.bypass_spectrum:
            return call_model(model_function, kwargs)

        step_index = raw_index - root.step_offset if raw_index >= 0 else -1
        total_steps = max(1, int(sigmas.numel()) - 1 - root.step_offset)

        cond_key = tuple(transformer_options.get("cond_or_uncond") or kwargs.get("cond_or_uncond") or ())
        patch_key = _patch_signature(transformer_options)
        # a model-specific part, such as FIBO 2's text phase: a new history where it changes
        try:
            model_key = tuple(adapter.branch_key(core, c, sigma))
        except Exception:
            model_key = ()
        branch_key = (cond_key, patch_key) + ((model_key,) if model_key else ())
        branch = root.branch(branch_key)
        runtime_config = branch.config

        image_tokens = int(x.shape[-2]) * int(x.shape[-1])
        known_shape = branch.forecaster.feature_shape
        if known_shape is not None and (len(known_shape) != 3 or int(known_shape[0]) != int(x.shape[0]) or int(known_shape[1]) != image_tokens):
            branch.reset()

        control_present = c.get("control") is not None or transformer_options.get("control") is not None
        actual, reason = _choose_actual(branch, step_index, total_steps, control_present, runtime_config)
        coord = _normalize_step_position(step_index, total_steps)

        try:
            if actual:
                out = run_actual(model_function, kwargs, branch, reason, step_index, total_steps, coord, branch_key, x)
            else:
                try:
                    out = run_forecast(branch, step_index, total_steps, coord, branch_key, x, sigma)
                except Exception as exc:
                    # A numerical/shape/API problem should never ruin a generation.
                    actual = True
                    reason = "forecast_failed"
                    _log_debug(runtime_config.debug, f"forecast failed ({type(exc).__name__}: {exc}); using real step")
                    out = run_actual(model_function, kwargs, branch, reason, step_index, total_steps, coord, branch_key, x)
        finally:
            _note_decision(branch, actual, reason, runtime_config)
            decisions.append((step_index, branch_key, actual, reason))
            last_seen[branch_key] = step_index
            try:
                bridge.note_branch_decision(step_index, total_steps, actual, reason, branch_key)
            except Exception as exc:
                _log_debug(runtime_config.debug, f"decision observer failed ({type(exc).__name__}: {exc})")
            root.last_step_index = raw_index
            root.last_sigmas_id = id(sigmas)
            root.last_total_steps = int(sigmas.numel()) - 1

        # Release anchors after all branches still running near the end have completed.
        if step_index >= 0 and step_index >= total_steps - 1:
            root.completed_branches.add(branch_key)
            live = {key for key, seen in last_seen.items() if seen >= total_steps - 2}
            if live <= root.completed_branches and not root.summary_logged:
                actual_calls = sum(b.actual_count for b in root.branches.values())
                forecast_calls = sum(b.forecast_count for b in root.branches.values())
                report = RunReport(total_steps, actual_calls, forecast_calls, len(root.branches), list(decisions), adapter.name)
                if runtime_config.debug:
                    total_calls = actual_calls + forecast_calls
                    ratio = (float(total_calls) / float(actual_calls)) if actual_calls else 0.0
                    _log_debug(
                        True,
                        f"run complete: branches={len(root.branches)} actual={actual_calls} "
                        f"forecast={forecast_calls} calls={total_calls} theoretical_transformer_speedup={ratio:.2f}x",
                    )
                root.summary_logged = True
                for b in root.branches.values():
                    b.forecaster.release_features()
                if on_run_complete is not None:
                    on_run_complete(report)

        return out

    setattr(spectrum_dit_wrapper, WRAPPER_TAG, True)
    spectrum_dit_wrapper.root_state = root  # type: ignore[attr-defined]
    spectrum_dit_wrapper.bridge = bridge  # type: ignore[attr-defined]

    def release() -> None:
        for b in root.branches.values():
            b.forecaster.release_features()
        bridge.release()

    spectrum_dit_wrapper.release = release  # type: ignore[attr-defined]
    return spectrum_dit_wrapper


def er_sde_lambdas_fn(predictor: Any) -> Callable[[torch.Tensor], torch.Tensor]:
    """ER-SDE's lambda coordinates, computed exactly as `sample_er_sde` computes them."""
    from k_diffusion.sampling import offset_first_sigma_for_snr, sigma_to_half_log_snr

    def lambdas(sigmas: torch.Tensor) -> torch.Tensor:
        return sigma_to_half_log_snr(offset_first_sigma_for_snr(sigmas, predictor), predictor).neg().exp()

    return lambdas


def install(unet: Any, config: SpectrumConfig, *, sampler_name: str, brownian_noise: bool = False, on_run_complete=None) -> tuple[Optional[Any], str]:
    """Clone `unet` and install Spectrum on it; `(None, reason)` when it must stand down."""
    config.validate()
    core = unet.get_model_object("diffusion_model")
    adapter = adapter_for(core)
    if adapter is None:
        return None, f"the loaded model is not {SUPPORTED}"

    previous = unet.model_options.get("model_function_wrapper")
    if previous is not None and getattr(previous, "__name__", "") == BUILTIN_SPECTRUM_WRAPPER:
        return None, "the built-in Spectrum Integrated extension is enabled; use one or the other"
    if previous is not None and getattr(previous, WRAPPER_TAG, False):
        previous = getattr(previous, "_previous", None)

    er_sde = sampler_name == "sample_er_sde"
    bridge = ERSDESolverBridge(debug=config.debug, enabled=er_sde, lambdas_fn=er_sde_lambdas_fn(unet.model.predictor) if er_sde else None)
    if er_sde and brownian_noise:
        # The validated bridge assumes ER-SDE's own noise; a custom one is not bridged.
        bridge.force_bypass("ER-SDE with a custom noise sampler is not bridged; using actual steps.")

    clone = unet.clone()
    wrapper = make_wrapper(adapter, core, clone.model, config, bridge, previous, on_run_complete)
    wrapper.model_name = adapter.name  # type: ignore[attr-defined]
    wrapper._previous = previous  # type: ignore[attr-defined]
    clone.set_model_unet_function_wrapper(wrapper)
    if er_sde and not bridge.bypass_spectrum:
        clone.set_model_sampler_post_cfg_function(bridge.post_cfg_function)
    return clone, "er_sde" if bridge.bridge_enabled else "plain"
