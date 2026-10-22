# Neo-Spectrum-MOSS
#
# SPDX-License-Identifier: GPL-3.0-only

"""
Spectrum for MOSS-Image in WebUI Forge Neo: a port of Rkkss' ComfyUI-Spectrum-MOSS v0.8.0
(https://github.com/Rkkss/ComfyUI-Spectrum-MOSS).

Unlike the built-in "Spectrum Integrated", which forecasts the model's whole output, this
forecasts MOSS-Image's last-block hidden state and runs the real timestep-conditioned output
head on it. The two nodes of the original (Manual and Adaptive) are one Mode switch here.
"""

import gradio as gr

from lib_spectrum_moss import core
from lib_spectrum_moss.forge import install, logger

from modules import scripts
from modules.ui_components import InputAccordion

MANUAL = "Manual"
ADAPTIVE = "Adaptive"
CACHE_DEVICES = ["main_device", "offload_device", "cpu"]


class NeoSpectrumMOSS(scripts.Script):
    # after the built-in Spectrum (2026), so its wrapper is already in place to be detected
    sorting_priority = 2027

    #   class attributes: the wrapper reports long after `process_before_every_sampling`
    #   returned, and `postprocess` is not guaranteed to see the same instance
    reports: list = []
    wrappers: list = []

    def title(self):
        return "Spectrum MOSS-Image"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        #   labels must be unique within the script: `ui-config.json` keys saved defaults
        #   by label, so two "Window size" sliders would share one saved value
        with InputAccordion(False, label=self.title()) as enable:
            mode = gr.Radio(value=MANUAL, choices=[MANUAL, ADAPTIVE], label="Mode", info="Manual is a fixed schedule; Adaptive measures forecast error at each real step and adjusts")

            with gr.Group(visible=True) as g_manual:
                with gr.Row():
                    manual_warmup_steps = gr.Slider(minimum=0, maximum=64, value=8, step=1, label="Warmup steps", info="first steps that always run the full transformer; below 8 the layout changes")
                    manual_tail_steps = gr.Slider(minimum=0, maximum=64, value=3, step=1, label="Tail actual steps", info="last steps that always run the full transformer; higher = safer")
                with gr.Row():
                    manual_window_size = gr.Slider(minimum=1.0, maximum=16.0, value=3.0, step=0.25, label="Window size", info="starting forecast window before a real step")
                    manual_flex_window = gr.Slider(minimum=0.0, maximum=8.0, value=0.5, step=0.05, label="Window growth", info="how much the window grows after each refresh")
                max_consecutive_forecasts = gr.Slider(minimum=0, maximum=32, value=3, step=1, label="Max consecutive forecasts", info="forecasts allowed in a row before a real step; higher = faster, riskier. ER SDE: 1")
                with gr.Row():
                    history_points = gr.Slider(minimum=2, maximum=32, value=8, step=1, label="History points", info="real hidden states kept for the predictor; more uses more memory. ER SDE: ignored")
                    chebyshev_degree = gr.Slider(minimum=1, maximum=12, value=4, step=1, label="Chebyshev degree", info="higher fits more complex trends, may be less stable. ER SDE: ignored")
                with gr.Row():
                    ridge_lambda = gr.Slider(minimum=0.0, maximum=10.0, value=0.1, step=0.01, label="Ridge lambda", info="higher damps extrapolation more. ER SDE: ignored")
                    blend_weight = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="Blend weight", info="0 = linear extrapolation only, 1 = spectral only. ER SDE: ignored")
                force_actual_on_control = gr.Checkbox(True, label="Real steps while a ControlNet is active")

            with gr.Group(visible=False) as g_adaptive:
                with gr.Row():
                    adaptive_warmup_steps = gr.Slider(minimum=0, maximum=64, value=8, step=1, label="Adaptive warmup steps", info="first steps that always run the full transformer; higher = safer")
                    adaptive_tail_steps = gr.Slider(minimum=0, maximum=16, value=2, step=1, label="Adaptive tail steps", info="last steps that always run the full transformer; higher = safer")
                with gr.Row():
                    detail_guard_steps = gr.Slider(minimum=0, maximum=32, value=8, step=1, label="Detail guard steps", info="after warmup, at most one forecast between real steps; higher = safer early detail")
                with gr.Row():
                    post_guard_horizon_cap = gr.Slider(minimum=1, maximum=16, value=6, step=1, label="Post-guard horizon cap", info="longest forecast streak after the guard; higher = faster, riskier. ER SDE: 1")
                error_tolerance = gr.Slider(minimum=0.75, maximum=1.5, value=1.0, step=0.05, label="Error tolerance", info="forecast error tolerated before backing off; higher = faster, riskier")
                with gr.Row():
                    adaptive_window_size = gr.Slider(minimum=1.0, maximum=16.0, value=2.0, step=0.25, label="Forecast budget", info="starting forecast budget before the next real step. ER SDE: ignored")
                    adaptive_flex_window = gr.Slider(minimum=0.0, maximum=8.0, value=0.5, step=0.05, label="Budget growth", info="how fast the budget grows after good refreshes. ER SDE: ignored")

            with gr.Accordion("Advanced", open=False):
                with gr.Row():
                    cache_device = gr.Dropdown(value="main_device", choices=CACHE_DEVICES, label="Cache device", info="where the hidden-state history lives; main_device is fastest but uses VRAM")
                    apply_to_hr = gr.Checkbox(True, label="Apply to the Hires. fix pass")
                with gr.Row():
                    warn_unverified_sampler = gr.Checkbox(True, label="Warn on unverified samplers", info="Euler and ER SDE are verified; the warning changes nothing")
                    debug = gr.Checkbox(False, label="Debug log", info="every decision, forecast error and the final counts in the console")

        mode.change(
            fn=lambda chosen: [gr.update(visible=chosen == MANUAL), gr.update(visible=chosen == ADAPTIVE)],
            inputs=[mode],
            outputs=[g_manual, g_adaptive],
            show_progress=False,
        )

        self.infotext_fields = [
            (enable, lambda d: "Spectrum MOSS mode" in d),
            (mode, "Spectrum MOSS mode"),
            #   one infotext key per pair of sliders: the restored mode picks which one counts
            (manual_warmup_steps, "Spectrum MOSS warmup"),
            (adaptive_warmup_steps, "Spectrum MOSS warmup"),
            (manual_tail_steps, "Spectrum MOSS tail"),
            (adaptive_tail_steps, "Spectrum MOSS tail"),
            (detail_guard_steps, "Spectrum MOSS detail guard"),
            (post_guard_horizon_cap, "Spectrum MOSS horizon cap"),
            (adaptive_window_size, "Spectrum MOSS adaptive window"),
            (adaptive_flex_window, "Spectrum MOSS adaptive growth"),
            (error_tolerance, "Spectrum MOSS error tolerance"),
            (manual_window_size, "Spectrum MOSS window"),
            (manual_flex_window, "Spectrum MOSS growth"),
            (max_consecutive_forecasts, "Spectrum MOSS max forecasts"),
            (history_points, "Spectrum MOSS history"),
            (chebyshev_degree, "Spectrum MOSS degree"),
            (ridge_lambda, "Spectrum MOSS ridge"),
            (blend_weight, "Spectrum MOSS blend"),
            (force_actual_on_control, "Spectrum MOSS real on control"),
        ]

        return [
            enable,
            mode,
            manual_warmup_steps,
            manual_tail_steps,
            adaptive_warmup_steps,
            adaptive_tail_steps,
            detail_guard_steps,
            post_guard_horizon_cap,
            adaptive_window_size,
            adaptive_flex_window,
            error_tolerance,
            manual_window_size,
            manual_flex_window,
            max_consecutive_forecasts,
            history_points,
            chebyshev_degree,
            ridge_lambda,
            blend_weight,
            force_actual_on_control,
            cache_device,
            apply_to_hr,
            warn_unverified_sampler,
            debug,
        ]

    @staticmethod
    def make_config(mode, manual_warmup_steps, manual_tail_steps, adaptive_warmup_steps, adaptive_tail_steps, detail_guard_steps, post_guard_horizon_cap, adaptive_window_size, adaptive_flex_window, error_tolerance, manual_window_size, manual_flex_window, max_consecutive_forecasts, history_points, chebyshev_degree, ridge_lambda, blend_weight, force_actual_on_control, cache_device, debug) -> tuple[core.SpectrumConfig, dict]:
        """UI values -> the config the node would have built, and the infotext for it.

        Adaptive pins the predictor shape to the validated constants and always forces real
        steps under a ControlNet, exactly as the Adaptive node does.
        """
        if mode == MANUAL:
            config = core.SpectrumConfig(
                mode="manual",
                warmup_steps=int(manual_warmup_steps),
                tail_actual_steps=int(manual_tail_steps),
                window_size=float(manual_window_size),
                flex_window=float(manual_flex_window),
                max_consecutive_forecasts=int(max_consecutive_forecasts),
                history_points=int(history_points),
                chebyshev_degree=int(chebyshev_degree),
                ridge_lambda=float(ridge_lambda),
                blend_weight=float(blend_weight),
                cache_device=str(cache_device),
                force_actual_on_control=bool(force_actual_on_control),
                debug=bool(debug),
            )
            params = {
                "Spectrum MOSS window": config.window_size,
                "Spectrum MOSS growth": config.flex_window,
                "Spectrum MOSS max forecasts": config.max_consecutive_forecasts,
                "Spectrum MOSS history": config.history_points,
                "Spectrum MOSS degree": config.chebyshev_degree,
                "Spectrum MOSS ridge": config.ridge_lambda,
                "Spectrum MOSS blend": config.blend_weight,
                "Spectrum MOSS real on control": config.force_actual_on_control,
            }
        else:
            config = core.SpectrumConfig(
                mode="adaptive",
                warmup_steps=int(adaptive_warmup_steps),
                tail_actual_steps=int(adaptive_tail_steps),
                window_size=float(adaptive_window_size),
                flex_window=float(adaptive_flex_window),
                history_points=core._ADAPTIVE_HISTORY_POINTS,
                chebyshev_degree=core._ADAPTIVE_CHEBYSHEV_DEGREE,
                ridge_lambda=core._ADAPTIVE_RIDGE_LAMBDA,
                blend_weight=core._ADAPTIVE_BLEND_WEIGHT,
                cache_device=str(cache_device),
                force_actual_on_control=True,
                detail_guard_steps=int(detail_guard_steps),
                post_guard_horizon_cap=max(1, int(post_guard_horizon_cap)),
                error_tolerance=float(error_tolerance),
                debug=bool(debug),
            )
            params = {
                "Spectrum MOSS detail guard": config.detail_guard_steps,
                "Spectrum MOSS horizon cap": config.post_guard_horizon_cap,
                "Spectrum MOSS adaptive window": config.window_size,
                "Spectrum MOSS adaptive growth": config.flex_window,
                "Spectrum MOSS error tolerance": config.error_tolerance,
            }

        params = {
            "Spectrum MOSS mode": MANUAL if config.mode == "manual" else ADAPTIVE,
            "Spectrum MOSS warmup": config.warmup_steps,
            "Spectrum MOSS tail": config.tail_actual_steps,
            **params,
        }
        return config, params

    def process(self, p, enable, *args):
        cls = NeoSpectrumMOSS
        cls.release_all()
        cls.reports = []

    def process_before_every_sampling(self, p, enable, mode, *args, **kwargs):
        if not enable:
            return

        apply_to_hr, warn_unverified_sampler, debug = args[-3:]
        if getattr(p, "is_hr_pass", False) and not apply_to_hr:
            return

        try:
            config, params = self.make_config(mode, *args[:-4], args[-4], debug)
            config.validate()
        except Exception as exc:
            logger.error(f"Spectrum MOSS: invalid settings ({exc}); standing down")
            return

        sampler = getattr(p, "sampler", None)
        func = getattr(sampler, "func", None)
        sampler_name = str(getattr(func, "__name__", type(sampler).__name__))
        options = getattr(getattr(sampler, "config", None), "options", None) or {}
        if warn_unverified_sampler and sampler_name not in {"sample_euler", "sample_er_sde"}:
            logger.warning(f"Spectrum MOSS: unverified sampler {sampler_name}; the forecasts were validated with Euler and ER SDE")

        unet = p.sd_model.forge_objects.unet
        patched, how = install(unet, config, sampler_name=sampler_name, brownian_noise=bool(options.get("brownian_noise", False)), on_run_complete=NeoSpectrumMOSS.reports.append)
        if patched is None:
            logger.warning(f"Spectrum MOSS: standing down, {how}")
            return

        NeoSpectrumMOSS.wrappers.append(patched.model_options["model_function_wrapper"])
        p.sd_model.forge_objects.unet = patched
        p.extra_generation_params.update(params)
        if how == "er_sde":
            capped = config.mode == "manual" and config.max_consecutive_forecasts > 1
            logger.info("Spectrum MOSS: ER SDE detected, forecasts go through the solver-space bridge" + (", at most 1 in a row" if capped else ""))

    def postprocess(self, p, processed, *args):
        cls = NeoSpectrumMOSS
        for report in cls.reports:
            calls = report.actual_calls + report.forecast_calls
            if report.actual_calls:
                logger.info(f"Spectrum MOSS: {report.total_steps} steps, {report.actual_calls} of {calls} transformer calls run ({calls / report.actual_calls:.2f}x fewer)")
        cls.reports = []
        cls.release_all()

    @classmethod
    def release_all(cls):
        """An interrupted run never reaches its last step, so its anchors are freed here."""
        for wrapper in cls.wrappers:
            try:
                wrapper.release()
            except Exception:
                pass
        cls.wrappers = []
