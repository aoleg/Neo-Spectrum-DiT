# Neo Spectrum MOSS-Image

A **Forge Neo** extension port of [Rkkss/ComfyUI-Spectrum-MOSS](https://github.com/Rkkss/ComfyUI-Spectrum-MOSS) v0.8.0: Spectrum-style hidden-state forecasting that skips transformer evaluations on **MOSS-Image**.

On a forecast step the transformer does not run. The extension predicts MOSS-Image's last-block hidden state from the real steps before it (a ridge-regularised Chebyshev fit blended with linear extrapolation) and runs only the model's small timestep-conditioned output head on the prediction. The original reports about 2x to 2.5x fewer transformer calls with its defaults.

## How it differs from the built-in Spectrum

Forge Neo ships **Spectrum Integrated**, which forecasts the model's whole output and works on any model. This extension is MOSS-Image only and forecasts one level earlier, before the output head, so the head still sees the true timestep. It also brings the original's two schedulers and its ER SDE bridge. Use one or the other: with Spectrum Integrated enabled, this extension stands down and says so in the console.

## Install

Extensions → Install from URL:

```text
https://github.com/aoleg/Neo-Spectrum-MOSS
```

Or clone into your Forge Neo `extensions` folder and restart. No extra dependencies. It needs a Forge Neo with native MOSS-Image support (be6fa5d4, 2026-10-21, or later). While the older `moss-image-forge-neo` extension is installed, its model loader wins and this extension stands down, because it rebuilds the output head from Forge Neo's own MOSS-Image code.

## Usage

Open the **Spectrum MOSS-Image** accordion, tick it, and generate. The console reports how many transformer calls ran, for example `50 steps, 46 of 100 transformer calls run (2.17x fewer)` (cond and uncond count separately).

Recommended settings, from the original:

- OpenMOSS: 50 steps, Euler, Simple (Forge Neo's MOSS-Image preset).
- Comfy-Org template: 25 steps, ER SDE, Simple. ER SDE is detected and goes through the solver bridge automatically.

Other samplers run, but only Euler and ER SDE were validated; the console warns once per pass (Advanced → Warn on unverified samplers turns that off).

### Skip Layer Guidance

Forge Neo enables MOSS-Image Skip Layer Guidance by default, from 0 to 0.5 (Settings → Moss-Image). While SLG runs, cond and uncond are separate model calls; when it stops, they are batched into one, and the forecaster has to learn that new call from scratch. That costs about five real steps at the midpoint. For the most speed, set **End Percent** to 1.0, as the original's README recommends for ComfyUI.

## Modes

### Manual (default)

The original's **Spectrum MOSS-Image (Manual)** node: a fixed expanding-window schedule. The defaults here are faster than the node's (warmup 8, window 2.0, growth 0.75, 2 forecasts in a row); the node's values are in brackets.

| Control | Default | Description |
| --- | --- | --- |
| Warmup steps | 4 (8) | First steps that always run the transformer. |
| Tail actual steps | 2 | Last steps that always run the transformer. Shared by both modes. |
| Max consecutive forecasts | 8 (2) | Forecasts allowed in a row before a real step. The original recommends 1 under ER SDE. |
| Window size | 8.0 (2.0) | Starting forecast window before a real step. |
| Window growth | 2.0 (0.75) | How much the window grows after each refresh. |
| History points | 8 | Real hidden states kept for the predictor. ER SDE: fixed at 8. |
| Chebyshev degree | 4 | Polynomial degree of the predictor. ER SDE: fixed at 4. |
| Ridge lambda | 0.1 | Regularisation; higher damps extrapolation. ER SDE: fixed at 0.1. |
| Blend weight | 0.5 | 0 = linear extrapolation only, 1 = spectral only. ER SDE: fixed at 0.5. |
| Real steps while a ControlNet is active | on | |

### Adaptive

The original's **Spectrum MOSS-Image (Adaptive)** node, with its defaults. At every real step it measures how far the forecast would have been off, on a sample of image tokens (95th and 99th percentile, so a small drifting region such as a hand is not averaged away), and shortens or lengthens the forecast streak.

| Control | Default | Description |
| --- | --- | --- |
| Adaptive warmup steps | 8 | First steps that always run the transformer. |
| Detail guard steps | 8 | After warmup, at most one forecast between real steps. |
| Post-guard horizon cap | 6 | Longest forecast streak after the guard. ER SDE: 1. |
| Error tolerance | 1.0 | Forecast error tolerated before backing off. Higher = faster, riskier. |
| Forecast budget | 2.0 | Starting forecast budget before the next real step (the node's `window_size`). ER SDE: ignored. |
| Budget growth | 0.5 | How fast the budget grows after good refreshes (the node's `flex_window`). ER SDE: ignored. |

### Advanced

| Control | Default | Description |
| --- | --- | --- |
| Cache device | main_device | Where the hidden-state history lives. `main_device` is fastest but uses VRAM (about 25 MB per kept state at 1024x1024 per batch item); `cpu` or `offload_device` save VRAM at some speed. |
| Apply to the Hires. fix pass | on | |
| Warn on unverified samplers | on | Changes nothing but the console. |
| Debug log | off | Every decision, the forecast error and the final counts. |

All settings are written to the infotext and restored by "Read generation parameters".

## Implementation notes

The scheduler, the forecaster and the adaptive controller (`lib_spectrum_moss/core.py`) are the original's code, unchanged apart from two flags that Forge supplies differently. What is new:

- **Hook.** The node wraps the diffusion model; Forge Neo's nearest hook is the `model_function_wrapper` around `KModel.apply_model`. A forecast step reproduces what `apply_model` does around the transformer: `calculate_input`, the head at `predictor.timestep(sigma)`, `calculate_denoised`. Another extension's wrapper already in that slot is chained, not replaced, and sees only the real steps.
- **Step index.** It comes from `transformer_options["sampling_sigmas"]`, rebased to the first sigma of the run: for img2img and the hires pass Forge publishes the full schedule while the sampler runs only its tail, so without the rebase warmup and the detail guard would be skipped.
- **ER SDE.** The node copies ComfyUI's sampler only to capture ER SDE's lambda coordinates. Forge Neo's `sample_er_sde` computes them from the sigmas, so the bridge computes the same values up front and never touches the sampler; the guided result reaches it through a post-CFG function.
- **Release with partial SLG.** The node frees its stored hidden states only after every call pattern reaches the last step, which never happens when SLG ends at the midpoint. The port frees them once every pattern still running near the end has finished.

## Credits

- Spectrum-MOSS node: Rkkss, https://github.com/Rkkss/ComfyUI-Spectrum-MOSS
- Spectrum method: *Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration*, arXiv:2603.01623; official implementation https://github.com/hanjq17/Spectrum
- ER SDE solver-space bridge approach: https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3

## License

GPL-3.0-only, as the original. See `LICENSE`.
