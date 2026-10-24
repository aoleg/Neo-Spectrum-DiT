# Neo-Spectrum-DiT
#
# SPDX-License-Identifier: GPL-3.0-only

"""What Spectrum needs from each DiT it supports, one adapter per model.

Spectrum forecasts the hidden state of the image tokens as it enters the model's output head,
and on a forecast step runs only that head. Per model that takes three things:

- `matches(core)`: whether the loaded diffusion model is this one;
- `hooks(core, store)`: forward hooks that hand the image tokens' hidden state to `store`
  during a real call (`store` may be called more than once per call; the wrapper joins the
  pieces along the batch);
- `head(core, xc, t, feature)`: the model output (B, C, H, W) from a hidden state, where `xc`
  is the model input in the computation dtype and `t` the model timestep, as
  `KModel.apply_model` passes them.

`branch_key(core, c, sigma)` adds to the forecast history's key anything that changes the
hidden-state trajectory mid-run, so a new history starts there.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch


class MOSSImage:
    """Forge Neo's native MOSS-Image (`backend/nn/moss_image.py`).

    The older moss-image-forge-neo extension, while installed, loads its own model class;
    its head is not guaranteed to match, so only Forge Neo's own class is accepted.
    """

    name = "MOSS-Image"

    @staticmethod
    def matches(core: Any) -> bool:
        from backend.nn import moss_image as moss_nn

        return isinstance(core, moss_nn.MOSSImage)

    @staticmethod
    def hooks(core: Any, store: Callable[[torch.Tensor], None]) -> list:
        """The last block's output, cut to the image tokens of this call.

        The pre-hook records how many image tokens *this* call has, so a caller that splits
        the latent into tiles cannot have a tile's tokens mistaken for the whole image's.
        """
        tokens: dict = {}

        def pre_hook(_module: Any, args: tuple, kwargs: dict) -> None:
            x = args[0] if args else kwargs.get("x")
            if torch.is_tensor(x) and x.ndim == 4:
                tokens["n"] = int(x.shape[-2]) * int(x.shape[-1])

        def last_block_hook(_module: Any, _args: tuple, output: Any) -> None:
            n = tokens.get("n")
            if n is None or not torch.is_tensor(output) or output.ndim != 3 or output.shape[1] < n:
                return
            store(output[:, :n])

        return [
            core.register_forward_pre_hook(pre_hook, with_kwargs=True),
            core.layers[-1].register_forward_hook(last_block_hook),
        ]

    @staticmethod
    def head(core: Any, xc: torch.Tensor, t: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
        """The last lines of `MOSSImage.forward`, through the same module functions, so a
        quantized or offloaded head casts its weights the way the model does."""
        from backend.nn import moss_image as moss_nn

        dtype = xc.dtype
        device = xc.device
        batch, _, height, width = xc.shape

        hidden = feature.to(device=device, dtype=dtype, non_blocking=(feature.device.type == device.type))
        vec = core.time_embedding(moss_nn.timestep_embedding(t, 256).to(dtype))
        head_shift, head_scale = core.video_head_modulation(vec).unflatten(-1, (2, -1)).unsqueeze(2).unbind(1)
        out = core.final_linear_video(moss_nn.modulated_rms_norm(hidden, core.final_norm_video, head_scale, head_shift))
        return out.transpose(1, 2).reshape(batch, int(core.latent_channels), height, width)

    @staticmethod
    def branch_key(core: Any, c: dict, sigma: torch.Tensor) -> tuple:
        return ()


# FIBO 2's text phases (fibo2-forge-neo, forge_fibo2/native/phases.py): a caption with
# text_render entries carries two conditionings, the EARLY rows for sigma above 0.9 and the
# LATE rows from 0.9 down, told apart by an extra last feature.
_FIBO2_EARLY = 1.0
_FIBO2_TEXT_IMAGES_SIGMA = 0.9
_FIBO2_TOLERANCE = 1e-6


class Fibo2:
    """FIBO 2.0 from the fibo2-forge-neo extension (`forge_fibo2/native/transformer.py`).

    Recognised by class name and the attributes Spectrum touches, not by import: the
    extension may load after this one, or not at all.
    """

    name = "FIBO 2"

    @staticmethod
    def matches(core: Any) -> bool:
        return type(core).__name__ == "Fibo2DiT" and all(
            hasattr(core, name) for name in ("final_layer", "t_embedder", "in_channels", "text_dim")
        )

    @staticmethod
    def hooks(core: Any, store: Callable[[torch.Tensor], None]) -> list:
        """The final layer's input: `Fibo2DiT.run` hands it exactly the image tokens
        (`x[:, num_gist:end]`), past the gist tokens, whose count follows the prompt length.
        Samples whose prompts differ in length run one by one, so this fires once per sample."""

        def pre_hook(_module: Any, args: tuple) -> None:
            if args and torch.is_tensor(args[0]) and args[0].ndim == 3:
                store(args[0])

        return [core.final_layer.register_forward_pre_hook(pre_hook)]

    @staticmethod
    def head(core: Any, xc: torch.Tensor, t: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
        """The end of `Fibo2DiT.run`: the image's time embedding at sigma * 1000, then the final
        layer on the image tokens."""
        dtype = xc.dtype
        device = xc.device
        batch, _, height, width = xc.shape

        noisy = core.t_embedder(t.to(device=device, dtype=torch.float32) * 1000.0, dtype)
        hidden = feature.to(device=device, non_blocking=(feature.device.type == device.type))
        out = core.final_layer(hidden, noisy, dtype)
        return out.transpose(1, 2).reshape(batch, int(core.in_channels), height, width)

    @staticmethod
    def branch_key(core: Any, c: dict, sigma: torch.Tensor) -> tuple:
        """("early",) or ("late",) when the conditioning switches text at sigma 0.9, else ()."""
        context = c.get("c_crossattn")
        if not torch.is_tensor(context) or context.ndim != 3 or context.shape[-1] != int(core.text_dim) + 1:
            return ()
        if not bool((context[..., -1].round() == _FIBO2_EARLY).any()):
            return ()
        late = float(sigma.reshape(-1)[0]) <= _FIBO2_TEXT_IMAGES_SIGMA + _FIBO2_TOLERANCE
        return ("late",) if late else ("early",)


ADAPTERS = (MOSSImage, Fibo2)
SUPPORTED = " or ".join(adapter.name for adapter in ADAPTERS)


def adapter_for(core: Any) -> Optional[type]:
    for adapter in ADAPTERS:
        try:
            if adapter.matches(core):
                return adapter
        except Exception:
            continue
    return None
