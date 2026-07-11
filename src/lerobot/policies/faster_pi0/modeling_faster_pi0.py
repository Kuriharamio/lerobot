#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FASTER Pi0 policy — Horizon-Aware Schedule for flow-matching VLA models.

This module provides:
  - ``FasterPI0Pytorch``: drop-in replacement for ``PI0Pytorch`` that overrides
    the training ``forward`` pass and adds a FASTER-specific ``sample_actions_faster``
    inference method.  The model architecture (weights / layers) is *identical* to
    ``PI0Pytorch``, so pre-trained Pi0 checkpoints can be loaded without any
    weight remapping.
  - ``FasterPI0Policy``: ``PreTrainedPolicy`` wrapper that wires the above model
    into the standard LeRobot training and inference APIs.

Algorithm reference
-------------------
"FASTER: Fast Action Sampling for ImmediaTE Reaction" (2025).

Key changes vs. plain Pi0
--------------------------
1. *Per-step noise level* ``tau ∈ R^H`` (Horizon-Aware Schedule) replaces the
   scalar global time ``t``.
2. *Mixed schedule* training: with probability ``has_mix_prob`` use HAS,
   otherwise fall back to the constant-time schedule.
3. *Masked loss*: prefix action steps (``i < d``) are excluded from the MSE
   loss and from the loss normalisation denominator.
4. *Inference early stopping*: the Euler ODE loop terminates as soon as the
   required ``execution_horizon`` actions are fully denoised (``tau == 0``).
"""

import builtins
import copy
import logging
from collections import deque
from pathlib import Path
from typing import Unpack

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lerobot.policies.faster_pi0.configuration_faster_pi0 import FasterPI0Config
from lerobot.policies.pi0.modeling_pi0 import (
    PI0Policy,
    PI0Pytorch,
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
    pad_vector,
)
from lerobot.policies.pretrained import PreTrainedPolicy, T
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

# Small constant to prevent division by zero in the HAS formula when
# ``u_i`` approaches 1.
_EPS: float = 1e-5


# ---------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------


class FasterPI0Pytorch(PI0Pytorch):
    """Pi0 core model extended with a Horizon-Aware Schedule (FASTER).

    The architecture (parameters) is identical to ``PI0Pytorch``; only the
    ``forward`` (training) and inference methods differ.  Pre-trained Pi0
    weights therefore load without any key remapping.
    """

    def __init__(self, config: FasterPI0Config, rtc_processor=None):
        super().__init__(config, rtc_processor=rtc_processor)

    # ------------------------------------------------------------------
    # Per-step timestep embedding
    # ------------------------------------------------------------------

    def embed_suffix_faster(
        self,
        state: Tensor,
        noisy_actions: Tensor,
        tau: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, None]:
        """Embed state, noisy actions, and a (possibly per-step) timestep.

        Parameters
        ----------
        state : Tensor
            Shape ``(B, state_dim)``.
        noisy_actions : Tensor
            Shape ``(B, H, action_dim)``.
        tau : Tensor
            Either ``(B,)`` — one scalar timestep per sample (same as the
            original ``embed_suffix``) — or ``(B, H)`` — one timestep per
            action step.  When ``(B, H)`` is supplied the sinusoidal encoding
            is applied independently to each ``(b, i)`` pair so that different
            horizon positions receive different time conditioning.

        Returns
        -------
        Tuple of ``(embs, pad_masks, att_masks, None)`` matching the signature
        of the parent ``embed_suffix``.
        """
        embs = []
        pad_masks = []
        att_masks_list: list[int] = []

        if self.state_proj.weight.dtype == torch.float32:
            state = state.to(torch.float32)

        def _state_proj(s: Tensor) -> Tensor:
            return self.state_proj(s)

        state_emb = self._apply_checkpoint(_state_proj, state)
        bsize = state_emb.shape[0]
        device = state_emb.device

        embs.append(state_emb[:, None, :])
        pad_masks.append(torch.ones(bsize, 1, dtype=torch.bool, device=device))
        att_masks_list += [1]

        D = self.action_in_proj.out_features

        if tau.ndim == 2:
            # Per-step sinusoidal embedding: (B, H) → flatten → embed → reshape.
            # This gives each horizon position its own time conditioning vector.
            B, H = tau.shape
            tau_flat = tau.reshape(B * H).to(device=device, dtype=torch.float32)
            time_emb_flat = create_sinusoidal_pos_embedding(
                tau_flat,
                D,
                min_period=self.config.min_period,
                max_period=self.config.max_period,
                device=device,
            )  # (B*H, D)
            time_emb = time_emb_flat.reshape(B, H, D)  # (B, H, D)
        else:
            # Scalar timestep per sample — broadcast across the horizon axis.
            tau_1d = tau.to(device=device, dtype=torch.float32)
            time_emb_2d = create_sinusoidal_pos_embedding(
                tau_1d,
                D,
                min_period=self.config.min_period,
                max_period=self.config.max_period,
                device=device,
            )  # (B, D)
            time_emb = time_emb_2d[:, None, :].expand(bsize, self.config.chunk_size, D)

        # Cast to float32 (same as the original embed_suffix — bfloat16 cast is
        # applied in the main forward after embed_suffix_faster returns).
        time_emb = time_emb.to(dtype=torch.float32)

        def _action_proj(x: Tensor) -> Tensor:
            return self.action_in_proj(x)

        action_emb = self._apply_checkpoint(_action_proj, noisy_actions)

        # action_emb may be bfloat16 if model weights are bfloat16; cast time_emb
        # to match so the concatenation is dtype-consistent.
        if action_emb.dtype != time_emb.dtype:
            time_emb = time_emb.to(dtype=action_emb.dtype)

        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        def _mlp(x: Tensor) -> Tensor:
            return self.action_time_mlp_out(F.silu(self.action_time_mlp_in(x)))

        action_time_emb = self._apply_checkpoint(_mlp, action_time_emb)

        embs.append(action_time_emb)
        action_time_dim = action_time_emb.shape[1]
        pad_masks.append(
            torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        )
        # First action token can see all prior tokens; remaining action tokens
        # only see earlier action tokens (causal within the action group).
        att_masks_list += [1] + [0] * (self.config.chunk_size - 1)

        embs_cat = torch.cat(embs, dim=1)
        pad_masks_cat = torch.cat(pad_masks, dim=1)
        att_masks_t = torch.tensor(att_masks_list, dtype=embs_cat.dtype, device=device)
        att_masks_t = att_masks_t[None, :].expand(bsize, len(att_masks_list))

        return embs_cat, pad_masks_cat, att_masks_t, None  # adarms_cond = None

    # ------------------------------------------------------------------
    # HAS schedule helpers
    # ------------------------------------------------------------------

    def _compute_has_tau_batch(
        self,
        rho: Tensor,
        d: Tensor,
        z: Tensor,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        """Compute per-step tau for a full batch (training).

        Parameters
        ----------
        rho : Tensor
            Shape ``(B,)`` — global progress values in ``[0, 1]``.
        d : Tensor
            Shape ``(B,)`` int — prefix lengths, sampled from
            ``[0, has_d_max]``.
        z : Tensor
            Shape ``(B,)`` float — schedule-type flags; 1 ⟹ use HAS,
            0 ⟹ use constant-time schedule.
        device : torch.device
            Target device.  All intermediate tensors are created on this
            device to avoid cross-device errors.

        Returns
        -------
        tau_bh : Tensor
            Shape ``(B, H)`` — per-step noise levels.
        mask_bh : Tensor
            Shape ``(B, H)`` float — 1 at valid (``i >= d``) positions, 0 at
            prefix (``i < d``) positions.
        """
        H = self.config.chunk_size
        alpha = self.config.has_alpha
        u_d_val = self.config.has_u_d

        # Horizon index tensor — must live on the same device as the batch.
        i_h = torch.arange(H, device=device, dtype=torch.float32).unsqueeze(0)  # (1, H)
        d_b = d.float().unsqueeze(1)  # (B, 1)

        # Valid-position mask: i >= d
        mask_bh = (i_h >= d_b).float()  # (B, H)

        # ----- HAS branch -----
        # Denominator: max(H - 1 - d, 1) — clamp so we never divide by zero.
        denom = torch.clamp(H - 1 - d_b, min=1.0)  # (B, 1)
        # Relative position within valid segment, 0 at i==d, 1 at i==H-1.
        rel_pos = torch.clamp((i_h - d_b) / denom, min=0.0)  # (B, H)
        # Hit-time schedule: decays from u_d at i==d to 0 at i==H-1.
        u_i_has = (1.0 - rel_pos) ** alpha * u_d_val  # (B, H)
        # Tau for HAS: clamp(·) ensures tau >= 0; eps guards against u_i ≈ 1.
        tau_has = torch.clamp(
            (rho.unsqueeze(1) - u_i_has) / (1.0 - u_i_has + _EPS),
            min=0.0,
        )  # (B, H)

        # ----- Constant-schedule branch -----
        tau_const = rho.unsqueeze(1).expand(-1, H)  # (B, H)

        # Mix the two branches according to z, then zero out prefix positions.
        z_b = z.unsqueeze(1)  # (B, 1)
        tau_bh = (z_b * tau_has + (1.0 - z_b) * tau_const) * mask_bh  # (B, H)

        return tau_bh, mask_bh

    def _compute_has_tau_scalar(
        self,
        rho_scalar: float,
        d_int: int,
        device: torch.device,
    ) -> Tensor:
        """Compute per-step tau for a single ``rho`` value (inference).

        Parameters
        ----------
        rho_scalar : float
            Current global-progress value, typically ``(N - j + 1) / N``.
        d_int : int
            Prefix length (fixed for the duration of one inference call).
        device : torch.device
            Device on which to create the output tensor.

        Returns
        -------
        tau : Tensor
            Shape ``(H,)`` — per-step noise levels for this denoising step.
        """
        H = self.config.chunk_size
        alpha = self.config.has_alpha
        u_d_val = self.config.has_u_d

        # All intermediate tensors explicitly on the target device.
        i_h = torch.arange(H, device=device, dtype=torch.float32)  # (H,)
        denom = max(float(H - 1 - d_int), 1.0)
        rel_pos = torch.clamp((i_h - float(d_int)) / denom, min=0.0)  # (H,)
        u_i = (1.0 - rel_pos) ** alpha * u_d_val  # (H,)
        tau = torch.clamp(
            (rho_scalar - u_i) / (1.0 - u_i + _EPS),
            min=0.0,
        )  # (H,)
        # Zero out prefix positions.
        mask = (i_h >= float(d_int)).float()
        return tau * mask  # (H,)

    # ------------------------------------------------------------------
    # Training forward  (overrides PI0Pytorch.forward)
    # ------------------------------------------------------------------

    def forward(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        actions,
        noise=None,
        time=None,
    ) -> tuple[Tensor, Tensor]:
        """FASTER training forward pass with mixed Horizon-Aware Schedule.

        Replaces the scalar global time ``t`` with a per-step noise-level
        vector ``tau ∈ R^H`` sampled via the mixed HAS / constant-schedule
        procedure.  The MSE loss is computed only at valid horizon positions
        (``i >= d``); prefix positions are zeroed out in the returned tensor.

        Parameters
        ----------
        images, img_masks, lang_tokens, lang_masks, state, actions :
            Same as ``PI0Pytorch.forward``.
        noise : Tensor, optional
            Pre-sampled Gaussian noise of shape ``(B, H, max_action_dim)``.
            Sampled internally when ``None``.
        time : ignored
            Accepted for API compatibility; FASTER always samples its own
            schedule parameters.

        Returns
        -------
        losses_masked : Tensor
            Shape ``(B, H, max_action_dim)`` — per-element MSE losses, zeroed
            at prefix positions.
        mask_bh : Tensor
            Shape ``(B, H)`` float — 1 at valid positions, 0 at prefix
            positions.  Used by the policy wrapper for correct loss averaging.
        """
        device = actions.device
        B, H, _ = actions.shape

        if noise is None:
            noise = self.sample_noise(actions.shape, device)

        # ---- Sample HAS parameters (all tensors explicitly on `device`) ----
        rho = torch.rand(B, device=device)
        d = torch.randint(0, self.config.has_d_max + 1, (B,), device=device)
        z = torch.bernoulli(torch.full((B,), self.config.has_mix_prob, device=device))

        # ---- Build per-step tau and valid-position mask ----
        tau_bh, mask_bh = self._compute_has_tau_batch(rho, d, z, device)
        # (B, H, 1) for broadcasting over the action dimension D.
        tau_bhd = tau_bh.unsqueeze(2)

        # ---- Noise injection ----
        x_t = tau_bhd * noise + (1.0 - tau_bhd) * actions  # (B, H, D)
        u_t = noise - actions  # (B, H, D)  — target velocity

        # ---- Embed prefix (images + language) ----
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )

        # ---- Embed suffix (state + noisy actions) with per-step tau ----
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
            self.embed_suffix_faster(state, x_t, tau_bh)
        )

        # bfloat16 cast (mirrors PI0Pytorch.forward)
        if (
            self.paligemma_with_expert.paligemma.model.language_model.layers[0]
            .self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def _forward_fn(pfx, sfx, att4d, pos_ids, adarms):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att4d,
                position_ids=pos_ids,
                past_key_values=None,
                inputs_embeds=[pfx, sfx],
                use_cache=False,
                adarms_cond=[None, adarms],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            _forward_fn,
            prefix_embs,
            suffix_embs,
            att_2d_masks_4d,
            position_ids,
            adarms_cond,
        )

        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        def _proj_fn(x: Tensor) -> Tensor:
            return self.action_out_proj(x)

        v_t = self._apply_checkpoint(_proj_fn, suffix_out)

        # ---- Masked MSE loss: zero out prefix positions ----
        losses = F.mse_loss(u_t, v_t, reduction="none")  # (B, H, D)
        losses_masked = losses * mask_bh.unsqueeze(2)     # zero prefix

        return losses_masked, mask_bh

    # ------------------------------------------------------------------
    # Inference — per-step denoising with HAS tau conditioning
    # ------------------------------------------------------------------

    def denoise_step_faster(
        self,
        state: Tensor,
        prefix_pad_masks: Tensor,
        past_key_values,
        x_t: Tensor,
        tau_per_step: Tensor,
    ) -> Tensor:
        """Single denoising step conditioned on a per-step tau tensor.

        Identical to ``PI0Pytorch.denoise_step`` except it calls
        ``embed_suffix_faster`` so that each horizon position receives its
        own sinusoidal time embedding.

        Parameters
        ----------
        tau_per_step : Tensor
            Shape ``(B, H)`` — per-step noise levels for the current ODE step.
        """
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
            self.embed_suffix_faster(state, x_t, tau_per_step)
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
            batch_size, suffix_len, prefix_len
        )
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        # Deep-copy KV cache to avoid in-place modification across steps.
        past_kv_copy = copy.deepcopy(past_key_values)

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_kv_copy,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)

    @torch.no_grad()
    def sample_actions_faster(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state: Tensor,
        action_prefix: Tensor | None = None,
        noise: Tensor | None = None,
        num_steps: int | None = None,
        execution_horizon: int | None = None,
    ) -> Tensor:
        """Horizon-Aware Schedule inference with optional early stopping.

        Parameters
        ----------
        action_prefix : Tensor, optional
            Shape ``(B, d, action_dim)`` — already-executed action steps that
            are pinned throughout the denoising loop.  When ``None``, ``d = 0``
            and the full chunk is denoised from scratch.
        noise : Tensor, optional
            Initial noise of shape ``(B, H, max_action_dim)``.
        num_steps : int, optional
            Number of Euler steps.  Defaults to ``config.num_inference_steps``.
        execution_horizon : int, optional
            If provided, the loop terminates early once all actions in the
            window ``[d, d + execution_horizon)`` satisfy ``tau == 0``, i.e.
            are fully clean.

        Returns
        -------
        Tensor
            Shape ``(B, H, max_action_dim)`` — predicted clean actions.
        """
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        B = state.shape[0]
        H = self.config.chunk_size
        device = state.device

        d = action_prefix.shape[1] if action_prefix is not None else 0

        if noise is None:
            noise = self.sample_noise((B, H, self.config.max_action_dim), device)

        x_t = noise.clone()

        # Pin prefix actions into the initial trajectory.
        prefix_padded: Tensor | None = None
        if action_prefix is not None:
            prefix_padded = pad_vector(action_prefix, self.config.max_action_dim)
            x_t[:, :d] = prefix_padded

        # ---- Pre-compute VLM KV cache (same as PI0Pytorch.sample_actions) ----
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        N = num_steps

        for j in range(1, N + 1):
            # Current and next global-progress values follow the schedule
            # rho^j = (N - j + 1) / N, starting at 1 and ending at 1/N.
            rho_j = float(N - j + 1) / N
            rho_j1 = float(N - j) / N  # = 0 at the final step j == N

            # Per-step tau tensors on the correct device.
            tau_j = self._compute_has_tau_scalar(rho_j, d, device)    # (H,)
            tau_j1 = self._compute_has_tau_scalar(rho_j1, d, device)  # (H,)

            # ---- Early stopping (C) ----
            # All actions in the execution window are fully denoised (tau == 0)
            # so additional Euler steps cannot improve them.
            if execution_horizon is not None:
                start_idx = d
                end_idx = min(d + execution_horizon, H)
                if end_idx > start_idx and torch.all(tau_j[start_idx:end_idx] == 0.0):
                    break

            # Expand to batch dimension for the denoising call.
            tau_j_b = tau_j.unsqueeze(0).expand(B, H)  # (B, H)

            v_t = self.denoise_step_faster(
                state, prefix_pad_masks, past_key_values, x_t, tau_j_b
            )

            # Euler update with per-step delta_tau.
            delta_tau = tau_j1 - tau_j  # (H,)  — may be negative (noise decreases)
            x_t = x_t + v_t * delta_tau[None, :, None]  # broadcast over B and D

            # Enforce the prefix constraint after every Euler step.
            if prefix_padded is not None:
                x_t[:, :d] = prefix_padded

        return x_t


# ---------------------------------------------------------------------------
# LeRobot policy wrapper
# ---------------------------------------------------------------------------


class FasterPI0Policy(PI0Policy):
    """LeRobot policy wrapper for the FASTER Pi0 model.

    Registers as ``"faster_pi0"`` in the LeRobot policy registry.

    Loading pre-trained Pi0 weights
    --------------------------------
    Because ``FasterPI0Pytorch`` has the exact same architecture as
    ``PI0Pytorch``, you can initialise from a Pi0 checkpoint by passing a
    ``FasterPI0Config`` to ``from_pretrained``::

        config = FasterPI0Config(...)
        policy = FasterPI0Policy.from_pretrained("lerobot/pi0", config=config)

    Training
    ---------
    The ``forward`` method returns a scalar (or per-sample) loss computed only
    over the non-prefix horizon positions so the gradient signal is unaffected
    by the simulated execution delay.
    """

    config_class = FasterPI0Config
    name = "faster_pi0"

    def __init__(self, config: FasterPI0Config, **kwargs):
        # Bypass ``PI0Policy.__init__`` which hard-codes ``PI0Pytorch``.
        # We call ``PreTrainedPolicy.__init__`` directly and then replicate
        # the remaining initialisation with ``FasterPI0Pytorch``.
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config

        self.init_rtc_processor()
        self.model = FasterPI0Pytorch(config, rtc_processor=self.rtc_processor)

        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(config.device)
        self.reset()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def forward(
        self,
        batch: dict[str, Tensor],
        reduction: str = "mean",
    ) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the FASTER loss.

        The MSE loss is averaged exclusively over valid (non-prefix) horizon
        positions.  Prefix positions (``i < d``) are zeroed before averaging so
        that the denominator reflects only truly supervised elements.

        Parameters
        ----------
        batch : dict
            Standard LeRobot training batch.
        reduction : {"mean", "none"}
            ``"mean"`` returns a scalar loss (default, suitable for standard
            training).  ``"none"`` returns per-sample losses of shape ``(B,)``
            for reward-weighted BC or other sample-level weighting schemes.

        Returns
        -------
        loss : Tensor
            Scalar (``reduction="mean"``) or shape ``(B,)``
            (``reduction="none"``).
        loss_dict : dict
            Logging-friendly dict containing ``"loss"`` and
            ``"loss_per_dim"``.
        """
        images, img_masks = self._preprocess_images(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        state = self.prepare_state(batch)
        actions = self.prepare_action(batch)

        # losses_masked: (B, H, max_action_dim), zeroed at prefix positions.
        # mask_bh:       (B, H) float, 1 at valid positions.
        losses_masked, mask_bh = self.model.forward(
            images, img_masks, lang_tokens, lang_masks, state, actions
        )

        # Truncate to the actual output action dimension.
        D = self.config.output_features[ACTION].shape[0]
        losses_masked = losses_masked[:, :, :D]

        # Per-dimension loss for logging (average only over valid elements).
        # mask_bh.sum() is the total number of valid (batch × horizon) pairs.
        valid_bh = mask_bh.sum().clamp(min=1.0)
        loss_per_dim = (losses_masked.sum(dim=(0, 1)) / valid_bh).detach().cpu().numpy().tolist()
        loss_dict = {"loss_per_dim": loss_per_dim}

        if reduction == "none":
            # Per-sample loss: sum over (H, D), divide by each sample's valid count.
            valid_per_sample = mask_bh.sum(dim=1).clamp(min=1.0)  # (B,)
            per_sample_loss = losses_masked.sum(dim=(1, 2)) / (valid_per_sample * D)
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict
        else:
            # Scalar mean over all valid elements in the batch.
            valid_elements = mask_bh.sum() * D
            loss = losses_masked.sum() / valid_elements.clamp(min=1.0)
            loss_dict["loss"] = loss.item()
            return loss, loss_dict

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: dict[str, Tensor],
        **kwargs,
    ) -> Tensor:
        """Predict a full action chunk using the FASTER inference loop.

        In addition to the standard batch, the following optional keyword
        arguments are supported:

        action_prefix : Tensor, optional
            Shape ``(B, d, action_dim)`` — already-executed prefix actions to
            pin during the denoising loop.
        execution_horizon : int, optional
            Stop the denoising loop early once actions
            ``[d, d + execution_horizon)`` are fully clean.
        noise : Tensor, optional
            Pre-sampled initial noise of shape ``(B, H, max_action_dim)``.
        """
        self.eval()

        images, img_masks = self._preprocess_images(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        state = self.prepare_state(batch)

        action_prefix: Tensor | None = kwargs.pop("action_prefix", None)
        execution_horizon: int | None = kwargs.pop("execution_horizon", None)
        noise: Tensor | None = kwargs.pop("noise", None)

        actions = self.model.sample_actions_faster(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            action_prefix=action_prefix,
            noise=noise,
            execution_horizon=execution_horizon,
        )

        # Unpad to the actual output action dimension.
        original_action_dim = self.config.output_features[ACTION].shape[0]
        return actions[:, :, :original_action_dim]

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Return one action step for the current environment observation.

        Uses the action queue from the parent class; the queue is refilled by
        calling ``predict_action_chunk`` (FASTER version) when empty.
        """
        self.eval()
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch, **kwargs)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()
