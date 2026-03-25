import types
import torch
from lerobot.configs.policies import PreTrainedConfig

from lerobot.policies.pi0 import PI0Policy
from lerobot.policies.pi05 import PI05Policy

from lerobot.policies.pi0.modeling_pi0 import make_att_2d_masks as make_att_2d_masks_pi0
from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks as make_att_2d_masks_pi05
from lerobot.policies.pi0.modeling_pi0 import ActionSelectKwargs as ActionSelectKwargs_pi0
from lerobot.policies.pi05.modeling_pi05 import ActionSelectKwargs as ActionSelectKwargs_pi05
from typing import Unpack


def denoise_step_pi0(
    self,
    state,
    prefix_pad_masks,
    past_key_values,
    x_t,
    timestep,
):
    """Apply one denoising step of the noise `x_t` at a given timestep."""
    suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

    suffix_len = suffix_pad_masks.shape[1]
    batch_size = prefix_pad_masks.shape[0]
    prefix_len = prefix_pad_masks.shape[1]

    prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
    suffix_att_2d_masks = make_att_2d_masks_pi0(suffix_pad_masks, suffix_att_masks)
    full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

    prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
    position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

    full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
    self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

    from transformers.cache_utils import DynamicCache
    step_cache = DynamicCache(ddp_cache_data=[(k, v, sw) for k, v, sw in past_key_values])
    outputs_embeds, _ = self.paligemma_with_expert.forward(
        attention_mask=full_att_2d_masks_4d,
        position_ids=position_ids,
        past_key_values=step_cache,
        inputs_embeds=[None, suffix_embs],
        use_cache=False,
        adarms_cond=[None, adarms_cond],
    )

    suffix_out = outputs_embeds[1]
    suffix_out = suffix_out[:, -self.config.chunk_size :]
    suffix_out = suffix_out.to(dtype=torch.float32)
    return self.action_out_proj(suffix_out)

def denoise_step_pi05(
    self,
    prefix_pad_masks,
    past_key_values,
    x_t,
    timestep,
):
    """Apply one denoising step of the noise `x_t` at a given timestep."""
    suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, timestep)

    suffix_len = suffix_pad_masks.shape[1]
    batch_size = prefix_pad_masks.shape[0]
    prefix_len = prefix_pad_masks.shape[1]

    prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
    suffix_att_2d_masks = make_att_2d_masks_pi05(suffix_pad_masks, suffix_att_masks)
    full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

    prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
    position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

    full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
    self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

    from transformers.cache_utils import DynamicCache
    step_cache = DynamicCache(ddp_cache_data=[(k, v, sw) for k, v, sw in past_key_values])
    outputs_embeds, _ = self.paligemma_with_expert.forward(
        attention_mask=full_att_2d_masks_4d,
        position_ids=position_ids,
        past_key_values=step_cache,
        inputs_embeds=[None, suffix_embs],
        use_cache=False,
        adarms_cond=[None, adarms_cond],
    )

    suffix_out = outputs_embeds[1]
    suffix_out = suffix_out[:, -self.config.chunk_size :]
    suffix_out = suffix_out.to(dtype=torch.float32)
    return self.action_out_proj(suffix_out)


@torch.no_grad()  # see openpi `sample_actions` (slightly adapted)
def sample_actions_pi0(
    self,
    images,
    img_masks,
    lang_tokens,
    lang_masks,
    state,
    noise=None,
    num_steps=None,
    **kwargs: Unpack[ActionSelectKwargs_pi0],
) -> torch.Tensor:
    """Do a full inference forward and compute the action."""
    if num_steps is None:
        num_steps = self.config.num_inference_steps

    bsize = state.shape[0]
    device = state.device

    if noise is None:
        # Sample noise with padded dimension as expected by action_in_proj
        actions_shape = (
            bsize,
            self.config.chunk_size,
            self.config.max_action_dim,
        )  # Use config max_action_dim for internal processing
        noise = self.sample_noise(actions_shape, device)

    prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
        images, img_masks, lang_tokens, lang_masks
    )
    prefix_att_2d_masks = make_att_2d_masks_pi0(prefix_pad_masks, prefix_att_masks)
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

    dt = -1.0 / num_steps

    x_t = noise

    steps_tensor = torch.arange(num_steps, dtype=torch.float32, device=device)
    time_schedule = 1.0 + steps_tensor * dt

    for step in range(num_steps):
        time = 1.0 + step * dt
        time_tensor = time_schedule[step].expand(bsize)

        def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
            return self.denoise_step(
                state=state,
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=input_x_t,
                timestep=current_timestep,
            )

        if self._rtc_enabled():
            inference_delay = kwargs.get("inference_delay")
            prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
            execution_horizon = kwargs.get("execution_horizon")

            v_t = self.rtc_processor.denoise_step(
                x_t=x_t,
                prev_chunk_left_over=prev_chunk_left_over,
                inference_delay=inference_delay,
                time=time,
                original_denoise_step_partial=denoise_step_partial_call,
                execution_horizon=execution_horizon,
            )
        else:
            v_t = denoise_step_partial_call(x_t)

        x_t = x_t + dt * v_t

        if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
            self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)

    return x_t

@torch.no_grad()  # see openpi `sample_actions` (slightly adapted)
def sample_actions_pi05(
    self,
    images,
    img_masks,
    tokens,
    masks,
    noise=None,
    num_steps=None,
    **kwargs: Unpack[ActionSelectKwargs_pi05],
) -> torch.Tensor:
    """Do a full inference forward and compute the action."""
    if num_steps is None:
        num_steps = self.config.num_inference_steps

    bsize = tokens.shape[0]
    device = tokens.device

    if noise is None:
        # Sample noise with padded dimension as expected by action_in_proj
        actions_shape = (
            bsize,
            self.config.chunk_size,
            self.config.max_action_dim,
        )  # Use config max_action_dim for internal processing
        noise = self.sample_noise(actions_shape, device)

    prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
    prefix_att_2d_masks = make_att_2d_masks_pi05(prefix_pad_masks, prefix_att_masks)
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

    dt = -1.0 / num_steps

    x_t = noise

    steps_tensor = torch.arange(num_steps, dtype=torch.float32, device=device)
    time_schedule = 1.0 + steps_tensor * dt

    for step in range(num_steps):
        time = 1.0 + step * dt
        time_tensor = time_schedule[step].expand(bsize)

        def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
            return self.denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=input_x_t,
                timestep=current_timestep,
            )

        if self._rtc_enabled():
            inference_delay = kwargs.get("inference_delay")
            prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
            execution_horizon = kwargs.get("execution_horizon")

            v_t = self.rtc_processor.denoise_step(
                x_t=x_t,
                prev_chunk_left_over=prev_chunk_left_over,
                inference_delay=inference_delay,
                time=time,
                original_denoise_step_partial=denoise_step_partial_call,
                execution_horizon=execution_horizon,
            )
        else:
            v_t = denoise_step_partial_call(x_t)

        x_t = x_t + dt * v_t

        if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
            self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)

    return x_t


def get_policy(model: str, compile: bool=True, compile_mode: str="max-autotune", amp: bool=True, replace_method: bool=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model == "pi0":
        model_id = "lerobot/pi0_libero_finetuned"
        policy_cfg = PreTrainedConfig.from_pretrained(model_id)
        policy_cfg.pretrained_path = model_id
        policy_cfg.compile_model = compile
        policy_cfg.compile_mode = compile_mode
        policy_cfg.use_amp = amp
        policy = PI0Policy.from_pretrained(model_id, config=policy_cfg)
        if replace_method:
            policy.model.denoise_step = types.MethodType(denoise_step_pi0, policy.model)
            policy.model.sample_actions = types.MethodType(sample_actions_pi0, policy.model)
            if compile:
                policy.model.sample_actions = torch.compile(policy.model.sample_actions, mode=compile_mode)
    elif model == "pi05":
        model_id = "lerobot/pi05_libero_finetuned"
        policy_cfg = PreTrainedConfig.from_pretrained(model_id)
        policy_cfg.pretrained_path = model_id
        policy_cfg.compile_model = compile
        policy_cfg.compile_mode = compile_mode
        policy_cfg.use_amp = amp
        policy = PI05Policy.from_pretrained(model_id, config=policy_cfg)
        if replace_method:
            policy.model.denoise_step = types.MethodType(denoise_step_pi05, policy.model)
            policy.model.sample_actions = types.MethodType(sample_actions_pi05, policy.model)
            if compile:
                policy.model.sample_actions = torch.compile(policy.model.sample_actions, mode=compile_mode)
    else:
        raise ValueError(f"Model '{model}' not supported. Choose from: pi0, pi05")

    policy = policy.to(device).eval()
    return policy, model_id


