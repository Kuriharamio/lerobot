import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi0 import PI0Policy
from lerobot.policies.pi05 import PI05Policy
from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy

def get_policy(model: str, compile: bool=True, compile_mode: str="max-autotune", amp: bool=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model == "pi0":
        model_id = "lerobot/pi0_libero_finetuned"
        policy_cfg = PreTrainedConfig.from_pretrained(model_id)
        policy_cfg.pretrained_path = model_id
        policy_cfg.compile_model = compile
        policy_cfg.compile_mode = compile_mode
        policy_cfg.use_amp = amp
        policy = PI0Policy.from_pretrained(model_id, config=policy_cfg)

    elif model == "pi05":
        model_id = "lerobot/pi05_libero_finetuned"
        policy_cfg = PreTrainedConfig.from_pretrained(model_id)
        policy_cfg.pretrained_path = model_id
        policy_cfg.compile_model = compile
        policy_cfg.compile_mode = compile_mode
        policy_cfg.use_amp = amp
        policy = PI05Policy.from_pretrained(model_id, config=policy_cfg)

    elif model == "pi0fast":
        model_id = "lerobot/pi0fast-libero"
        policy_cfg = PreTrainedConfig.from_pretrained(model_id)
        policy_cfg.pretrained_path = model_id
        policy_cfg.compile_model = compile
        policy_cfg.compile_mode = compile_mode
        policy_cfg.use_kv_cache = not compile
        policy_cfg.use_amp = amp
        policy = PI0FastPolicy.from_pretrained(model_id, config=policy_cfg)
    else:
        raise ValueError(f"Model {model} not supported")

    # if dtype == "bfloat16":
    #     policy.to(torch.bfloat16)
    # elif dtype == "float32":
    #     policy.to(torch.float32)
    policy = policy.to(device).eval()
    return policy, model_id

def format_batch(batch_dict):
    for k, v in batch_dict.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            batch_dict[k] = v.to(torch.bfloat16)
    return batch_dict