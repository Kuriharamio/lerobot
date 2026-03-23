import sys
import os
import argparse
from contextlib import nullcontext
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors

# import torch._inductor.config
# torch._inductor.config.coordinate_descent_tuning = True
# torch._inductor.config.layout_optimization = False
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True

MODEL_CHOICES = ["pi0", "pi05", "pi0fast"]
DATASET_REPO_ID = "lerobot/libero"
EPISODE_INDEX = 0
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, required=True, choices=MODEL_CHOICES)
parser.add_argument("-t", "--tag", type=str, required=True)
args = parser.parse_args()

WORKSPACE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(WORKSPACE_PATH)
from utils.action_analysis import action_analysis
from utils.time_analysis import get_time, time_analysis
from utils.print_color import print_green, print_yellow, print_red
from utils.inference_utils import get_policy, format_batch

WARMUP = True
PROFILE = True
ANALYSIS = True

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_path = f"{WORKSPACE_PATH}/logs/{args.model}/{args.tag}"

    print_green("Load policy...")
    policy, model_id = get_policy(model=args.model, compile=True, compile_mode="max-autotune", amp=True)
    amp_ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16) if device.type == "cuda" and policy.config.use_amp else nullcontext()
    total_params = sum(p.numel() for p in policy.parameters())
    print_yellow(f"[*] Total Parameters: {total_params / 1e6:.2f} M")
    print_yellow(f"[*] Compile: {policy.config.compile_model} (Mode: {policy.config.compile_mode})")
    print_yellow(f"[*] amp: {policy.config.use_amp}")

    preprocess, postprocess = make_pre_post_processors(policy.config, model_id, preprocessor_overrides={"device_processor": {"device": str(device)}})
    dataset = LeRobotDataset(DATASET_REPO_ID, video_backend="pyav")
    from_index = dataset.meta.episodes["dataset_from_index"][EPISODE_INDEX]
    to_index = dataset.meta.episodes["dataset_to_index"][EPISODE_INDEX]

    gt_actions_list = []
    pred_actions_list = []
    total_prep_time = 0.0
    total_inf_time = 0.0
    total_post_time = 0.0
    total_pipeline_time = 0.0
    num_frames = 0

    print_green("Load data to memory...")
    memory_frames = []
    for i in range(from_index, to_index):
        frame = dict(dataset[i])
        pinned_frame = {k: (v.pin_memory() if isinstance(v, torch.Tensor) else v) 
                        for k, v in frame.items()}
        memory_frames.append(pinned_frame)

    if WARMUP:
        print_green("Warmup...")
        with torch.inference_mode(), amp_ctx:
            dummy_frame = memory_frames[0]
            for _ in range(5 if PROFILE or ANALYSIS else 1):
                batch = preprocess(dummy_frame)
                _ = policy.select_action(batch)
                policy.reset()
        torch.cuda.synchronize() 

    if PROFILE:
        print_green("Profile...")
        policy.reset()
        batch = preprocess(memory_frames[0])
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],  # 分析 CPU 和 CUDA 活动
            schedule=torch.profiler.schedule(
                wait=1,     # 前1步不采样
                warmup=3,   # 第2步作为热身，不计入结果
                active=1,   # 采集后面3步的性能数据
                repeat=2),  # 重复2轮
            on_trace_ready=torch.profiler.tensorboard_trace_handler(output_path),  # 保存日志以供 TensorBoard 可视化
            record_shapes=True,     # 记录输入张量的形状
            profile_memory=True,    # 分析内存分配
            with_stack=True,        # 记录操作的调用堆栈信息
        ) as profiler, torch.inference_mode(), amp_ctx:
            # for frame in memory_frames[0:min(10, len(memory_frames))]:
            for _ in range(10):
                # batch = preprocess(frame)
                pred_action = policy.select_action(batch)
                # pred_action = postprocess(pred_action)
                profiler.step()
                policy.reset()         
        torch.cuda.synchronize() 

    if ANALYSIS:
        print_green("Time Consumption Analysis...")
        policy.reset()
        with torch.inference_mode(), amp_ctx:
            for frame in memory_frames[0:min(50, len(memory_frames))]:
                t0 = get_time()
                batch = preprocess(frame)
                t1 = get_time()
                pred_action = policy.select_action(batch)
                t2 = get_time()
                pred_action = postprocess(pred_action)
                t3 = get_time()

                policy.reset()
                total_prep_time += t1 - t0
                total_inf_time += t2 - t1
                total_post_time += t3 - t2
                frame_time = t3 - t0
                total_pipeline_time += frame_time
                num_frames += 1
                print_yellow(f"Frame {num_frames} processed in {frame_time * 1000:.2f} ms")
        torch.cuda.synchronize() 

        print_green("Model Output Analysis...")
        policy.reset()
        with torch.inference_mode(), amp_ctx:
            for frame in memory_frames:
                batch = preprocess(frame)
                pred_action = policy.select_action(batch)
                pred_action = postprocess(pred_action)
                gt_actions_list.append(frame["action"].squeeze().cpu().numpy())
                pred_actions_list.append(pred_action.squeeze().cpu().numpy())
        torch.cuda.synchronize() 
        
        action_analysis(gt_actions_list, pred_actions_list, output_path)
        time_analysis(total_prep_time, total_inf_time, total_post_time, total_pipeline_time, num_frames, output_path)


if __name__ == "__main__":
    main()
