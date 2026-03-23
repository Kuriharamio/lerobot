import torch
import time

def get_time():
    torch.cuda.synchronize()
    return time.perf_counter()

def time_analysis(total_prep_time, total_inf_time, total_post_time, total_pipeline_time, num_frames, save_path):
    if num_frames > 0:
        avg_prep_t = total_prep_time / num_frames
        avg_inf_t = total_inf_time / num_frames
        avg_post_t = total_post_time / num_frames
        avg_pipeline_t = total_pipeline_time / num_frames

        avg_prep_hz = 1.0 / avg_prep_t if avg_prep_t > 0 else 0
        avg_inf_hz = 1.0 / avg_inf_t if avg_inf_t > 0 else 0
        avg_post_hz = 1.0 / avg_post_t if avg_post_t > 0 else 0
        avg_pipeline_hz = 1.0 / avg_pipeline_t if avg_pipeline_t > 0 else 0

        print("\n" + "="*55)
        print(f"推理时间 (共处理帧数: {num_frames})")
        print("-" * 55)
        print(f"{'阶段':<13} | {'平均耗时 (ms)':<12} | {'平均频率 (Hz)':<14}")
        print("-" * 55)
        print(f"{'Preprocess':<15} | {avg_prep_t * 1000:<15.2f} | {avg_prep_hz:<15.2f}")
        print(f"{'Inference':<15} | {avg_inf_t * 1000:<15.2f} | {avg_inf_hz:<15.2f}")
        print(f"{'Postprocess':<15} | {avg_post_t * 1000:<15.2f} | {avg_post_hz:<15.2f}")
        print("-" * 55)
        print(f"{'Total Pipeline':<15} | {avg_pipeline_t * 1000:<15.2f} | {avg_pipeline_hz:<15.2f}")
        print("="*55 + "\n")

        # 保存为 txt
        with open(f"{save_path}/time_analysis.txt", "w") as f:
            f.write(f"推理时间 (共处理帧数: {num_frames})\n")
            f.write("-" * 55 + "\n")
            f.write(f"{'阶段':<13} | {'平均耗时 (ms)':<12} | {'平均频率 (Hz)':<14}\n")
            f.write("-" * 55 + "\n")
            f.write(f"{'Preprocess':<15} | {avg_prep_t * 1000:<15.2f} | {avg_prep_hz:<15.2f}\n")
            f.write(f"{'Inference':<15} | {avg_inf_t * 1000:<15.2f} | {avg_inf_hz:<15.2f}\n")
            f.write(f"{'Postprocess':<15} | {avg_post_t * 1000:<15.2f} | {avg_post_hz:<15.2f}\n")
            f.write("-" * 55 + "\n")
            f.write(f"{'Total Pipeline':<15} | {avg_pipeline_t * 1000:<15.2f} | {avg_pipeline_hz:<15.2f}\n")
            f.write("="*55 + "\n")