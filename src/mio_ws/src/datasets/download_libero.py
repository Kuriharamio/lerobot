from pathlib import Path
from typing import Any

from datasets import load_dataset

WORKSPACE = Path(__file__).resolve().parents[4]  # .../lerobot
DATASET_REPO_ID = "HuggingFaceVLA/libero"
DATASET_ROOT = WORKSPACE / "src" / "mio_ws" / "datasets" / "libero"

# 使用与 run_inference.py 一致的本地目录
DATASET_ROOT.mkdir(parents=True, exist_ok=True)

# 1. 使用 streaming=True 模式（不会下载整个数据集），并把缓存放到 DATASET_ROOT
dataset = load_dataset(
    DATASET_REPO_ID,
    streaming=True,
    split="train",
    cache_dir=str(DATASET_ROOT),
)

# 2. 取前 10 条数据
small_data = list[Any | dict](dataset.take(10))

# 3. 打印结构
print(f"数据集: {DATASET_REPO_ID}")
print(f"缓存目录: {DATASET_ROOT}")
print(f"成功获取 {len(small_data)} 条数据")
print(small_data[0].keys())