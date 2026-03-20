from typing import Any


from datasets import load_dataset

# 1. 使用 streaming=True 模式，这不会下载整个数据集
dataset = load_dataset("HuggingFaceVLA/libero", streaming=True, split="train")

# 2. 取前 10 条数据
small_data = list[Any | dict](dataset.take(10))

# 3. 打印第一条看看结构
print(f"成功获取 {len(small_data)} 条数据")
print(small_data[0].keys())