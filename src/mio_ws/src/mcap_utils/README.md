# MCAP 工具

`mcap_utils` 提供两个基于浏览器的本地工具：

- `lerobot-split-mcap`：预览相机画面，在时间轴上添加断点并分割 MCAP。
- `lerobot-convert-mcap-to-v30`：扫描 MCAP topic，并将其映射为 LeRobotDataset v3.0 feature。

两个工具都可以接收单个 `.mcap` 文件或目录；传入目录时会递归查找并按路径排序处理其中的 `.mcap` 文件。Web 服务默认只监听 `127.0.0.1`，数据不会因为打开界面而自动上传。

## 环境准备

在仓库根目录安装锁定的依赖。分割工具只需要基础依赖；转换工具还需要 LeRobot 数据集依赖：

```bash
uv sync --locked                       # 仅使用分割工具
uv sync --locked --extra dataset       # 使用 v3.0 转换工具
```

MCAP 读取、ROS 2 消息解码、图像处理和视频解码所需的 `mcap`、`mcap-ros2-support`、`Pillow` 和 `av` 已列入项目依赖。

## 分割 MCAP

### 启动

处理单个文件：

```bash
uv run lerobot-split-mcap --mcap /path/to/episode.mcap
```

批量处理目录：

```bash
uv run lerobot-split-mcap --mcap /path/to/mcap_directory
```

服务默认使用 `http://127.0.0.1:8766` 并自动打开浏览器。常用选项：

```text
--host HOST          监听地址，默认 127.0.0.1
--port PORT          首选端口，默认 8766；被占用时依次尝试后续端口
--no-browser         不自动打开浏览器
--log-level LEVEL    Python 日志级别，默认 INFO
```

例如，在远程机器上监听所有网卡且不启动浏览器：

```bash
uv run lerobot-split-mcap \
  --mcap /data/recordings \
  --host 0.0.0.0 \
  --port 8766 \
  --no-browser
```

`--host 0.0.0.0` 会允许网络访问，使用前应确认所在网络可信且端口访问受控。

### 操作流程

1. 从检测到的相机 topic 中选择一个作为预览时间轴。
2. 播放或拖动预览，在目标帧处添加一个或多个断点。首帧不能作为断点。
3. 检查每个片段的输出路径；所有路径必须唯一并以 `.mcap` 结尾。
4. 开始分割。输出成功后，批处理会自动进入下一个文件，也可以跳过当前文件。

第一个成功处理的文件会确定本批次默认片段数。后续文件采用不同片段数时，界面会要求再次确认。工具还会在后台预加载后续条目和所选相机，减少批量处理时的等待。

### 输出规则

默认输出位于输入源同级的分片目录中。例如：

```text
/data/open_lid_mcap/open_lid_1/xx.mcap
```

分成 3 段后，建议路径为：

```text
/data/open_lid_mcap_split_1/open_lid_1/xx.mcap
/data/open_lid_mcap_split_2/open_lid_1/xx.mcap
/data/open_lid_mcap_split_3/open_lid_1/xx.mcap
```

断点按消息的 `log_time` 生效。若断点为 `t1`、`t2`，三个片段分别包含：

```text
log_time < t1
t1 <= log_time < t2
log_time >= t2
```

因此，时间戳恰好等于断点的消息会进入后一个片段。输出会保留原 MCAP 的 profile、library、schema、channel metadata、消息字段、metadata 和 attachment，并拒绝产生空片段。

若输出文件已存在，界面会列出冲突并要求确认后才覆盖。写入使用临时文件；发生错误时会删除未完成的输出，并在覆盖场景下恢复原文件。

## 转换为 LeRobotDataset v3.0

### 启动

```bash
uv run lerobot-convert-mcap-to-v30 --mcap /path/to/episode.mcap
```

或递归转换一个目录：

```bash
uv run lerobot-convert-mcap-to-v30 --mcap /path/to/mcap_directory
```

服务默认使用 `http://127.0.0.1:8765` 并自动打开浏览器。它支持与分割工具相同的 `--host`、`--port`、`--no-browser` 和 `--log-level` 选项。

### 数据组织

- 每个 `.mcap` 文件转换为一个 LeRobot episode。
- 文件按递归发现后的排序依次写入数据集。
- 若目录包含 `episode_000000`、`episode_000001` 等目录，扫描结果会提示序号缺失，但不会自动补齐。
- 所选 topic 必须存在于每个文件中，且它们的时间范围必须有交集。

### Topic 映射

扫描完成后，界面会显示每个 topic 的 schema、消息数、频率和探测到的 shape。选择一个或多个源 topic，输入 LeRobot feature 名称，然后创建映射。

常见映射示例：

```text
/camera/front                -> observation.images.front
/joint_states                -> observation.state
/commanded_joint_positions   -> action
```

映射规则：

- 图像 feature 必须且只能映射一个图像 topic。
- 多个数值 topic 可以合并为一个 feature，数据会按界面中的选择顺序展平并拼接为 `float32` 向量。
- 数值 feature 默认使用原始 topic 和从 1 开始的维度索引生成名称。
- 映射卡片中可以逐行修改名称，名称数量必须等于合并后的向量维度，且不能为空或重复。
- 目标 feature 名称必须唯一。
- `timestamp`、`frame_index`、`episode_index`、`index` 和 `task_index` 由 LeRobot 管理，不作为映射目标。

导出时，以设置的 FPS 构建统一时间轴，并为每个 feature 选择时间上最近的消息。允许的最大时间差为 `0.6 / FPS` 秒；找不到足够接近的完整样本时，该帧会被跳过。

### 导出参数

| 参数        | 说明                                                                             |
| ----------- | -------------------------------------------------------------------------------- |
| FPS         | 输出数据集采样率，必须为正整数；界面会根据相机或其他 topic 频率给出建议值。      |
| Repo ID     | LeRobot 数据集标识，例如 `username/dataset_name`；即使不上传也必须填写。         |
| Root        | 本地输出目录，必须是尚不存在的路径。                                             |
| Task        | 写入每一帧的任务描述；存在唯一的 MCAP `episode.task` metadata 时会自动建议该值。 |
| Robot type  | 可选的机器人类型。                                                               |
| Use videos  | 开启时图像 feature 保存为视频，否则保存为逐帧图像。                              |
| Push to Hub | 导出结束后上传到 Hugging Face Hub。                                              |
| Private     | 上传时创建或更新为私有数据集。                                                   |

上传前需要登录 Hugging Face：

```bash
uv run hf auth login
```

转换器不会覆盖已有的 Root。若一次失败留下了不完整目录，请先检查日志和内容，再自行移动或删除该目录后重试。

## 支持的数据

工具通过 `mcap-ros2-support` 解码 ROS 2 消息，并支持常见的原始或压缩图像。原始图像编码包括：

```text
rgb8, bgr8, rgba8, bgra8, mono8, 8UC1, mono16, 16UC1, 32FC1
```

相机预览还支持 FlatBuffers 编码的 `foxglove.CompressedVideo`，视频格式包括 H.264、H.265/HEVC、VP9 和 AV1。实际能否解码还取决于本机 PyAV/FFmpeg 提供的 codec。

数值 feature 支持可转换为数值数组的标量、序列和常见 ROS 结构。转换器也包含对 FlatBuffers `foxglove.JointStates` 和 `foxglove.CompressedVideo` 的处理。未识别的 schema 可能出现在扫描结果中，但只有能够成功解码并转换为图像或数值的数据才能导出。

## 导出后处理

使用 [`lerobot-edit-dataset`](https://huggingface.co/docs/lerobot/using_dataset_tools) 进行后续处理：

- 删除剧集 - 从数据集中移除特定剧集
- 拆分数据集 - 将一个数据集拆分成多个较小的数据集
- 合并数据集 - 将多个数据集合并为一个数据集。数据集必须具有相同的特征，并且剧集将按照 repo_ids 中指定的顺序连接起来。
- 添加特征 - 向数据集添加新特征
- 移除特征 - 从数据集中移除特征
- 转换为视频 - 将基于图像的数据集转换为视频格式，以便高效存储（RGB 和深度摄像头使用单独的编码器进行编码）
- 视频重新编码 - 使用新的编码器设置对现有视频数据集的 RGB 和/或深度流进行重新编码
- 显示数据集信息 - 显示数据集信息的摘要。
  ```bash
  lerobot-edit-dataset \
    --repo_id local/dataset \
    --root path/to/your/dataset \
    --operation.type info \
    --operation.show_features true
  ```
- 修改 `action` 为 `Relative`
  ```bash
  lerobot-edit-dataset \
   --repo_id your_dataset \
   --operation.type recompute_stats \
   --operation.relative_action true \
   --operation.chunk_size 50 \ # 需要与 policy training 匹配
   --operation.relative_exclude_joints "['gripper']"  # 保留绝对控制的关节
  ```

使用 `lerobot-dataset-viz` 预览数据集：

```bash
lerobot-dataset-viz \
 --repo-id local/dataset \
 --root path/to/your/dataset \
 --episode-index 0 \
 --mode local
```

## 常见问题

**页面提示没有找到 MCAP 文件**

确认 `--mcap` 指向 `.mcap` 文件或至少包含一个 `.mcap` 文件的目录。扩展名匹配不区分大小写。

**相机 topic 能被扫描，但不能预览**

扫描阶段主要依据 schema 名称识别图像候选；预览阶段才执行实际解码。检查 schema、消息 encoding、原始图像 encoding，以及本机 FFmpeg 是否支持对应视频 codec。

**转换时报某个文件缺少 topic**

目录模式将每个 MCAP 视为结构一致的 episode。所有映射使用的 topic 都必须出现在每个文件中；可以调整输入目录或减少映射。

**转换后帧数少于预期**

只有所有映射 feature 都能在目标时间点附近找到样本时才会写入该帧。检查 topic 时间范围、时间戳质量和 FPS；过高的 FPS 会缩小最近样本的容许时间差。

**端口已被占用**

工具会从首选端口开始尝试最多 20 个连续端口，实际地址会打印在终端日志中。也可以显式传入其他 `--port`，或使用 `--port 0` 让操作系统分配端口。

## 测试

在仓库根目录运行 MCAP 工具测试：

```bash
uv run pytest tests/mio_ws/mcap_utils -svv
```

代码入口：

- `common.py`：文件发现、MCAP 扫描、图像与视频预览解码。
- `splitter.py`：分割校验、MCAP 记录复制和原子化输出。
- `mcap_split/server.py`：分割工具的本地 HTTP 服务与会话状态。
- `mcap_to_v30/converter.py`：topic 扫描、特征推断、时间对齐和数据集写入。
- `mcap_to_v30/server.py`：转换工具的本地 HTTP 服务。
