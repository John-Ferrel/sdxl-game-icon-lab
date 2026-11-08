# 安装与配置：先知道要下载什么

如果只想看结果，回到 [README](../README.md) 即可，不需要GPU。下面是Windows + NVIDIA的本地运行路线。本次在已有环境中验证独立仓库脚本；没有在第二台干净机器重装一遍。

## 下载预算

| 内容 | 本项目使用的版本 | 大小与来源 |
|---|---|---|
| 示例素材 | Kenney Generic Items 1.0，CC0 | 原包2,278,836 bytes，约2.17MiB；仓库已经附40张选图，无需再下载全包 |
| SDXL基础权重 | Base1.0，固定revision | `sd_xl_base_1.0.safetensors`，6,938,078,334 bytes，约6.46GiB；从[官方模型页](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)取得并查看OpenRAIL++许可 |
| ComfyUI源码 | v0.3.26，固定commit | 从[官方仓库](https://github.com/Comfy-Org/ComfyUI)克隆；不包含基础权重 |
| Python依赖 | PyTorch2.6.0/CUDA12.4及requirements.txt | CUDA轮子及依赖合计可达数GB，随平台而异；不能把2.17MiB素材包当作总安装成本 |
| 可选Canny控制权重 | SDXL Control-LoRA Canny rank128 | 395,733,680 bytes，约377.4MiB；[Stability AI官方仓库](https://huggingface.co/stabilityai/control-lora)，另有独立许可；不属于必需训练下载 |

已有同一模型文件时可以直接指向原文件，不必复制或重复下载。项目脚本不会自动下载模型、修改驱动、安装全局Python依赖或上传内容。

全新安装的总下载可能超过8GB。仓库本身与40张选图很小，主要成本来自基础模型和CUDA依赖；先复用已有文件会省去这部分下载。

## 路线A：已经有可用环境

先确认解释器能运行ComfyUI，以及有 `requirements.txt` 中的训练包。不要为了这个示例直接升级一个正常工作的环境。把本仓库放在任意独立目录，复制配置：

```powershell
Copy-Item config.example.json config.local.json
```

在 `config.local.json` 填入自己的ComfyUI源码目录和基础权重完整路径。可以填仓库相对路径或绝对路径，JSON中的Windows反斜杠要写成`\\`；写成正斜杠更简单：

```json
{
  "comfyui_dir": "C:/my-ai/ComfyUI",
  "checkpoint": "C:/my-models/sd_xl_base_1.0.safetensors"
}
```

这只是格式示例，不是声称你的电脑有这些目录。所有命令中的 `.venv/Scripts/python.exe` 都可以替换成自己已验证的解释器路径。缓存、训练和推理必须使用包含相应依赖的环境；仅有ComfyUI不等于已经装了训练包。

## 路线B：从独立虚拟环境开始

先准备Python3.12、Git和能正常识别NVIDIA显卡的驱动。以下命令会下载依赖，请先看上面的大小清单。

```powershell
py -3.12 -m venv .venv

# 本项目选择旧版固定组合，便于复核；不是要求所有新项目都使用旧版本。
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

git clone https://github.com/Comfy-Org/ComfyUI.git tools/ComfyUI
git -C tools/ComfyUI checkout 9aac21f894a122ddb8d825c57ad61c0db5e630db

.\.venv\Scripts\python.exe -m pip install -r tools/ComfyUI/requirements.txt -r requirements.txt
# 对齐本次实际使用的前端；该旧版源码自己的requirements锁了更早前端。
.\.venv\Scripts\python.exe -m pip install comfyui-frontend-package==1.12.8
.\.venv\Scripts\python.exe -m pip check
Copy-Item config.example.json config.local.json
```

把基础模型文件放到配置中指定的位置。这轮固定文件的SHA-256是：

```text
31e35c80fc4829d14f90153f4c74cd59c90b779f6afe05a74cd6120b893f7e5b
```

脚本会检查它，避免无意中拿另一家族或另一版本继续训练。`configs/base/` 只包含对应架构与噪声调度配置，不包含训练权重。

检查GPU：

```powershell
nvidia-smi
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

本项目预期显示 `2.6.0+cu124` 且CUDA可用。在本机没有为这轮训练单独安装CUDA Toolkit；PyTorch使用自己的运行时，驱动仍需兼容。遇到问题先对照[PyTorch官方2.6安装说明](https://pytorch.org/get-started/previous-versions/#v260)与[ComfyUI手动安装说明](https://docs.comfy.org/installation/manual_install)，不要把改驱动当作第一步。

## 第一次运行

按顺序：

```powershell
.\.venv\Scripts\python.exe -B scripts/prepare_data.py
.\.venv\Scripts\python.exe -B scripts/serve.py
```

浏览器打开 http://127.0.0.1:8190 ，拖入 `workflows/01_base.ui.json`。暂时没有LoRA文件时先用Base，这是正常的。训练步骤见 [README](../README.md#四步训练流程)。

## 可选：Canny结构控制

先理解它的用途：Canny提取边缘，ControlNet把边缘作为生成条件；它与学习画风的LoRA不是一回事。这里提供接口示例，没有把它加入本轮LoRA效果对照，也没有宣称新数据已完成ControlNet评测。

如果要用 `03_canny_optional.ui.json`，取得官方SDXL兼容的 `control-lora-canny-rank128.safetensors`，在local配置增加路径：

```json
{
  "comfyui_dir": "tools/ComfyUI",
  "checkpoint": "tools/ComfyUI/models/checkpoints/sd_xl_base_1.0.safetensors",
  "controlnet": "C:/my-models/control-lora-canny-rank128.safetensors"
}
```

重新启动 `serve.py`，加载工作流。参考图是仓库内 `data/input/reference.png`，来自同一CC0素材。模型名含`control-lora`，指预训练结构控制权重，不是本课训练的风格适配器。

## 运行与公开的边界

`config.local.json`、`runs/`、虚拟环境、缓存、权重及恢复文件被Git忽略。公开结果通过 `scripts/export_results.py` 选择性导出：只复制本轮公开数据训练的PNG和脱离机器路径的摘要，保留全部对照，不只挑好图。

重新做一轮时，可在local配置加入 `"run_dir": "runs/second-try"`，保留上一轮。模型、数据或配置变动后不沿用旧缓存或同名已完成实验。
