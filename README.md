# 智能图像自动裁剪与美学优化系统

基于 **YOLOv8 目标检测** 与 **CLIP 美学打分** 的智能图片裁剪系统，支持批量与单图裁剪，融合多模型能力，适用于图片美学增强、自动化图片处理等场景。

提供 **Web GUI 可视化界面** 与 **命令行接口**。

---

## 项目目录结构

```plaintext
.
├── config.fast.yaml
├── config.gaic.yaml
├── config.ultrafast.yaml
├── config.yaml
├── docs/
│   └── CHANGELOG.md
├── download_models.py
├── eval/
│   ├── baseline_compare.py
│   ├── evaluate.py
│   └── ...
├── generate_annotations.py
├── gui/
│   ├── app.py
│   ├── static/
│   └── templates/
├── models/
│   └── gaic_pairwise_final.json
├── outputs/
├── requirements.txt
├── run_final_gaic.ps1
├── setup_project.py
├── src/
│   ├── pipeline.py
│   ├── fusion.py
│   ├── u2net_model.py
│   └── ...
└── version3-CHANGES.md
```

---

# 快速开始

## 1. 环境准备

创建并激活虚拟环境：

```bash
conda create -n cv_env python=3.10 -y
conda activate cv_env
```

安装依赖：

```bash
pip install -r requirements.txt
```

如果出现缺失 `clip` 模块问题，可执行：

```bash
pip install git+https://github.com/openai/CLIP.git
```

---

## 2. 下载预训练模型

运行：

```bash
python download_models.py
```

所有模型权重将自动下载到：

```plaintext
models/
```

---

## 3. 启动 Web 可视化界面

运行：

```bash
python gui/app.py
```

浏览器访问：

```plaintext
http://localhost:5000
```

---

## 4. 命令行批量裁剪

运行：

```bash
python eval/baseline_compare.py \
    --config config.gaic.yaml \
    --input_dir ./your_images \
    --output_dir ./outputs
```

示例目录：

```plaintext
your_images/
├── img1.jpg
├── img2.png
└── img3.jpeg
```

输出结果：

```plaintext
outputs/
├── img1_crop.jpg
├── img2_crop.jpg
└── img3_crop.jpg
```

---

## 5. 命令行单图裁剪

运行：

```bash
python eval/evaluate.py \
    --config config.gaic.yaml \
    --image_path ./your_images/xxx.jpg \
    --output_path ./outputs/xxx_crop.jpg
```

---

# 配置文件说明（config.gaic.yaml）

示例配置：

```yaml
confidence_threshold: 0.25

aesthetic_weight: 0.7

positive_prompt: "a beautiful photo, well-composed"

negative_prompt: "blurry, poorly composed"

fusion:
  yolo_weight: 0.5
  clip_weight: 0.5
```

参数说明：

| 参数                     | 说明             |
| ---------------------- | -------------- |
| `confidence_threshold` | YOLO 目标检测置信度阈值 |
| `aesthetic_weight`     | 美学评分权重         |
| `positive_prompt`      | CLIP 正向美学提示词   |
| `negative_prompt`      | CLIP 负向美学提示词   |
| `fusion.yolo_weight`   | YOLO 分数融合权重    |
| `fusion.clip_weight`   | CLIP 分数融合权重    |

---

# 主要功能

✅ 智能目标检测与裁剪（YOLOv8）

✅ CLIP 美学打分与裁剪优化

✅ 多模型分数融合裁剪策略

✅ 支持批量图片处理

✅ 支持单张图片处理

✅ Web 可视化界面

✅ 命令行自动化处理

✅ 自定义配置与裁剪权重

---

# 核心模块说明

| 模块                   | 功能         |
| -------------------- | ---------- |
| `src/pipeline.py`    | 核心裁剪与融合流程  |
| `src/fusion.py`      | 多模型分数融合逻辑  |
| `src/u2net_model.py` | 显著性检测模型    |
| `gui/app.py`         | Web 服务启动入口 |
| `requirements.txt`   | 项目完整依赖列表   |

---

# 输出结果

默认输出目录：

```plaintext
outputs/
```

裁剪结果示例：

```plaintext
outputs/
└── xxx_crop.jpg
```

---

# 变更日志

查看：

```plaintext
docs/CHANGELOG.md
version3-CHANGES.md
```

---

# 致谢

本项目基于以下优秀开源项目：

* [YOLOv8](https://github.com/ultralytics/ultralytics)
  Ultralytics 提供的目标检测与视觉任务框架。

* [CLIP](https://github.com/openai/CLIP)
  OpenAI 提出的视觉—语言联合表示模型。

* [U²-Net](https://github.com/xuebinqin/U-2-Net)
  用于显著性目标检测的深度网络模型。

感谢开源社区提供支持。

---


---

# License

仅供学习与研究使用，如用于商业场景请遵循对应模型与依赖项目协议。
