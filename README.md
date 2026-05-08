# smart-framing

智能取景（Intelligent Cropping）课程项目实现：
给定信息丰富的场景图像，自动返回“最具美感”的取景坐标，并导出可视化结果与裁剪图。

## 1. 方法设计

本仓库使用“候选框生成 + 美感评分模型”的可解释方案：

1. **候选框生成**：多尺度、多纵横比滑窗，限制最大候选数量以保证速度。
2. **特征提取**：
   - 显著性强度（saliency mean）
   - 三分法构图对齐（thirds alignment）
   - 边缘密度（edge density）
   - 色彩变化（color variance）
   - 灰度熵（texture entropy）
   - 尺度占比（size ratio）
3. **美感评分**：
   - 无训练权重：使用经验权重直接推理；
   - 可训练模式：使用闭式解岭回归训练线性评分器（可复现、速度快）。
4. **输出**：返回最佳框 `bbox=(x1,y1,x2,y2)`、美感分数、各特征分。

## 2. 指标支持

- **美感评分预测**：模型输出 `aesthetic_score`
- **与真实取景区域 IoU**：`bbox_iou(pred_bbox, gt_bbox)`

在 `train_eval.py` 中默认输出：
- `mIoU`
- `mean_pred_score`

## 3. 数据格式（可复现实验）

准备标注文件 `annotations.json`（列表）示例：

```json
[
  {"image": "img001.jpg", "bbox": [120, 60, 720, 480], "score": 0.88},
  {"image": "img002.jpg", "bbox": [40, 100, 560, 420], "score": 0.80}
]
```

字段说明：
- `image`: 图像文件名（相对于 `--image-root`）
- `bbox`: 人工真实取景框
- `score`: 人工美感分（0~1，可选，默认 0.8）

## 4. 训练与评估

```bash
python train_eval.py --annotations ./annotations.json --image-root ./images --model-out ./model_weights.npz
```

说明：
- 按样本顺序 8:2 切分 train/val；
- 训练目标：`0.65 * IoU + 0.35 * score`；
- 模型：线性岭回归（闭式解），固定随机性，实验可重复。

## 5. GUI/可视化程序

```bash
python gui.py --image ./images/img001.jpg --model ./model_weights.npz --out ./framing_result.jpg
```

输出：
- `framing_result.jpg`：带预测框的原图
- `framing_result_crop.jpg`：最佳取景裁剪图
- 控制台打印 `bbox`、`score`、`feature_scores`

> 若未训练模型，可省略 `--model`，系统使用内置经验权重。

## 6. 性能建议

- 调小 `max_candidates` 可提速（在 `SmartFramer.generate_candidates` 中）。
- 可先将大图等比缩放到较短边 720~1080 再预测，最后映射回原图坐标。

## 7. 反抄袭说明

本实现为从零编写的可解释算法流程，非教程代码复制、非直接搬运开源仓库。
