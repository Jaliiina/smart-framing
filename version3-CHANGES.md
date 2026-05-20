# 本次修改记录

日期：2026-05-20


## 概要
为了解决系统偏向选取接近整图的裁剪框问题，本次对配置与融合逻辑做了调整：通过收缩候选框尺度范围、减少候选数，以及在融合阶段对大面积候选施加惩罚并收紧回退条件，降低“取出整图”现象的发生概率。

## 修改的文件

- [config.yaml](config.yaml)
  - `candidate_generation.area_ratios`：从 `[0.35, 0.45, 0.60, 0.75, 0.90]` 调整为 `[0.25, 0.40, 0.55, 0.70]`，优先生成中小尺度候选。
  - `candidate_generation.top_k`：从 `150` 降到 `100`，减少候选数量以降低噪音。
  - `candidate_generation.max_area_ratio`：从 `0.95` 改为 `0.80`，直接过滤接近整图的大候选。
  - `fusion.low_score_threshold`：初步从 `0.30` 改为 `0.15`（后续可调）。
  - 新增融合惩罚参数（`fusion` 下）：
    - `area_penalty_factor: 0.30`
    - `area_penalty_power: 1.0`
    - `area_penalty_min: 0.5`
    这些参数用于按候选面积对最终分数做下调，避免系统偏好大框。

- [src/fusion.py](src/fusion.py)
  - 在归一化并加权融合各维度分数后，增加“面积惩罚”机制：
    - 计算每个候选的归一化面积 norm_area = area / max_area。
    - 计算惩罚因子 penalty = clip(1 - area_penalty_factor * norm_area**area_penalty_power, area_penalty_min, 1.0)。
    - 将最终分数乘以 penalty，从而降低大面积候选的优先级。
  - 收紧低分回退条件：只有当最大的候选的最终分数明显高于当前最优（> 1.05×）且该候选的归一化面积小于 0.95 时，才允许将最优替换为最大框（避免“回退选整图”）。

## 变更理由
- 配置层面：直接限制生成靠近整图的候选可以根本减少该类候选被考虑的机会，改动简单可回滚。
- 融合层面：在无法完全依赖显著性/检测结果时，融合的回退逻辑有时会选择大框作为保守策略，引入面积惩罚使得选择更偏向于“更小且在多维评分上表现良好”的候选。

## 如何测试
- 单图快速测试（预测并保存可视化与裁剪）：
```bash
python eval/predict_test_b.py --image-dir PATH/TO/ONE_IMAGE_DIR --config config.yaml --output-dir out_test --output-format both
```
- GUI 测试：重启服务并上传图片
```bash
python gui/app.py --config config.yaml
# 打开 http://127.0.0.1:5000 上传并观察结果
```
- 若使用长期运行的服务（Flask）请务必重启进程或调用 `/api/config` 接口清除全局 `AestheticCropper` 实例，确保新配置生效：
```bash
curl -X POST http://127.0.0.1:5000/api/config -H "Content-Type: application/json" -d '{}'
```

## 回退方案
- 若希望恢复为原始行为，可在 `config.yaml` 中将以下字段还原：
  - `candidate_generation.max_area_ratio: 0.95`
  - `candidate_generation.area_ratios: [0.35, 0.45, 0.60, 0.75, 0.90]`
  - 删除 `fusion` 中新增的 `area_penalty_*` 参数或设为 0。

