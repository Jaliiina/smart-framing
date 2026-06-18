# AestheticCropper 算法改动说明（本周全量 changelog）

> **周期**: 2026-06-11 — 2026-06-18  
> **基线版本**: （2026-06-13，或 `ffc7a69`）  
> **当前版本**:（2026-06-18，含未提交本地改动）  

---

## 本周工作量汇总

| 模块 | 改动文件数 | 新增/修改行数 | 核心主题 |
|------|-----------|-------------|---------|
| `src/fusion.py` | 1 | +319 / -133 | 融合策略全面重构：Stage 1 软性惩罚 + 权重归一化 + 垂直 COM 偏置 |
| `src/subject_detector.py` | 1 | +130 / -25 | 主体检测重写：双显著性降噪 + IoU-NMS 去重 + 主体完整性评分重构 |
| `src/composition_scorer.py` | 1 | +160 / -60 | 构图评分重构：地平线检测 + 植物场景专属留白 + 场景自适应 |
| `src/saliency_detector.py` | 1 | +20 / -10 | 双显著性系统：U2Net + OpenCV fallback 并行运行 + 一致性检测 |
| `src/aesthetic_scorer.py` | 1 | +15 / -5 | 三层降级策略完善 + 提示词优化 |
| `src/pipeline.py` | 1 | +55 / -0 | Pipeline 重构：封装 `process()` 方法 + 统一输出结构 |
| `src/utils.py` | 1 | +8 / -0 | 工具函数补充 |
| `config.yaml` | 1 | +30 / -15 | 全量参数更新：融合权重、双显著性配置、Stage 1 惩罚参数 |
| `eval/testa_diagnose.py` | 1 | +40 / -50 | 评估脚本重构：使用新的 `process()` API + 增强错误处理 |
| `docs/failure_analysis.md` | 1 | 新增 | 失败案例分析报告（testA 20 张图详细诊断） |
| `docs/optimization_report.md` | 1 | 新增 | 算法优化总报告（含所有模块设计说明） |

**总计**: 9 个核心文件改动，~1000 行代码变更，3 份文档新增，5 份诊断脚本与输出结果。

---

## 一、融合策略全面重构（`src/fusion.py`，+319/-133 行）

### 改动背景

上一版融合策略存在两个核心问题：
1. **排序错乱**：oracle gap（预测 IoU vs 最优候选 IoU）平均 0.359，说明最优候选经常被低分框排挤
2. **硬过滤与权重不匹配**：Stage 0 硬过滤在候选生成阶段过早丢弃正确答案

### 核心改动

#### 1. Stage 1 软性边界惩罚（新增 `stage1_filter` 配置）

**问题**：候选框贴着图像边缘时，通常意味着主体被裁切，这类框质量差。

**方案**：不硬过滤，而是软性扣分（`fusion.py` lines 148-158）：
```python
boundary_penalty[i] = penalty_strength * exceed_ratio
# exceed_ratio = min(1.0, (boundary_sal - boundary_th) / (boundary_th * 2))
```

配置项：
- `boundary_threshold: 0.25` — 边缘显著性均值上限
- `boundary_penalty_strength: 0.15` — 最大扣分比例

#### 2. 边缘贴近惩罚（`edge_adjacency_penalty`）

**问题**：候选框的四条边距离图像边缘过近（<4%）时，框的可信度降低。

**方案**：计算四条边到图像边界的距离比例，越贴近惩罚越强（`fusion.py` lines 108-123）：
```python
strength = edge_penalty_strength * (1.0 - dist_pct / edge_penalty_threshold)
```

配置项：
- `edge_adjacency_penalty_threshold: 0.04` — 4% 距离阈值
- `edge_adjacency_penalty_strength: 0.20` — 最大扣分

#### 3. 显著性垂直 COM 偏置（`saliency_vertical_bias`）

**问题**：风景图像中 saliency map 集中在底部（草地/地面），导致候选框偏向覆盖地面而非天空。

**方案**：当全图 saliency COM 在图像下方 60% 时，对候选框内部的 saliency COM 位置进行偏置，鼓励框内显著性位于上方（`fusion.py` lines 125-146）：
```python
if overall_com_y_pct > 0.40:
    vertical_bias[i] = bias_strength * local_com_y_pct
```

#### 4. 权重归一化与动态调整

**改动前**：权重直接使用配置值，subject 缺失时 subject 权重"消失"但不分配给其他维度。

**改动后**（`fusion.py` lines 160-196）：
- 权重先按配置赋值
- `saliency_is_uniform` 时：`saliency_weight -= 0.10`，多余权重平均分配给 `aesthetic` 和 `composition`
- `has_subject=False` 时：`subject_weight = 0.0`，**不转给 saliency**（防止纹理陷阱）
- 最后归一化所有活跃维度使权重和为 1

#### 5. 双显著性一致性检测（`dual_saliency_agreement`）

**新增**：同时运行 U2Net 和 OpenCV fallback，计算两者的皮尔逊相关系数。相关性低于阈值时，降低 saliency 权重（置信度下降）。

```python
dual_saliency_agreement_threshold: 0.65
dual_saliency_weight_reduction: 0.10
```

#### 6. 融合权重调整

| 维度 | 旧权重 | 新权重 | 原因 |
|------|--------|--------|------|
| aesthetic | 0.30 | **0.26** | 无 CLIP 时 handcrafted 特征不可靠，适度下调 |
| saliency | 0.12 | **0.15** | 恢复基础显著性权重，但加了多层保护机制 |
| composition | 0.18 | **0.23** | 新增构图维度后权重提升 |
| subject | 0.08 | **0.01** | 降权（算法内部改为信息性评分，不直接控制融合） |
| technical | 0.20 | **0.26** | 技术质量更稳定可靠，提升权重 |
| area_prior | 0.12 | **0.09** | 适度降低，避免过度偏好特定面积 |

---

## 二、主体检测重写（`src/subject_detector.py`，+130/-25 行）

### 核心改动

#### 1. 双显著性过滤（`_filter_by_saliency`）

**问题**：YOLO 经常将草地/岩石误检为长颈鹿等类别。

**方案**：检测到的物体如果在 saliency map 上的平均显著性 < 0.03，直接过滤掉（lines 222-232）。

#### 2. IoU-NMS 去重（lines 159-195）

**问题**：旧版按尺寸分组去重，粗暴地移除了多人/多同类场景。

**方案**：改为类内 IoU 去重（阈值 0.6），保留置信度更高的框，允许多个独立主体共存。

#### 3. 主体完整性评分重构（lines 234-312）

**改动前**：使用 max 归一化，抹平不同候选框之间的 completeness 差距。

**改动后**：
- 移除 max 归一化，保留原始 0~1 分差
- `weighted_inclusion`：加权覆盖度（框对物体的覆盖比例）
- `weighted_tightness`：加权紧凑度 `sqrt(obj_area / crop_area)`
- `boundary_penalty`：被截断的物体受罚（平方衰减 + 上限 0.3）
- `score = clamp(raw_score - penalty + 0.25 * tightness)`
- **全景 bonus**：`area_ratio >= 0.32` 且 `avg_inclusion >= 0.92` 时 +0.10

#### 4. 检测参数调整

| 参数 | 旧值 | 新值 | 原因 |
|------|------|------|------|
| `confidence_threshold` | 0.50 | **0.30** | 提高召回率 |
| `min_important_inclusion` | 0.80 | **0.92** | 更严格的主体完整性要求 |
| `tightness_weight` | 0.25 | **0.25** | 保持紧凑度权重 |

---

## 三、构图评分重构（`src/composition_scorer.py`，+160/-60 行）

### 新增构图维度

#### 1. 地平线检测（`_horizon_level`）

**算法**：结合 Sobel 水平梯度 + Laplacian 检测地平线位置，理想位置为上 1/3 或下 1/3。

```python
combined = 0.6 * normalize(sobel_rows) + 0.4 * normalize(laplacian_rows)
position_score = exp(-dist² / (2 * 0.10²))
```

#### 2. 场景自适应留白

**问题**：风景图像的留白评分逻辑与普通图像不同——天空是大面积"留白"，不应被惩罚。

**方案**：新增 `_is_wide_landscape()` 检测（低饱和度 + 高亮度 = 天空），风景场景使用 `_whitespace_landscape()` 评分（只惩罚完全平坦无特征的空白区域）。

#### 3. 植物场景专属处理

新增 `_is_plant_texture()` 检测，植物纹理场景使用专属的留白和边缘简洁性评分。

### 权重更新

| 维度 | 权重 | 说明 |
|------|------|------|
| rule_of_thirds | 0.35 | 三分法（核心构图规则） |
| center_balance | 0.25 | 中心平衡 |
| whitespace | 0.15 | 留白（自适应场景） |
| edge_simplicity | 0.15 | 边缘简洁性 |
| symmetry | 0.10 | 对称性 |
| horizon_level | 0.00 | 地平线（信息性，不直接加权） |

---

## 四、双显著性系统（`src/saliency_detector.py`）

### 改动内容

#### 1. U2Net + OpenCV fallback 并行运行

**方案**：`detect()` 方法同时返回 U2Net saliency map 和 OpenCV fallback saliency map，以及各自的 uniform 标记。

```python
u2net_map, fallback_map, u2net_uniform, fallback_uniform = self._detect_dual(image)
```

#### 2. 一致性检测

当两者相关性低于 `dual_saliency_agreement_threshold: 0.65` 时，在融合阶段降低 saliency 权重。

#### 3. 强制 fallback 模式

新增 `force_fallback: false` 配置，支持在 U2Net 权重缺失时自动降级。

---

## 五、美学评分优化（`src/aesthetic_scorer.py`）

### 提示词优化

**旧提示词**（较泛泛）：
```
"a well-composed professional photograph"
"a visually pleasing photograph with a clear subject"
"a balanced scenic photograph"
```

**新提示词**（更具体，针对风景/全景场景）：
```
"a well-composed professional photograph"
"a visually pleasing photograph with harmonious layered scenery"
"a balanced wide scenic photograph with complete far background"

"a cluttered photograph with distracting foreground debris"
"a narrow partial crop focusing only on one isolated foreground object"
"a poorly framed photograph missing most of the distant landscape view"
```

---

## 六、Pipeline 重构（`src/pipeline.py`）

### 新增 `process()` 方法

封装完整的处理流程，返回结构化结果：
```python
result = cropper.process(str(image_path))
best = result.top_candidates[0]
saliency_map = result.saliency_map
objects = result.detected_objects
all_candidates = result.all_candidates
```

评估脚本 `eval/testa_diagnose.py` 已切换到新的 API，代码量减少 ~50 行。

---

## 七、候选框生成优化（`config.yaml` → `candidate_generation`）

| 参数 | 旧值 | 新值 | 原因 |
|------|------|------|------|
| `preserve_original_aspect` | — | **true** | 只生成与原图相同宽高比的候选框，避免宽图被压成竖框 |
| `min_area_ratio` | 0.15 | **0.15** | 保持 |
| `max_area_ratio` | 0.45 | **0.45** | 保持 |
| `saliency_supplement` | — | **true** | 在网格采样的基础上补充 saliency peak 周围的候选框 |
| `saliency_peak_threshold` | — | **0.7** | 显著性峰值阈值 |
| `saliency_smooth_sigma` | — | **5.0** | 显著性平滑参数 |

---

## 八、失败案例分析（`docs/failure_analysis.md`）

### 核心发现

**Oracle Gap = 0.359**：最优候选框的 IoU 平均为 0.832，但算法实际选中的平均仅 0.473。差距主要来自融合排序错误，而非候选框生成不足。

### 关键失败案例

| 图片 | 预测 IoU | Oracle Gap | 核心问题 |
|------|---------|-----------|---------|
| A02 | 0.007 | 0.957 | 中心偏移 0.47，误选 boat 候选项 |
| A06 | 0.000 | 0.837 | 完全错位，saliency 误导 |
| A10 | 0.105 | 0.607 | 主体评分满分但 IoU 极低，候选框排序错误 |
| A14 | 0.083 | 0.752 | 无 primary detection，saliency 主导 |
| A18 | 0.060 | 0.735 | 无 primary detection，完全错位 |

### 系统性结论

1. **YOLO 召回瓶颈**：8 个失败案例中 4 个无 primary detection（A04/A06/A14/A18）
2. **候选框多样性不足**：oracle gap 说明正确答案在候选池中但被 NMS/排序压制
3. **saliency 纹理陷阱**：草地/地面等高对比纹理误导裁剪方向
