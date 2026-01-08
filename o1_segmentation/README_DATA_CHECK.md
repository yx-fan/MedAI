# 数据检查工具说明

在训练模型之前，建议先运行数据检查工具，了解数据的"样子"，这样可以更好地配置模型参数。

## 推荐：使用综合检查工具

**`check_dataset.py`** - 一站式综合检查，包含所有检查项：

```bash
# 完整检查（推荐）
python o1_segmentation/check_dataset.py

# 指定数据目录
python o1_segmentation/check_dataset.py --data_dir ./data/raw

# 安静模式（减少输出）
python o1_segmentation/check_dataset.py --quiet
```

**检查内容：**
1. ✅ 文件匹配（images vs masks）
2. ✅ Shape一致性检查
3. ✅ Spacing分布统计
4. ✅ 强度范围（HU值）统计
5. ✅ 标签值和类别分布
6. ✅ ROI大小统计
7. ✅ 数据类型和内存使用
8. ✅ **模型配置建议**（patch_size, batch_size, normalization等）

**`check_dataset.py` 已经包含了所有单独的analyze文件的功能，建议直接使用它。**

## 单项检查工具（可选，已整合到check_dataset.py）

以下文件的功能已经全部整合到 `check_dataset.py` 中，**可以删除或保留作为轻量级快速检查的备用**：

- `check_labels.py` - 标签分布检查（已整合）
- `analyze_roi_size.py` - ROI大小分析（已整合）
- `analyze_spacing.py` - Spacing分析（已整合）
- `analyze_intensity_range.py` - 强度范围分析（已整合）

如果数据量很大，只想快速检查某个特定方面，可以使用这些单独的工具。但对于大多数情况，**直接使用 `check_dataset.py` 即可**。

### 可视化工具（保留）

- `plot_hu_histograms.py` - 绘制HU值直方图（可视化，保留）

## 使用建议

1. **训练前必做**：运行 `check_dataset.py` 获取完整数据概览
2. **根据检查结果调整**：
   - 如果spacing不一致 → 考虑使用Spacingd transform
   - 如果shape差异大 → 考虑resize或使用sliding window
   - 根据ROI大小确定patch_size
   - 根据强度范围设置normalization参数
3. **模型配置**：`check_dataset.py` 会给出推荐的patch_size和batch_size

## 输出示例

`check_dataset.py` 会输出：
- 数据统计（shape, spacing, intensity等）
- 潜在问题警告（如shape不一致、spacing不同等）
- **模型配置建议**（推荐的patch_size, batch_size, normalization等）

这些信息可以帮助你：
- 选择合适的patch_size
- 设置合适的batch_size（基于内存估算）
- 配置正确的normalization参数
- 决定是否需要数据预处理（如resampling）
