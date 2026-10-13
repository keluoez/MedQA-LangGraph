# 数据集评分口径校准说明（2026-10-13）

本文件记录病理（Pathological）模块历史质量分的口径修复与数据重算，是数据集版本变更的审计依据。

## 1. 背景与成因

病理模块旧的 verifier 评分公式混用了 LLM 语义维度与 deterministic 格式分，且 `pathology_mechanism` 维度未做上限钳制：

```
旧口径：total = pathology_mechanism（未钳制，LLM 最高可打 5）
              + disease_relevance（≤3）
              + evidence_support（≤2）
              + deterministic 格式分（≤2）
      → 理论上限 12 分（十分制下越界）
```

实证影响：4,988 条病理样本中有 **876 条超过 10 分**（11 分 × 780 条、12 分 × 96 条），平均分被系统性抬高。

## 2. 修复后的新口径

```python
# basic_pathological_environment/langgraph_pipeline.py（M4-B2 起）
total = min(4.0, pathology_mechanism) + disease_relevance + evidence_support + llm_format
# 上限 4 + 3 + 2 + 1 = 10 分
```

与其他三个模块（drug / experimental / physiological）一致，所有维度钳制在十分制内。

## 3. 历史数据重算

重算由 [recalculate_scores.py](../basic_pathological_environment/recalculate_scores.py) 完成，完全离线（无 LLM、无网络调用），依据历史记录 `quality_log.subscores` 中保留的 LLM 原始四维分数重新求和。

**执行记录（2026-10-13）**：

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| >10 分记录数 | 876 | **0** |
| 病理平均分 | 8.215 | **7.187** |
| 分差分布 | — | -2.0 × 144 条、-1.0 × 4,841 条、0.0 × 3 条 |

**全数据集影响**（四个域合计 22,571 条不变）：

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| 总平均分 | 8.57 | **8.34** |
| Gold（≥8.0） | 66.1%（14,918） | **62.0%（13,987）** |
| Silver（6–8） | 30.1% | **33.3%（7,509）** |
| Bronze（<6.0） | 3.7%（843） | **4.8%（1,075）** |
| 病理域平均分 | 8.22 | **7.19** |
| 病理域 Gold | 59.9% | **41.2%** |
| 病理域 Bronze | 9.0% | **13.6%** |

其余三个域（drug / experimental / physiological）不受影响。

## 4. 可审计性

- 每条被修改的记录在 `quality_log.score_recalc` 中保留原始分数与口径说明，可逐条核对：
  ```json
  "score_recalc": {
      "old_quality_score": 11.0,
      "caliber": "M4-B2: min(4,mech)+disease+evidence+llm_format"
  }
  ```
- 原始未重算数据完整备份于 `basic_pathological_environment/output/langgraph_batch_output.json.pre-recalc`（另有更早的 `.jsonl.bak`），随时可回滚。
- 全部对外引用数字（文档与演示口径）已同步更新，修改记录另行保存（不在本仓库）。

## 5. 时间线

| 时间 | 事件 |
|------|------|
| 2026-04 | 病理模块批量生成完成（旧口径，含 876 条 >10 分） |
| 2026-10-09（M4-B2） | verifier 代码修复 + 重算脚本编写，dry-run 验证（未写入数据） |
| 2026-10-13 | 执行 `--write` 重算并替换数据源、同步论文与全部文档（本次） |
