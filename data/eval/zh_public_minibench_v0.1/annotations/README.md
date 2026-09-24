# 评测标注说明

## 套件划分

不同类型的案例不能混在一起计算一个总分：

- `retrieval_questions.jsonl`：55 道正式全库检索题；用于 Chroma/Qdrant 对比。
- `questions.jsonl`：与正式检索集内容相同，作为兼容入口。
- `table_qa_cases.jsonl`：26 道已指定来源表格的问答；只测表格理解和回答。
- `parser_cases.jsonl`：12 个 OmniDocBench 页面解析案例；测版面、OCR、阅读顺序、表格和公式。
- `refusal_cases.jsonl`：4 道资料不足问题；测正确拒答和无依据陈述。
- `excluded_cases.jsonl`：被移出正式评分的候选题及原因。
- `questions_v0_draft.jsonl`：第一版候选题，仅供追溯，禁止用于正式对比。

`retrieval_questions.jsonl` 每行是一道正式检索题，核心字段如下：

| 字段 | 用途 |
| --- | --- |
| `question_id` | 稳定的问题主键 |
| `query` | 发送给检索系统的问题 |
| `gold_answer` | 标准答案 |
| `accepted_answers` | 可接受答案列表，供规则或语义评分使用 |
| `answerable` | 文档中是否存在答案；可用于拒答测试 |
| `split` | `smoke`、`dev` 或 `holdout-public` |
| `relevant_document_ids` | 文档级召回真值 |
| `evidence` | 细粒度证据，包括来源文档、页码/工作表/幻灯片/段落/图片区域、证据原文 |
| `question_type` | 查询类型 |
| `difficulty` | 难度标签 |

## 如何计算召回

文档级召回使用 `relevant_document_ids`：

```text
Recall@K = 前 K 个检索结果中命中的相关文档数 / 该问题的相关文档总数
```

MRR 使用第一个相关文档的排名；nDCG@K 使用 `relevance_judgments`。多文档题还要计算 complete-evidence Recall@K，只有相关文档全部进入前 K 才算完整命中。

切块以后，将每个 chunk 保留 `document_id`、原始 locator 和字符范围。只要 chunk 与 `evidence` 中的 block/页/表格行/幻灯片/图片区域重叠，就可以派生出 `relevant_chunk_ids`，用于计算 chunk Recall@K。

## 标注等级

- `manually_verified_block`：从 Office/PDF 原文件抽取并逐块核验的问题、答案和证据。
- `upstream_ground_truth_region`：直接使用 OmniDocBench 的阅读顺序和多边形区域标注。

## 其他文件

- `extracted_blocks.jsonl`：Office/PDF/Markdown 的位置感知文本块。
- `tableeval_zh_qa.jsonl`：保留的 TableEval 原始结构。
- `workspace_tasks.jsonl`：Workspace-Bench 任务和 rubric。
- `omnidocbench_zh_pages.jsonl`：OmniDocBench 页面级完整标注。
- `questions_summary.json`：正式检索集的数量和分布统计。
- `benchmark_manifest.json`：各套件边界、适用指标和已知限制。
- `audit_report.md`：逐项复核结论和被移除题目的原因。
