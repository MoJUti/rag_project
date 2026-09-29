# 正式知识库本地存储

本目录保存正式知识库运行数据，不保存评测实验报告。

```text
storage/
├─ sources/<document_id>/原始文件
├─ parsed/<document_id>/parsed_document.json
├─ assets/<document_id>/mineru/ 与 review_targets/
├─ chunks/<document_id>/chunks.jsonl
├─ indexes/  # 本地向量库、关键词索引等可重建索引
└─ evaluations/<dataset_id>/
   ├─ parsed/<document_id>/parsed_document.json
   ├─ assets/<document_id>/
   ├─ chunks/<strategy>/<document_id>/chunks.jsonl
   └─ reports/
```

顶层 `sources/parsed/assets/chunks/indexes` 属于正式知识库；`evaluations/` 只用于固定评测集，二者不能混用。除本说明和 `.gitkeep` 外，目录内容均被 Git 忽略。生产部署时可以将同一接口替换为对象存储和数据库，统一文档协议不变。
