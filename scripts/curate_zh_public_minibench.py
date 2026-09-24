"""Build the small, Chinese-only public evaluation bundle used by this repo.

The script intentionally keeps source downloads separate from the curated files.
It is deterministic and can be rerun after the same source revisions are fetched.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "data" / "eval" / "zh_public_minibench_v0.1"
DOCS = BUNDLE / "documents"
ANN = BUNDLE / "annotations"

WORKSPACE_TASKS = {33, 37, 192, 328, 329, 340, 346, 354}
WORKSPACE_SPLITS = {
    33: "smoke",
    340: "smoke",
    328: "dev",
    329: "dev",
    346: "dev",
    192: "holdout-public",
    37: "holdout-public",
    354: "holdout-public",
}

OMNI_IMAGES = [
    "page-affbb0cc-d616-481d-b493-80ed1ccb5a10.png",
    "page-14cd673f-d86d-45a7-a13e-2b4e1d91c08f.png",
    "page-ba4443e6-f432-435f-89e7-ddf0a571d2cd.png",
    "page-1c08a4a2-8163-44d3-93f0-9449457dbba3.png",
    "page-8e2f7ce3-83c4-4977-b6dc-278394c4bd64.png",
    "page-b145da5c-182f-4d4e-813a-28b8d6971b0f.png",
    "page-6e75aef4-33d0-478a-a3a3-489c782bfa39.png",
    "page-9edf7687-029e-4e72-a9a6-c23faa1e1861.png",
    "page-91b20bf2-3ad5-41f4-b739-4cc4b18ab1eb.png",
    "newspaper_019d4d5296ba8f1c21277d72fb0cf0db_1.jpg",
    "page-374e9c4b-6fdb-4176-8ac7-519b86d97e40.png",
    "yanbaor2_0ef01b835238626f3589faa671abdfb832f2b98987eadad53cb2e35699dcc039.pdf_13.jpg",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def copy_with_manifest(
    source: Path,
    destination: Path,
    *,
    item_id: str,
    source_dataset: str,
    split: str,
    kind: str,
    extra: dict | None = None,
) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    record = {
        "item_id": item_id,
        "source_dataset": source_dataset,
        "split": split,
        "kind": kind,
        "format": destination.suffix.lower().lstrip("."),
        "path": destination.relative_to(BUNDLE).as_posix(),
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
    }
    if extra:
        record.update(extra)
    return record


def curate_workspace(manifest: list[dict]) -> list[dict]:
    metadata_path = (
        BUNDLE
        / "_source_metadata"
        / "workspace_bench"
        / "task_lite_clean_cn_metadata_table.csv"
    )
    with metadata_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if int(row["absolute_id"]) in WORKSPACE_TASKS
        ]

    annotations = []
    for row in sorted(rows, key=lambda value: int(value["absolute_id"])):
        task_id = int(row["absolute_id"])
        task_dir = BUNDLE / "_source_workspace" / "task_lite_clean_cn" / str(task_id) / "data"
        files = sorted(path for path in task_dir.iterdir() if path.is_file())
        included = []
        for source in files:
            original_name = source.name.split("_", 1)[1] if "_" in source.name else source.name
            destination = DOCS / source.suffix.lower().lstrip(".") / f"wb{task_id}_{original_name}"
            item_id = f"workspace-{task_id}-{len(included) + 1:02d}"
            manifest.append(
                copy_with_manifest(
                    source,
                    destination,
                    item_id=item_id,
                    source_dataset="Workspace-Bench-Lite/task_lite_clean_cn",
                    split=WORKSPACE_SPLITS[task_id],
                    kind="original_file",
                    extra={"workspace_task_id": task_id, "original_name": original_name},
                )
            )
            included.append(item_id)

        annotations.append(
            {
                "workspace_task_id": task_id,
                "split": WORKSPACE_SPLITS[task_id],
                "persona": row["persona"],
                "task": row["task"],
                "task_difficulty": row["task_diff"],
                "output_files": json.loads(row["output_files"]),
                "rubrics": json.loads(row["rubrics"]),
                "rubric_types": json.loads(row["rubric_types"]),
                "file_dependency_graph": json.loads(row["file_dep_graph"]),
                "tested_capabilities": json.loads(row["tested_capabilities"]),
                "included_item_ids": included,
            }
        )
    write_jsonl(ANN / "workspace_tasks.jsonl", annotations)
    return annotations


def curate_tableeval(manifest: list[dict]) -> list[dict]:
    source_path = BUNDLE / "_source_metadata" / "tableeval" / "TableEval-test.jsonl"
    with source_path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    rows = [row for row in rows if row["context"]["table_language"] == "简体中文"]

    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_task[row["task_name"]].append(row)

    selected = []
    for task_name in sorted(by_task):
        group = sorted(by_task[task_name], key=lambda row: int(row["id"]))[:4]
        for index, row in enumerate(group):
            split = ("smoke", "dev", "dev", "holdout-public")[index]
            item_id = f"tableeval-{int(row['id']):04d}"
            destination = DOCS / "md" / f"{item_id}.md"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(row["context"]["context_markdown"].strip() + "\n", encoding="utf-8")
            manifest.append(
                {
                    "item_id": item_id,
                    "source_dataset": "wenge-research/TableEval",
                    "split": split,
                    "kind": "derived_structured_table_markdown",
                    "format": "md",
                    "path": destination.relative_to(BUNDLE).as_posix(),
                    "bytes": destination.stat().st_size,
                    "sha256": sha256(destination),
                    "tableeval_id": row["id"],
                    "table_id": row["table_id"],
                }
            )
            selected.append({**row, "bundle_item_id": item_id, "split": split})

    write_jsonl(ANN / "tableeval_zh_qa.jsonl", selected)
    return selected


def curate_omnidoc(manifest: list[dict]) -> list[dict]:
    metadata_path = BUNDLE / "_source_metadata" / "omnidocbench" / "OmniDocBench.json"
    all_pages = json.loads(metadata_path.read_text(encoding="utf-8"))
    by_name = {page["page_info"]["image_path"]: page for page in all_pages}
    selected = []
    for index, name in enumerate(OMNI_IMAGES):
        page = by_name[name]
        language = page["page_info"]["page_attribute"]["language"]
        if language != "simplified_chinese":
            raise ValueError(f"Unexpected non-Chinese OmniDocBench page: {name} ({language})")
        source = BUNDLE / "_source_omnidoc" / "images" / name
        destination = DOCS / "image" / f"omni{index + 1:02d}_{name}"
        split = ("smoke", "dev", "dev", "holdout-public")[index % 4]
        item_id = f"omnidoc-{index + 1:02d}"
        manifest.append(
            copy_with_manifest(
                source,
                destination,
                item_id=item_id,
                source_dataset="opendatalab/OmniDocBench",
                split=split,
                kind="original_page_image",
                extra={"original_name": name, **page["page_info"]["page_attribute"]},
            )
        )
        selected.append({**page, "bundle_item_id": item_id, "split": split})
    write_jsonl(ANN / "omnidocbench_zh_pages.jsonl", selected)
    return selected


def write_support_files(manifest: list[dict], stats: dict) -> None:
    write_jsonl(BUNDLE / "manifest.jsonl", sorted(manifest, key=lambda row: row["item_id"]))
    sources = {
        "bundle": "Chinese Public MiniBench v0.1",
        "generated_at": "2026-09-24",
        "sources": [
            {
                "dataset": "Workspace-Bench/Workspace-Bench-Lite",
                "revision": "60b08b1cc2e8054afbc3ca2160d37876b4f0765c",
                "url": "https://huggingface.co/datasets/Workspace-Bench/Workspace-Bench-Lite",
                "upstream_project": "https://github.com/longyuewangdcu/Workspace-Bench",
                "declared_license": "MIT",
            },
            {
                "dataset": "wenge-research/TableEval",
                "revision": "12a8eb64156e9fe28c74c07ef91ac0f652f24a8d",
                "url": "https://huggingface.co/datasets/wenge-research/TableEval",
                "upstream_project": "https://github.com/wenge-research/TableEval",
                "declared_license": "Apache-2.0",
            },
            {
                "dataset": "opendatalab/OmniDocBench",
                "revision": "aa1ee96d106dbe53d0ae59474d75c6e6d9b53fec",
                "url": "https://huggingface.co/datasets/opendatalab/OmniDocBench",
                "upstream_project": "https://github.com/opendatalab/OmniDocBench",
                "declared_license": "Apache-2.0",
            },
        ],
    }
    (BUNDLE / "sources.lock.json").write_text(
        json.dumps(sources, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    format_counts: dict[str, int] = defaultdict(int)
    split_counts: dict[str, int] = defaultdict(int)
    for item in manifest:
        format_counts[item["format"]] += 1
        split_counts[item["split"]] += 1
    readme = f"""# Chinese Public MiniBench v0.1

这是为本项目整理的第一版中文、多格式、公开评测小集合。它用于先跑通解析、切块、召回和答案评测链路，再对 Chroma 与 Qdrant 做同条件比较。

## 当前内容

- 共 **{len(manifest)} 个文档条目**：{json.dumps(dict(sorted(format_counts.items())), ensure_ascii=False)}
- 数据划分：{json.dumps(dict(sorted(split_counts.items())), ensure_ascii=False)}
- Workspace-Bench：{stats['workspace_files']} 份原始 Office/PDF 文件，保留任务、rubric 和文件依赖关系。
- TableEval：{stats['tableeval_docs']} 份简体中文表格 Markdown 文档及对应问答。它们是官方 JSONL 中表格序列化的派生文件，并非原始 XLSX。
- OmniDocBench：{stats['omnidoc_pages']} 张简体中文复杂版面页面，保留布局/OCR/表格/公式标注。

## 目录

```text
documents/       实际送入解析与索引管线的文件
annotations/     任务、问题、答案、rubric、布局标注
manifest.jsonl   每个条目的来源、划分、大小与 SHA-256
sources.lock.json 固定数据集 revision、项目地址与声明许可证
_source_*        下载暂存区，不应作为正式评测输入
```

## 使用边界

1. `smoke` 用于快速冒烟，`dev` 用于调参，`holdout-public` 只在阶段验收时运行。
2. 这是公开数据，模型可能见过；`holdout-public` 不能替代未来自建、保密的最终测试集。
3. 文档条目与标注必须按 `item_id` 关联，评测时不要把答案/rubric 写入索引。
4. 上游项目声明了开源许可证，但部分文档内容可能来自第三方公开材料；再次分发或商用前仍需逐项核验内容权利。
5. 当前覆盖 DOCX、PPTX、XLSX、PDF、图片和 Markdown；后续再补 HTML、TXT、扫描整本 PDF 与真实跨文档问答。

## 推荐第一轮测试

- 解析完整率：文件成功率、页/表/幻灯片数量、乱码率、图片与表格保留率。
- 切块质量：边界正确率、标题路径保留、表格不被错误拆散、跨页重复率。
- 召回：Recall@5/10、MRR、nDCG@10，并分别比较稠密、BM25、混合检索与重排。
- 回答：答案正确性、证据命中、引用精度/召回率、无答案拒答。
- 可复现性：固定解析器/嵌入/重排模型版本、索引参数、随机种子与本文件中的数据 revision。
"""
    (BUNDLE / "README.md").write_text(readme, encoding="utf-8", newline="\n")


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    ANN.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    workspace = curate_workspace(manifest)
    tableeval = curate_tableeval(manifest)
    omni = curate_omnidoc(manifest)
    stats = {
        "workspace_files": sum(1 for item in manifest if item["source_dataset"].startswith("Workspace")),
        "tableeval_docs": len(tableeval),
        "omnidoc_pages": len(omni),
        "workspace_tasks": len(workspace),
    }
    write_support_files(manifest, stats)
    print(json.dumps({"bundle": str(BUNDLE), "items": len(manifest), **stats}, ensure_ascii=False))


if __name__ == "__main__":
    main()
