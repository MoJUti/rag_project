"""Build a document-grounded Chinese QA set for retrieval and answer evaluation."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "data" / "eval" / "zh_public_minibench_v0.1"
ANN = BUNDLE / "annotations"


# Two questions per Workspace-Bench file.  Every answer below was checked against
# the extracted block referenced in evidence_block_ids.
WORKSPACE_QUESTIONS = [
    ("workspace-192-01", "宏海科技2024年营业收入是多少？", "462,903,179.20元", ["workspace-192-01#b00007"], "numeric_lookup", "medium"),
    ("workspace-192-01", "武汉宏海科技股份有限公司2024年年度报告中的法定代表人是谁？", "周宏", ["workspace-192-01#b00005"], "fact_lookup", "easy"),
    ("workspace-192-02", "太湖远大的股票于哪一天正式在北京证券交易所上市交易？", "2024年8月22日", ["workspace-192-02#b00002"], "date_lookup", "easy"),
    ("workspace-192-02", "浙江太湖远大新材料股份有限公司的法定代表人是谁？", "俞丽琴", ["workspace-192-02#b00007"], "fact_lookup", "easy"),
    ("workspace-328-01", "2024年下半年实际招聘人数和招聘完成率分别是多少？", "18人，75%", ["workspace-328-01#b00005"], "multi_fact_lookup", "medium"),
    ("workspace-328-01", "招聘工作总结中给出的新员工到岗率是多少？", "71.4%", ["workspace-328-01#b00007"], "numeric_lookup", "easy"),
    ("workspace-328-02", "人力资源管理模式计划从传统人事管理转向哪种主导模式？", "以业绩管理为主导的人力资源管理模式", ["workspace-328-02#b00003"], "fact_lookup", "medium"),
    ("workspace-328-02", "行政事务管理中被指出不足的一项内部沟通问题是什么？", "内部沟通的有效途径不足", ["workspace-328-02#b00006"], "fact_lookup", "medium"),
    ("workspace-328-03", "年度总入职人数和离职人数分别是多少？", "入职178人，离职75人", ["workspace-328-03#b00006"], "multi_fact_lookup", "easy"),
    ("workspace-328-03", "2024年本科及以上人员所占比例是多少？", "88.20%", ["workspace-328-03#b00008"], "numeric_lookup", "easy"),
    ("workspace-328-04", "前台文员岗位的应聘人数和录用人数分别是多少？", "应聘172人，录用1人", ["workspace-328-04#b00105"], "multi_fact_lookup", "medium"),
    ("workspace-328-04", "报告计算出的人均招聘成本是多少？", "181元", ["workspace-328-04#b00169"], "numeric_lookup", "easy"),
    ("workspace-329-01", "薪酬分析报告覆盖多少名员工和多少个核心部门？", "131名员工，5个核心部门", ["workspace-329-01#b00004"], "multi_fact_lookup", "easy"),
    ("workspace-329-01", "员工满意度抽样调查的样本量和覆盖率分别是多少？", "80人，61%", ["workspace-329-01#b00008"], "multi_fact_lookup", "medium"),
    ("workspace-33-01", "2023年全国社区卫生服务中心和服务站总数分别是多少？", "社区卫生服务中心10,070个，社区卫生服务站27,107个", ["workspace-33-01#b00005"], "multi_fact_lookup", "medium"),
    ("workspace-33-01", "2023年东部地区社区卫生服务中心和服务站分别有多少个？", "社区卫生服务中心4,372个，社区卫生服务站15,499个", ["workspace-33-01#b00006"], "multi_fact_lookup", "medium"),
    ("workspace-33-02", "2023年全国医疗卫生机构总数是多少？", "1,070,785个", ["workspace-33-02#b00005"], "numeric_lookup", "easy"),
    ("workspace-33-02", "2023年中部地区医院数量是多少？", "11,548个", ["workspace-33-02#b00007"], "numeric_lookup", "medium"),
    ("workspace-33-03", "2023年医疗卫生机构中公立和非公立机构分别有多少个？", "公立544,964个，非公立525,821个", ["workspace-33-03#b00006"], "multi_fact_lookup", "medium"),
    ("workspace-33-03", "2023年医院总数是多少？", "38,355个", ["workspace-33-03#b00007"], "numeric_lookup", "easy"),
    ("workspace-340-01", "员工生日会PPT目录把活动分成几个环节？", "6个环节", ["workspace-340-01#b00002"], "count_lookup", "easy"),
    ("workspace-340-01", "生日会PPT中提到的4位寿星来自哪个部门？", "财务部", ["workspace-340-01#b00003"], "fact_lookup", "easy"),
    ("workspace-340-02", "3月员工生日会通知中的活动时间是什么？", "2024年3月x日16:00-17:00", ["workspace-340-02#b00001"], "date_lookup", "easy"),
    ("workspace-340-02", "3月员工生日会通知中的活动地点在哪里？", "公司活动室", ["workspace-340-02#b00001"], "fact_lookup", "easy"),
    ("workspace-340-03", "员工生日祝福卡上的核心生日祝福语是什么？", "祝你生日快乐，幸福美满！", ["workspace-340-03#b00006"], "quote_lookup", "easy"),
    ("workspace-340-03", "生日祝福卡感谢员工的哪项付出？", "为公司所付出的努力", ["workspace-340-03#b00007"], "fact_lookup", "easy"),
    ("workspace-346-01", "固定资产报废报损申请审批表一式几联，分别交给哪些部门？", "一式三联：财务管理部、会计部和仓库各一联", ["workspace-346-01#b00003"], "multi_fact_lookup", "medium"),
    ("workspace-346-01", "固定资产报废报损申请表中的申请基本信息包括哪三项？", "申请部门、申请人、申请日期", ["workspace-346-01#b00004"], "list_lookup", "easy"),
    ("workspace-354-01", "公司行政体系建设蓝图提出的愿景是什么？", "打造“高效、规范、安全”的行政体系，成为公司战略落地的核心支撑", ["workspace-354-01#b00004"], "quote_lookup", "medium"),
    ("workspace-354-01", "行政管理六大模块分别是什么？", "流程制度管理、会务外联管理、文印资质管理、档案资料管理、采购资产管理、后勤安保管理", ["workspace-354-01#b00004"], "list_lookup", "hard"),
    ("workspace-37-01", "人事流动分析图表首页列出的四项核心人数指标是什么？", "本月离职人数、本月入职人数、累计离职人数、累计入职人数", ["workspace-37-01#b00004"], "list_lookup", "medium"),
    ("workspace-37-01", "工号20013416的员工属于哪个部门？", "销售部", ["workspace-37-01#b00021"], "fact_lookup", "medium"),
    ("workspace-37-02", "销售部KPI中D1指标的名称和权重是什么？", "销售目标完成率，权重40%", ["workspace-37-02#b00004"], "multi_fact_lookup", "medium"),
    ("workspace-37-02", "销售目标完成率的计算公式是什么？", "当月结单金额/当月预计结单金额×100%", ["workspace-37-02#b00004"], "formula_lookup", "medium"),
    ("workspace-37-03", "人员结构数据中，中专学历人数是多少？", "50人", ["workspace-37-03#b00006"], "numeric_lookup", "easy"),
    ("workspace-37-03", "人员结构数据中已婚和未婚人数合计是多少？", "250人", ["workspace-37-03#b00004", "workspace-37-03#b00005"], "arithmetic", "medium"),
    ("workspace-37-04", "人力资源决策分析大屏中的出勤人数是多少？", "800人", ["workspace-37-04#b00004"], "numeric_lookup", "easy"),
    ("workspace-37-04", "人力资源决策分析大屏中的休假人数是多少？", "10人", ["workspace-37-04#b00008"], "numeric_lookup", "easy"),
]


MULTI_DOCUMENT_QUESTIONS = [
    {
        "query": "宏海科技和太湖远大中，哪家公司2024年营业收入同比增速更高？两家的增速分别是多少？",
        "gold_answer": "宏海科技更高；宏海科技为32.09%，太湖远大为4.64%",
        "document_ids": ["workspace-192-01", "workspace-192-02"],
        "evidence_block_ids": ["workspace-192-01#b00007", "workspace-192-02#b00009"],
        "question_type": "cross_document_comparison",
        "difficulty": "hard",
        "split": "holdout-public",
    },
    {
        "query": "两份2023年卫生统计表中，全国社区卫生服务中心和服务站的总数是否一致？",
        "gold_answer": "一致，均为社区卫生服务中心10,070个、社区卫生服务站27,107个",
        "document_ids": ["workspace-33-01", "workspace-33-02"],
        "evidence_block_ids": ["workspace-33-01#b00005", "workspace-33-02#b00005"],
        "question_type": "cross_document_verification",
        "difficulty": "hard",
        "split": "smoke",
    },
]


OMNI_QUESTIONS = [
    ("omnidoc-01", "这页教材属于哪一章？", "第二章 行列式", [0], "layout_text_lookup", "easy"),
    ("omnidoc-02", "页面说明最小的自然数是多少？", "0", [5], "ocr_fact_lookup", "easy"),
    ("omnidoc-03", "这页课件介绍的系统类型是什么？", "线性时不变系统", [1], "layout_text_lookup", "easy"),
    ("omnidoc-04", "当函数在区间(a,b)单调时，页面给出的ω上界是什么？", "ω≤π/|b-a|", [6], "formula_lookup", "medium"),
    ("omnidoc-05", "古籍页面显示的本卦和变卦分别是什么？", "本卦为山风蛊，变卦为火泽睽", [1, 2], "multi_region_lookup", "hard"),
    ("omnidoc-06", "第13题页面中标记选择了哪个选项？", "D", [6], "visual_mark_lookup", "easy"),
    ("omnidoc-07", "文中称NASA内部预计裁撤多少名科学家？", "23名", [5], "ocr_fact_lookup", "easy"),
    ("omnidoc-08", "招商银行会计报表注释中的金额以什么单位列示？", "人民币千元", [1], "ocr_fact_lookup", "easy"),
    ("omnidoc-09", "表80的名称是什么？", "影像地图质量错漏分类表", [1], "table_caption_lookup", "easy"),
    ("omnidoc-10", "报纸头版报道要求在哪一年深入开展文化科技卫生“三下乡”活动？", "2011年", [2], "ocr_fact_lookup", "easy"),
    ("omnidoc-11", "页面上展示的选择题答案是什么选项，对应数值是多少？", "C，对应数值4", [2, 4], "multi_region_lookup", "medium"),
    ("omnidoc-12", "2018年英国航空旅客信息泄露事件的罚单金额是多少？", "2000万英镑", [5], "ocr_fact_lookup", "easy"),
]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def tableeval_answers(answer_record: dict) -> list[str]:
    result = []
    for question in answer_record.get("问题列表", []):
        result.extend(str(value) for value in question.get("最终答案", []))
    return result


def table_evidence(context: str, answers: list[str]) -> tuple[dict, str]:
    lines = [line.strip() for line in context.splitlines() if line.strip()]
    hits = []
    for line_no, line in enumerate(lines, 1):
        normalized = line.replace(",", "").replace("，", "")
        if any(
            answer.replace(",", "").replace("，", "") in normalized
            for answer in answers
            if len(answer) >= 2 and "无法回答" not in answer
        ):
            hits.append((line_no, line))
    if hits:
        line_numbers = [line_no for line_no, _ in hits[:5]]
        return {"type": "line", "lines": line_numbers}, "\n".join(line for _, line in hits[:5])
    return {"type": "document"}, context[:1200].strip()


def main() -> None:
    manifest = read_jsonl(BUNDLE / "manifest.jsonl")
    documents = {row["item_id"]: row for row in manifest}
    blocks = {row["block_id"]: row for row in read_jsonl(ANN / "extracted_blocks.jsonl")}
    omni_pages = {row["bundle_item_id"]: row for row in read_jsonl(ANN / "omnidocbench_zh_pages.jsonl")}
    questions = []

    # Preserve TableEval's official gold answers while normalizing the schema.
    for table in read_jsonl(ANN / "tableeval_zh_qa.jsonl"):
        document_id = table["bundle_item_id"]
        context = table["context"]["context_markdown"]
        for turn, query in enumerate(table["question_list"], 1):
            answers = tableeval_answers(table["golden_answer_list"][turn - 1])
            locator, quote = table_evidence(context, answers)
            answerable = not any("无法回答" in answer or "未提供" in answer for answer in answers)
            questions.append(
                {
                    "question_id": f"te-{int(table['id']):04d}-{turn:02d}",
                    "query": query,
                    "gold_answer": "；".join(answers),
                    "accepted_answers": answers,
                    "answerable": answerable,
                    "question_type": table["task_name"],
                    "difficulty": "medium",
                    "split": table["split"],
                    "relevant_document_ids": [document_id],
                    "evidence": [
                        {
                            "document_id": document_id,
                            "locator": locator,
                            "quote": quote,
                            "annotation_level": "dataset_gold_document",
                        }
                    ],
                    "source_dataset": "wenge-research/TableEval",
                    "source_record_id": table["id"],
                    "conversation_turn": turn,
                }
            )

    # Add manually verified, location-aware questions for every Office/PDF file.
    workspace_counter: Counter[str] = Counter()
    for document_id, query, answer, block_ids, question_type, difficulty in WORKSPACE_QUESTIONS:
        workspace_counter[document_id] += 1
        evidence = []
        for block_id in block_ids:
            block = blocks[block_id]
            evidence.append(
                {
                    "document_id": document_id,
                    "block_id": block_id,
                    "locator": block["locator"],
                    "quote": block["text"],
                    "annotation_level": "manually_verified_block",
                }
            )
        questions.append(
            {
                "question_id": f"wb-{document_id.removeprefix('workspace-')}-{workspace_counter[document_id]:02d}",
                "query": query,
                "gold_answer": answer,
                "accepted_answers": [answer],
                "answerable": True,
                "question_type": question_type,
                "difficulty": difficulty,
                "split": documents[document_id]["split"],
                "relevant_document_ids": [document_id],
                "evidence": evidence,
                "source_dataset": "Workspace-Bench-Lite/task_lite_clean_cn",
            }
        )

    for index, spec in enumerate(MULTI_DOCUMENT_QUESTIONS, 1):
        evidence = []
        for document_id, block_id in zip(spec["document_ids"], spec["evidence_block_ids"]):
            block = blocks[block_id]
            evidence.append(
                {
                    "document_id": document_id,
                    "block_id": block_id,
                    "locator": block["locator"],
                    "quote": block["text"],
                    "annotation_level": "manually_verified_block",
                }
            )
        questions.append(
            {
                "question_id": f"wb-cross-{index:02d}",
                "query": spec["query"],
                "gold_answer": spec["gold_answer"],
                "accepted_answers": [spec["gold_answer"]],
                "answerable": True,
                "question_type": spec["question_type"],
                "difficulty": spec["difficulty"],
                "split": spec["split"],
                "relevant_document_ids": spec["document_ids"],
                "evidence": evidence,
                "source_dataset": "Workspace-Bench-Lite/task_lite_clean_cn",
            }
        )

    # Use OmniDocBench's original reading-order and polygon annotations.
    for document_id, query, answer, orders, question_type, difficulty in OMNI_QUESTIONS:
        page = omni_pages[document_id]
        regions = [region for region in page["layout_dets"] if region.get("order") in orders]
        evidence = []
        for region in regions:
            evidence.append(
                {
                    "document_id": document_id,
                    "locator": {
                        "type": "page_region",
                        "page": page["page_info"].get("page_no", 0) + 1,
                        "order": region.get("order"),
                        "polygon": region["poly"],
                    },
                    "quote": region.get("text") or region.get("latex", ""),
                    "annotation_level": "upstream_ground_truth_region",
                }
            )
        if len(evidence) != len(orders):
            raise ValueError(f"Missing OmniDocBench evidence regions for {document_id}: {orders}")
        questions.append(
            {
                "question_id": f"omni-{int(document_id.rsplit('-', 1)[1]):02d}-01",
                "query": query,
                "gold_answer": answer,
                "accepted_answers": [answer],
                "answerable": True,
                "question_type": question_type,
                "difficulty": difficulty,
                "split": documents[document_id]["split"],
                "relevant_document_ids": [document_id],
                "evidence": evidence,
                "source_dataset": "opendatalab/OmniDocBench",
            }
        )

    # Fail loudly if the set cannot support document-level retrieval metrics.
    question_ids = [row["question_id"] for row in questions]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("Duplicate question_id detected")
    covered_documents = set()
    for row in questions:
        if not row["query"] or not row["gold_answer"] or not row["relevant_document_ids"]:
            raise ValueError(f"Incomplete question: {row.get('question_id')}")
        for document_id in row["relevant_document_ids"]:
            if document_id not in documents:
                raise ValueError(f"Unknown document_id: {document_id}")
            covered_documents.add(document_id)
        for evidence in row["evidence"]:
            if evidence["document_id"] not in row["relevant_document_ids"]:
                raise ValueError(f"Evidence/source mismatch: {row['question_id']}")
    missing = set(documents) - covered_documents
    if missing:
        raise ValueError(f"Documents without questions: {sorted(missing)}")

    questions.sort(key=lambda row: row["question_id"])
    write_jsonl(ANN / "questions_v0_draft.jsonl", questions)
    summary = {
        "schema_version": "1.0",
        "question_count": len(questions),
        "document_count": len(documents),
        "covered_document_count": len(covered_documents),
        "answerable_count": sum(row["answerable"] for row in questions),
        "unanswerable_count": sum(not row["answerable"] for row in questions),
        "by_source": dict(sorted(Counter(row["source_dataset"] for row in questions).items())),
        "by_split": dict(sorted(Counter(row["split"] for row in questions).items())),
        "by_difficulty": dict(sorted(Counter(row["difficulty"] for row in questions).items())),
        "retrieval_ground_truth": "relevant_document_ids",
        "fine_grained_ground_truth": "evidence[].block_id or evidence[].locator",
    }
    (ANN / "questions_v0_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    review_lines = [
        "# 问题集人工复核表",
        "",
        f"共 {len(questions)} 道问题，覆盖 {len(covered_documents)} 个文档。完整机器可读版本见 `questions.jsonl`。",
        "",
        "| ID | 问题 | 标准答案 | 相关文档 | 证据位置 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in questions:
        def escape(value: object) -> str:
            return str(value).replace("|", "\\|").replace("\n", " ")

        locations = []
        for evidence in row["evidence"]:
            locator = evidence["locator"]
            locations.append(
                evidence["document_id"] + ":" + json.dumps(locator, ensure_ascii=False, separators=(",", ":"))
            )
        review_lines.append(
            "| "
            + " | ".join(
                escape(value)
                for value in (
                    row["question_id"],
                    row["query"],
                    row["gold_answer"],
                    ", ".join(row["relevant_document_ids"]),
                    "; ".join(locations),
                )
            )
            + " |"
        )
    (ANN / "questions_v0_review.md").write_text(
        "\n".join(review_lines) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
