"""Create the verified v1 evaluation suites from the v0 candidate questions.

The important design choice is that parser, retrieval, table-QA, and refusal
cases are kept separate.  Only retrieval_questions.jsonl is eligible for
Chroma/Qdrant retrieval comparisons.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "data" / "eval" / "zh_public_minibench_v0.1"
ANN = BUNDLE / "annotations"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def evidence_from_blocks(blocks: dict[str, dict], pairs: list[tuple[str, str]]) -> list[dict]:
    result = []
    for document_id, block_id in pairs:
        block = blocks[block_id]
        result.append(
            {
                "document_id": document_id,
                "block_id": block_id,
                "locator": block["locator"],
                "quote": block["text"],
                "annotation_level": "manually_verified_block",
            }
        )
    return result


REMOVED_WORKSPACE = {
    "wb-329-01-02": "抽样方法元数据不是典型知识检索需求，替换为薪酬风险分析题",
    "wb-340-01-01": "只数PPT目录项目，信息价值和检索难度过低",
    "wb-340-03-01": "通用生日祝福可由模型常识直接回答，无法证明检索有效",
    "wb-340-03-02": "通用感谢语信息量过低，且可能在多份材料中重复",
}

REMOVED_OMNI = {
    "omni-01-01": "章节标题识别只适合解析冒烟，不是全库真实检索需求",
    "omni-02-01": "答案属于常识，不依赖知识库",
    "omni-03-01": "直接询问页面标题，关键词泄漏且难度过低",
    "omni-06-01": "识别勾选项属于视觉解析任务，不应计入文本检索排名",
    "omni-11-01": "识别试卷选项属于视觉解析任务，不应计入文本检索排名",
}

TABLE_STRUCTURE_IDS = {"te-2040-01", "te-2041-01", "te-2042-01", "te-2043-01"}

QUERY_OVERRIDES = {
    "wb-328-01-01": "《招聘工作总结汇报》显示，2024年下半年实际招聘人数和招聘完成率分别是多少？",
    "wb-328-01-02": "《招聘工作总结汇报》指出的新员工到岗率是多少？",
    "wb-328-02-01": "《人事行政工作总结》计划将传统人事管理转向哪种主导模式？",
    "wb-328-02-02": "《人事行政工作总结》指出，行政事务管理在内部沟通方面存在哪项不足？",
    "wb-328-03-01": "《人力资源部招聘工作总结》记录的年度总入职人数和离职人数分别是多少？",
    "wb-328-03-02": "《人力资源部招聘工作总结》中，2024年本科及以上人员占比是多少？",
    "wb-328-04-01": "《招聘工作年度分析报告》中，前台文员岗位的应聘人数和录用人数分别是多少？",
    "wb-328-04-02": "《招聘工作年度分析报告》计算出的人均招聘成本是多少？",
    "wb-37-01-01": "《人事流动分析图表》的首页列出了哪四项核心人数指标？",
    "wb-37-01-02": "《人事流动分析图表》的员工信息登记表中，工号20013416属于哪个部门？",
    "wb-37-02-01": "销售部KPI表中，D1指标的名称和权重是什么？",
    "wb-37-02-02": "销售部KPI表中，销售目标完成率采用什么计算公式？",
    "wb-37-03-01": "公司人员结构看板的数据表中，中专学历人数是多少？",
    "wb-37-03-02": "公司人员结构看板的数据表中，已婚和未婚人数合计是多少？",
    "wb-37-04-01": "人力资源决策分析大屏的今日考勤数据中，出勤人数是多少？",
    "wb-37-04-02": "人力资源决策分析大屏的今日考勤数据中，休假人数是多少？",
    "omni-04-01": "在正弦函数“轴心加单调”方法中，若函数在区间(a,b)单调，ω需要满足的上界是什么？",
    "omni-05-01": "古籍卦象记录中的本卦和变卦分别是什么？",
    "omni-07-01": "2025年3月关于美国科研预算调整的报道中，NASA内部预计裁撤多少名科学家？",
    "omni-08-01": "招商银行会计报表注释中，除特别说明外，金额以什么单位列示？",
    "omni-09-01": "GB/T 24356—2023标准中的表80名称是什么？",
    "omni-10-01": "《农民日报》2010年12月21日的报道要求在哪一年深入开展文化科技卫生“三下乡”活动？",
    "omni-12-01": "普华永道航空简讯提到，2018年英国航空旅客信息泄露事件收到的罚单是多少？",
}


HEALTH_RELEVANCE = {
    "wb-33-01-01": [
        ("workspace-33-01", "workspace-33-01#b00005"),
        ("workspace-33-02", "workspace-33-02#b00005"),
    ],
    "wb-33-01-02": [
        ("workspace-33-01", "workspace-33-01#b00006"),
        ("workspace-33-02", "workspace-33-02#b00006"),
    ],
    "wb-33-02-01": [
        ("workspace-33-02", "workspace-33-02#b00005"),
        ("workspace-33-03", "workspace-33-03#b00006"),
    ],
    "wb-33-03-02": [
        ("workspace-33-03", "workspace-33-03#b00007"),
        ("workspace-33-02", "workspace-33-02#b00005"),
    ],
}


STANDALONE_TABLE_QUERIES = {
    "te-2009-02": "《关于全省长流程钢铁企业环保绩效全面创A工作的通知》表格是否提供了四项重点任务的具体执行内容？",
    "te-2010-02": "中国制造业PMI表中，2023年11月的产成品库存指数是多少？",
    "te-2010-03": "中国制造业PMI表中，2023年12月的产成品库存指数比2023年11月变化了多少个百分点？",
    "te-2011-02": "主要会计数据和财务指标表中，上年同期调整后的营业收入是多少？",
    "te-2011-03": "主要会计数据和财务指标表中，本报告期营业收入与上年同期调整后营业收入相差多少？",
    "te-2013-02": "现金管理情况表中，金额在50,000万元以上的产品是否都已赎回？",
}


TABLE_RETRIEVAL_SELECTION = {
    "te-0003-01": "股东信息表中，陕西建工实业有限公司在前十大股东中的持股比例是多少？",
    "te-0007-01": "齐耀重工的废水排放口分布在什么位置？",
    "te-0638-01": "2024年1—2月社会消费品零售总额表列出的限额以上单位商品零售分类中，哪个类别同比增长率最高？",
    "te-0639-01": "本次降准后的上市银行存款准备金变动表中，释放流动性最高的是哪家银行？",
    "te-0642-01": "2024年1—5月规模以上工业生产主要数据中，哪个主要行业增加值同比增长率最高？",
    "te-1325-01": "发起人持股表中，彭建虎的持股比例为何远高于其他发起人？",
    "te-2009-01": "2023年《关于全省长流程钢铁企业环保绩效全面创A工作的通知》列出的四项重点任务是什么？",
    "te-2010-01": "中国制造业PMI其他相关指标表中，2023年12月的产成品库存指数是多少？",
    "te-2013-01": "现金管理前十二个月情况表中，金额在50,000万元以上的产品及其签约机构有哪些？",
}


TABLE_CASE_CORRECTIONS = {
    "te-1187-01": {
        "query": "按版本顺序比较特斯拉自动驾驶硬件平台，相邻版本算力提升量分别是多少，哪次提升最大？",
        "gold_answer": "提升量依次为19.744 TOPS、0 TOPS、124 TOPS、576 TOPS和6480 TOPS；从HW4.0到AI 5的提升最大。",
        "accepted_answers": ["19.744 TOPS；0 TOPS；124 TOPS；576 TOPS；6480 TOPS；HW4.0到AI 5最大"],
    },
    "te-2010-03": {
        "gold_answer": "-0.4个百分点",
        "accepted_answers": ["-0.4个百分点", "下降0.4个百分点"],
    },
    "te-2013-01": {
        "gold_answer": "（机构专属）中银理财-稳享(封闭式)2023041—中国银行哈尔滨香坊支行；利多多公司稳利24JG3305期（月月滚利2期特供款）—上海浦东发展银行哈尔滨分行营业部；利多多公司稳利24JG6066期(三层看涨)人民币对公结构性存款—上海浦东发展银行哈尔滨分行营业部",
        "accepted_answers": ["中银理财2023041—中国银行哈尔滨香坊支行；24JG3305—上海浦东发展银行哈尔滨分行营业部；24JG6066—上海浦东发展银行哈尔滨分行营业部"],
        "evidence_pairs": [
            ("tableeval-2013", "tableeval-2013#b00006"),
            ("tableeval-2013", "tableeval-2013#b00018"),
            ("tableeval-2013", "tableeval-2013#b00024"),
        ],
        "correction_note": "上游答案漏掉金额100,000万元的第15行；“以上”按包含50,000万元处理",
    },
    "te-2013-02": {
        "gold_answer": "否；前两个产品已赎回，利多多公司稳利24JG6066期尚未赎回。",
        "accepted_answers": ["否", "不是全部已赎回", "24JG6066期未赎回"],
        "evidence_pairs": [
            ("tableeval-2013", "tableeval-2013#b00006"),
            ("tableeval-2013", "tableeval-2013#b00018"),
            ("tableeval-2013", "tableeval-2013#b00024"),
        ],
        "correction_note": "上游答案与表格赎回状态矛盾；第21行明确标注“否”",
    },
}


def apply_table_correction(item: dict, blocks: dict[str, dict]) -> dict:
    correction = TABLE_CASE_CORRECTIONS.get(item["question_id"])
    if not correction:
        item["gold_verification"] = "upstream_gold_reviewed"
        return item
    item["upstream_gold_answer"] = item["gold_answer"]
    for key in ("query", "gold_answer", "accepted_answers", "correction_note"):
        if key in correction:
            item[key] = correction[key]
    if "evidence_pairs" in correction:
        item["evidence"] = evidence_from_blocks(blocks, correction["evidence_pairs"])
    item["gold_verification"] = "locally_corrected_against_source_table"
    return item


def build_retrieval_questions(v0: list[dict], blocks: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    retrieval = []
    excluded = []
    for row in v0:
        qid = row["question_id"]
        if qid.startswith("te-"):
            continue
        reason = REMOVED_WORKSPACE.get(qid) or REMOVED_OMNI.get(qid)
        if reason:
            excluded.append({"question_id": qid, "original_query": row["query"], "reason": reason})
            continue

        item = dict(row)
        item["query"] = QUERY_OVERRIDES.get(qid, row["query"])
        item["suite"] = "retrieval"
        item["modality"] = "image_ocr" if qid.startswith("omni-") else "document_text"
        item["eligible_metrics"] = ["recall_at_k", "mrr", "ndcg_at_k"]
        item["requires_complete_evidence_set"] = len(item["relevant_document_ids"]) > 1
        if qid in HEALTH_RELEVANCE:
            pairs = HEALTH_RELEVANCE[qid]
            item["relevant_document_ids"] = list(dict.fromkeys(document_id for document_id, _ in pairs))
            item["evidence"] = evidence_from_blocks(blocks, pairs)
            item["requires_complete_evidence_set"] = len(item["relevant_document_ids"]) > 1
        item["relevance_judgments"] = {
            document_id: 2 for document_id in item["relevant_document_ids"]
        }
        retrieval.append(item)

    # TableEval is mainly a source-given table-QA benchmark.  Only the small,
    # explicitly reviewed self-contained subset below is also eligible for
    # whole-corpus retrieval metrics, giving Markdown positive coverage.
    v0_by_id = {row["question_id"]: row for row in v0}
    for source_id, query in TABLE_RETRIEVAL_SELECTION.items():
        source = v0_by_id[source_id]
        item = apply_table_correction(dict(source), blocks)
        item["question_id"] = "ret-" + source_id
        item["original_query"] = source["query"]
        item["query"] = query
        item["suite"] = "retrieval"
        item["modality"] = "structured_markdown"
        item["eligible_metrics"] = ["recall_at_k", "mrr", "ndcg_at_k"]
        item["retrieval_score_eligible"] = True
        item["requires_complete_evidence_set"] = False
        item["relevance_judgments"] = {
            document_id: 2 for document_id in item["relevant_document_ids"]
        }
        retrieval.append(item)

    # Replace a methodology-only salary question with a real diagnostic query.
    retrieval.append(
        {
            "question_id": "wb-329-01-risk-01",
            "query": "《薪酬分析报告》用哪些数字说明专家级人才存在薪酬竞争力不足和流失风险？",
            "gold_answer": "专家级薪酬缺口为16%（32,000元对比市场38,000元），主动离职率为20%（行业均值12%）。",
            "accepted_answers": ["薪酬缺口16%；主动离职率20%；行业均值12%"],
            "answerable": True,
            "question_type": "evidence_synthesis",
            "difficulty": "hard",
            "split": "dev",
            "relevant_document_ids": ["workspace-329-01"],
            "relevance_judgments": {"workspace-329-01": 2},
            "evidence": evidence_from_blocks(blocks, [("workspace-329-01", "workspace-329-01#b00119")]),
            "source_dataset": "Workspace-Bench-Lite/task_lite_clean_cn",
            "suite": "retrieval",
            "modality": "document_text",
            "eligible_metrics": ["recall_at_k", "mrr", "ndcg_at_k"],
            "requires_complete_evidence_set": False,
        }
    )

    extra_cross = [
        {
            "question_id": "wb-cross-03",
            "query": "结合3月员工生日会通知和生日会PPT，活动安排的时间、地点以及六个流程环节分别是什么？",
            "gold_answer": "时间为2024年3月x日16:00-17:00，地点为公司活动室；六个环节为寿星登场、节目欣赏、发放礼物、寿星感言、游戏环节、合影留念。",
            "pairs": [("workspace-340-02", "workspace-340-02#b00001"), ("workspace-340-01", "workspace-340-01#b00002")],
            "split": "smoke",
        },
        {
            "question_id": "wb-cross-04",
            "query": "两份招聘总结分别给出了哪些总体招聘数据：2024年下半年实际招聘人数与完成率，以及另一份材料中的年度入职与离职人数？",
            "gold_answer": "2024年下半年实际招聘18人、完成率75%；另一份材料记录年度入职178人、离职75人。",
            "pairs": [("workspace-328-01", "workspace-328-01#b00005"), ("workspace-328-03", "workspace-328-03#b00006")],
            "split": "dev",
        },
    ]
    for spec in extra_cross:
        document_ids = [document_id for document_id, _ in spec["pairs"]]
        retrieval.append(
            {
                "question_id": spec["question_id"],
                "query": spec["query"],
                "gold_answer": spec["gold_answer"],
                "accepted_answers": [spec["gold_answer"]],
                "answerable": True,
                "question_type": "cross_document_synthesis",
                "difficulty": "hard",
                "split": spec["split"],
                "relevant_document_ids": document_ids,
                "relevance_judgments": {document_id: 2 for document_id in document_ids},
                "evidence": evidence_from_blocks(blocks, spec["pairs"]),
                "source_dataset": "Workspace-Bench-Lite/task_lite_clean_cn",
                "suite": "retrieval",
                "modality": "cross_document",
                "eligible_metrics": ["recall_at_k", "mrr", "ndcg_at_k", "complete_evidence_recall_at_k"],
                "requires_complete_evidence_set": True,
            }
        )
    return sorted(retrieval, key=lambda row: row["question_id"]), excluded


def build_table_qa(v0: list[dict], blocks: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    rows = [row for row in v0 if row["question_id"].startswith("te-")]
    excluded = []
    accepted = []
    history_by_record: dict[str, list[dict]] = {}
    for row in sorted(rows, key=lambda value: value["question_id"]):
        qid = row["question_id"]
        record_id = row["source_record_id"]
        history = history_by_record.setdefault(record_id, [])
        if qid in TABLE_STRUCTURE_IDS:
            excluded.append(
                {
                    "question_id": qid,
                    "original_query": row["query"],
                    "reason": "当前正式输入是Markdown；合并单元格的rowspan/colspan结构已丢失，题目与输入表示不匹配",
                }
            )
        else:
            item = apply_table_correction(dict(row), blocks)
            item["original_query"] = row["query"]
            item["query"] = STANDALONE_TABLE_QUERIES.get(qid, item["query"])
            item["conversation_history"] = list(history)
            item["suite"] = "table_qa"
            item["eligible_metrics"] = ["answer_correctness", "answer_f1", "faithfulness"]
            item["retrieval_score_eligible"] = False
            item["note"] = "源表已指定；该套件评测表格理解和回答，不参与全库向量检索排名"
            accepted.append(item)
        history.append({"query": row["query"], "answer": row["gold_answer"]})
    return accepted, excluded


def build_parser_cases(omni_pages: list[dict], documents: dict[str, dict]) -> list[dict]:
    cases = []
    for page in omni_pages:
        document_id = page["bundle_item_id"]
        categories = Counter(region["category_type"] for region in page["layout_dets"] if not region.get("ignore"))
        metrics = ["layout_detection_f1", "reading_order_edit_distance", "text_normalized_edit_distance"]
        if categories.get("table") or categories.get("table_caption"):
            metrics.append("table_structure_similarity")
        if categories.get("equation_isolated") or categories.get("equation_inline"):
            metrics.append("formula_normalized_edit_distance")
        cases.append(
            {
                "case_id": f"parse-{document_id}",
                "document_id": document_id,
                "path": documents[document_id]["path"],
                "split": documents[document_id]["split"],
                "language": page["page_info"]["page_attribute"]["language"],
                "page_attribute": page["page_info"]["page_attribute"],
                "expected_non_ignored_regions": sum(not region.get("ignore") for region in page["layout_dets"]),
                "expected_ordered_regions": sum(region.get("order") is not None for region in page["layout_dets"]),
                "expected_categories": dict(sorted(categories.items())),
                "ground_truth_file": "annotations/omnidocbench_zh_pages.jsonl",
                "eligible_metrics": metrics,
                "suite": "document_parsing",
            }
        )
    return cases


def build_refusal_cases() -> list[dict]:
    return [
        {
            "case_id": "refuse-01",
            "query": "宏海科技2025年已经实现的营业收入是多少？",
            "candidate_document_ids": ["workspace-192-01"],
            "gold_answer": "资料仅提供截至2024年的实际营业收入，无法确定2025年已经实现的营业收入。",
            "answerable": False,
            "reason": "requested_period_not_covered",
        },
        {
            "case_id": "refuse-02",
            "query": "3月员工生日会因天气取消后的补办日期是哪一天？",
            "candidate_document_ids": ["workspace-340-01", "workspace-340-02", "workspace-340-03"],
            "gold_answer": "现有生日会材料未说明活动取消或补办日期。",
            "answerable": False,
            "reason": "event_not_present",
        },
        {
            "case_id": "refuse-03",
            "query": "固定资产报废申请提交后承诺在几个工作日内审批完成？",
            "candidate_document_ids": ["workspace-346-01"],
            "gold_answer": "申请审批表未规定审批完成时限。",
            "answerable": False,
            "reason": "field_not_present",
        },
        {
            "case_id": "refuse-04",
            "query": "公司行政体系六大模块在2025年的实际完成率分别是多少？",
            "candidate_document_ids": ["workspace-354-01"],
            "gold_answer": "行政体系建设蓝图列出了六大模块，但未提供2025年各模块的实际完成率。",
            "answerable": False,
            "reason": "requested_measurement_not_present",
        },
    ]


def validate_retrieval(rows: list[dict], documents: dict[str, dict], blocks: dict[str, dict]) -> dict:
    errors = []
    ids = [row["question_id"] for row in rows]
    if len(ids) != len(set(ids)):
        errors.append("duplicate question_id")
    banned_prefixes = ("这页", "页面说明", "这页课件", "页面上")
    for row in rows:
        if row["query"].startswith(banned_prefixes):
            errors.append(f"deictic query: {row['question_id']}")
        if not row["answerable"] or not row["gold_answer"]:
            errors.append(f"invalid answer: {row['question_id']}")
        if not row["relevant_document_ids"]:
            errors.append(f"missing relevance: {row['question_id']}")
        for document_id in row["relevant_document_ids"]:
            if document_id not in documents:
                errors.append(f"unknown document {document_id}: {row['question_id']}")
            elif documents[document_id]["split"] != row["split"]:
                errors.append(f"split leakage {document_id}: {row['question_id']}")
        evidence_documents = {evidence["document_id"] for evidence in row["evidence"]}
        if not set(row["relevant_document_ids"]).issubset(evidence_documents):
            errors.append(f"relevant document lacks evidence: {row['question_id']}")
        for evidence in row["evidence"]:
            block_id = evidence.get("block_id")
            if block_id and block_id not in blocks:
                errors.append(f"unknown block {block_id}: {row['question_id']}")
    if errors:
        raise ValueError("\n".join(errors))
    return {
        "question_count": len(rows),
        "by_split": dict(sorted(Counter(row["split"] for row in rows).items())),
        "by_modality": dict(sorted(Counter(row["modality"] for row in rows).items())),
        "by_difficulty": dict(sorted(Counter(row["difficulty"] for row in rows).items())),
        "multi_document_count": sum(len(row["relevant_document_ids"]) > 1 for row in rows),
        "all_have_answers": all(bool(row["gold_answer"]) for row in rows),
        "all_have_evidence": all(bool(row["evidence"]) for row in rows),
        "all_relevant_documents_have_evidence": True,
    }


def validate_other_suites(
    table_qa: list[dict], parser_cases: list[dict], refusal_cases: list[dict], documents: dict[str, dict]
) -> dict:
    errors = []
    for row in table_qa:
        if not row.get("query") or not row.get("gold_answer") or not row.get("evidence"):
            errors.append(f"incomplete table QA: {row.get('question_id')}")
        if any(document_id not in documents for document_id in row.get("relevant_document_ids", [])):
            errors.append(f"unknown table QA document: {row.get('question_id')}")
        if row.get("retrieval_score_eligible") is not False:
            errors.append(f"table QA incorrectly enabled for retrieval: {row.get('question_id')}")
    for case in parser_cases:
        path = BUNDLE / case["path"]
        if case["document_id"] not in documents or not path.exists():
            errors.append(f"invalid parser case: {case.get('case_id')}")
        if case["expected_non_ignored_regions"] <= 0:
            errors.append(f"empty parser ground truth: {case.get('case_id')}")
    for case in refusal_cases:
        if case.get("answerable") is not False or not case.get("gold_answer"):
            errors.append(f"invalid refusal case: {case.get('case_id')}")
        if any(document_id not in documents for document_id in case["candidate_document_ids"]):
            errors.append(f"unknown refusal candidate: {case.get('case_id')}")
    if errors:
        raise ValueError("\n".join(errors))
    return {
        "table_qa_valid": True,
        "parser_cases_valid": True,
        "refusal_cases_valid": True,
        "locally_corrected_upstream_gold_count": sum(
            row.get("gold_verification") == "locally_corrected_against_source_table"
            for row in table_qa
        ),
    }


def write_review(rows: list[dict]) -> None:
    def esc(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "# 正式检索问题人工复核表（v1）",
        "",
        "只有本表中的问题参与 Chroma/Qdrant 的 Recall、MRR、nDCG 比较。",
        "",
        "| ID | 问题 | 标准答案 | 相关文档 | 类型 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(
                esc(value)
                for value in (
                    row["question_id"], row["query"], row["gold_answer"],
                    ", ".join(row["relevant_document_ids"]), row["question_type"],
                )
            ) + " |"
        )
    (ANN / "questions_review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    documents = {row["item_id"]: row for row in read_jsonl(BUNDLE / "manifest.jsonl")}
    blocks = {row["block_id"]: row for row in read_jsonl(ANN / "extracted_blocks.jsonl")}
    v0 = read_jsonl(ANN / "questions_v0_draft.jsonl")
    omni_pages = read_jsonl(ANN / "omnidocbench_zh_pages.jsonl")

    retrieval, excluded = build_retrieval_questions(v0, blocks)
    table_qa, table_excluded = build_table_qa(v0, blocks)
    excluded.extend(table_excluded)
    parser_cases = build_parser_cases(omni_pages, documents)
    refusal_cases = build_refusal_cases()
    validation = validate_retrieval(retrieval, documents, blocks)
    suite_validation = validate_other_suites(
        table_qa, parser_cases, refusal_cases, documents
    )

    write_jsonl(ANN / "retrieval_questions.jsonl", retrieval)
    write_jsonl(ANN / "questions.jsonl", retrieval)
    write_jsonl(ANN / "table_qa_cases.jsonl", table_qa)
    write_jsonl(ANN / "parser_cases.jsonl", parser_cases)
    write_jsonl(ANN / "refusal_cases.jsonl", refusal_cases)
    write_jsonl(ANN / "excluded_cases.jsonl", sorted(excluded, key=lambda row: row["question_id"]))
    write_review(retrieval)

    relevant_documents = {
        document_id for row in retrieval for document_id in row["relevant_document_ids"]
    }
    validation["relevant_document_count"] = len(relevant_documents)
    validation["relevant_documents_by_format"] = dict(
        sorted(Counter(documents[document_id]["format"] for document_id in relevant_documents).items())
    )
    (ANN / "questions_summary.json").write_text(
        json.dumps({"schema_version": "1.1", "suite": "retrieval", **validation}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    benchmark = {
        "schema_version": "1.1",
        "status": "verified_v1",
        "retrieval": validation,
        "table_qa_case_count": len(table_qa),
        "parser_case_count": len(parser_cases),
        "refusal_case_count": len(refusal_cases),
        "excluded_case_count": len(excluded),
        "validation": suite_validation,
        "metric_boundaries": {
            "retrieval_questions.jsonl": ["Recall@K", "MRR", "nDCG@K", "complete-evidence Recall@K"],
            "table_qa_cases.jsonl": ["answer correctness", "answer F1", "faithfulness"],
            "parser_cases.jsonl": ["layout F1", "reading-order edit distance", "text/formula normalized edit distance"],
            "refusal_cases.jsonl": ["correct rejection rate", "unsupported claim rate"],
        },
        "known_limitations": [
            "公开数据可能出现在模型预训练语料中，不能替代私有盲测集",
            "当前为单人复核，正式发布前仍建议由第二位标注员独立复核并计算一致率",
            "TableEval表格问答不用于全库检索排名，因为多数问题默认用户已在查看指定表格",
        ],
    }
    (ANN / "benchmark_manifest.json").write_text(
        json.dumps(benchmark, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    audit_lines = [
        "# 评测数据复核报告（v1）",
        "",
        "## 结论",
        "",
        f"正式检索集保留 {len(retrieval)} 题；表格问答 {len(table_qa)} 题；解析 {len(parser_cases)} 例；拒答 {len(refusal_cases)} 题。不同套件不混算总分。",
        "",
        "## 已执行检查",
        "",
        "- 问题 ID 唯一，问题、答案、相关文档和证据均非空。",
        "- 每个相关文档都有对应证据，所有 block ID 均能在抽取结果中解析。",
        "- 问题与相关文档处于同一 split，未发现跨 split 泄漏。",
        "- 对卫生统计重复事实补充了全部已知相关文档，避免正确召回被误判。",
        "- 多轮 TableEval 问题补充了独立检索表述和对话历史。",
        "- 复算并纠正了4处上游答案/表述问题，其中现金管理产品题的上游答案漏行且赎回状态错误。",
        "- 解析、全库检索、指定表格问答和拒答被拆分为独立套件。",
        "",
        "## 移出正式评分的候选题",
        "",
        "| ID | 原因 |",
        "| --- | --- |",
    ]
    for row in sorted(excluded, key=lambda value: value["question_id"]):
        audit_lines.append(f"| {row['question_id']} | {row['reason'].replace('|', '/')} |")
    audit_lines.extend(
        [
            "",
            "## 仍需保留的边界",
            "",
            "- 这是公开数据，不能排除模型预训练污染。",
            "- 当前完成的是单人逐题复核，不等同于双标注员独立复核。",
            "- chunk 级真值必须在最终切块后，依据 evidence locator 映射生成，不能预先猜测 chunk ID。",
        ]
    )
    (ANN / "audit_report.md").write_text("\n".join(audit_lines) + "\n", encoding="utf-8")
    print(json.dumps(benchmark, ensure_ascii=False))


if __name__ == "__main__":
    main()
