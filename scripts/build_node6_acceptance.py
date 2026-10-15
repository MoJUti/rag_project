"""从核对过的原文生成新增验收题、等价来源和冻结记录。"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_chunking_benchmark import read_jsonl, write_jsonl

BUNDLE = ROOT / "data/eval/zh_public_minibench_v0.1"
ANN = BUNDLE / "annotations"

# 题目与答案在运行检索前确定；证据只来自原文件，不能从模型回答反推。
SPECS = [
    ("a01", "华安证券研究所的现金流量表预计2025年的投资活动现金流和现金净增加额分别是多少？", "-65百万元；2326百万元", [("tableeval-0002", "| 投资活动现金流"), ("tableeval-0002", "| 现金净增加额")], "multi_fact_lookup"),
    ("a02", "主要财务比率表预计2025年的流动比率和速动比率分别是多少？", "流动比率0.88；速动比率0.40", [("tableeval-0004", "| 流动比率"), ("tableeval-0004", "| 速动比率")], "multi_fact_lookup"),
    ("a03", "太平洋证券现金流量表中，2023年融资性现金流和现金增加额分别是多少？", "304百万；727百万", [("tableeval-0641", "| 融资性现金流"), ("tableeval-0641", "| 现金增加额")], "multi_fact_lookup"),
    ("a04", "销售费用表中的广告宣传费本期比上期减少了多少？请列出计算用的两个原始金额。", "上期13,234,292.44，本期8,624,085.81，减少4,610,206.63；原表未标明金额单位", [("tableeval-1184", "| 广告宣传费")], "arithmetic"),
    ("a05", "财务预测摘要利润表预计2024年的保费业务收入与银行业务利息净收入分别是多少？", "保费业务收入5499百万元；银行业务利息净收入2178百万元", [("tableeval-1186", "| 保费业务收入"), ("tableeval-1186", "| 银行业务利息净收入")], "multi_fact_lookup"),
    ("a06", "特斯拉自动驾驶硬件版本对比表里，HW3.0和HW4.0的算力与功耗分别是多少？", "HW3.0：144 TOPS、220 W；HW4.0：720 TOPS、80 W", [("tableeval-1187", "| HW3.0"), ("tableeval-1187", "| HW4.0")], "table_comparison"),
    ("a07", "主要区域贸易协定的中小企业条款对比中，哪项协定明确要求定期召开中小企业对话？条款号是什么？", "USMCA，第25.5条", [("tableeval-1189", "| 中小企业对话条款")], "fact_lookup"),
    ("a08", "围绕地缘利益与安全困境的叙事表中，作者概括的俄罗斯与乌克兰叙事主题分别是什么？", "表中概括：俄罗斯遭遇外敌入侵，被迫发起反击；乌克兰是遭遇侵略的受害者。这是表中的叙事归纳。", [("tableeval-1322", "| 俄罗斯"), ("tableeval-1322", "| 乌克兰")], "attributed_summary"),
    ("a09", "采用未来6个月公司评级口径的投资评级声明，对买入和持有分别设定了什么相对基准指数涨幅条件？", "买入：20%以上；持有：10%与20%之间", [("tableeval-1323", "| 公司评级 | 买入"), ("tableeval-1323", "|      | 持有")], "multi_fact_lookup"),
    ("a10", "采用未来6至12个月股票评级口径的说明中，买入和增持的相对基准指数涨幅条件分别是什么？", "买入：15%以上；增持：5%至15%之间", [("tableeval-1326", "| 股票评级"), ("tableeval-1326", "| 增持 | 预期未来6~12个月内相对同期基准指数涨幅")], "multi_fact_lookup"),
    ("a11", "主要会计数据表为何追溯调整上年同期数据？涉及哪家公司哪个季度？", "同一控制收购山东高速信息集团有限公司，追溯调整其2023年第1季度财务数据", [("tableeval-2011", "追溯调整同一控制收购")], "fact_lookup"),
    ("a12", "合肥维信诺拟实施的资产重组方案中，拟收购股权比例是多少，对应已实缴和未实缴注册资本分别是多少？", "40.91%；已实缴59.60亿元；尚未实缴30.40亿元", [("tableeval-2041", "本次公司拟通过发行股份及支付现金")], "multi_fact_lookup"),
    ("a13", "不同银行处置不良贷款方式表中，2023年邮储银行与招商银行现金清收金额分别是多少？", "邮储银行218.3亿元；招商银行112.0亿元", [("tableeval-2042", "| 现金清收")], "table_comparison"),
    ("a14", "行列式例2.62的拉普拉斯展开论证中，为什么k阶子式为零？", "因为k阶子式必有一列全为0", [("omnidoc-01", "2.0.38")], "formula_reasoning"),
    ("a15", "线性时不变系统课件用什么输入与响应关系说明叠加原理？", "系统输入线性和的响应等于响应的线性和", [("omnidoc-03", "系统输入线性和的响应")], "concept_lookup"),
    ("x01", "华安证券与太平洋证券的两份现金流量表，2025年预计现金净增加额各是多少？不要将它们视为同一家公司的连续数据。", "华安证券表2326百万元；太平洋证券表487百万", [("tableeval-0002", "| 现金净增加额"), ("tableeval-0641", "| 现金增加额")], "cross_document_comparison"),
    ("x02", "两份投资评级说明对公司或股票买入的收益阈值与时间窗口分别如何规定？请分别说明未来6个月口径和未来6至12个月口径。", "未来6个月口径：相对基准指数涨幅20%以上；未来6至12个月口径：相对同期基准指数涨幅15%以上", [("tableeval-1323", "| 公司评级 | 买入"), ("tableeval-1326", "| 股票评级")], "cross_document_comparison"),
    ("x03", "两份投资评级说明对行业增持或强于大市的相对收益要求各是多少？请说明它们的时间窗口。", "未来6个月强于大市：高于基准指数5%以上；未来6至12个月行业增持：对同期基准指数涨幅10%以上", [("tableeval-1323", "| 行业评级"), ("tableeval-1326", "| 行业评级")], "cross_document_comparison"),
    ("x04", "对照主要财务比率表与财务预测摘要利润表，2025年预计速动比率和保费业务收入分别是多少？它们是否使用相同计量单位？", "速动比率0.40，是比率；保费业务收入5673百万元，是金额，单位不同", [("tableeval-0004", "| 速动比率"), ("tableeval-1186", "| 保费业务收入")], "cross_document_synthesis"),
]

REFUSALS = [
    ("r01", "华安证券研究所这份现金流量表预计2027年的财务费用是多少？", "该表只列到2026年，未提供2027年的财务费用", "tableeval-0002", "period_absent"),
    ("r02", "特斯拉硬件版本对比表给出的HW4.0图像处理能力是多少fps？", "HW4.0的图像处理能力标为破折号，不能据此给出数值", "tableeval-1187", "value_missing"),
    ("r03", "这份销售费用表说明了广告宣传费下降的具体业务原因吗？", "表中只有本期和上期金额，没有说明下降的具体业务原因", "tableeval-1184", "explanation_absent"),
    ("r04", "合肥维信诺这份拟实施的资产重组方案给出了交易最终完成的日期吗？", "这份方案说明拟实施的交易安排，未给出最终完成日期", "tableeval-2041", "outcome_absent"),
]


def build() -> list[dict]:
    manifest = {row["item_id"]: row for row in read_jsonl(BUNDLE / "manifest.jsonl")}
    pages = {row["bundle_item_id"]: row for row in read_jsonl(ANN / "omnidocbench_zh_pages.jsonl")}
    rows = []
    for key, query, answer, pairs, kind in SPECS:
        evidence = []
        groups = []
        for item_id, fragment in pairs:
            item = manifest[item_id]
            equivalent = [name for name, other in manifest.items() if other["sha256"] == item["sha256"]]
            if equivalent not in groups:
                groups.append(equivalent)
            if item_id in pages:
                regions = [region for region in pages[item_id]["layout_dets"]
                           if fragment in region.get("text", "")]
                if len(regions) != 1:
                    raise ValueError(f"原图区域无法唯一定位：{item_id}/{fragment}")
                region = regions[0]
                quote = region["text"]
                locator = {"type": "page_region", "page": 1, "polygon": region["poly"]}
                contexts = []
            else:
                lines = (BUNDLE / item["path"]).read_text(encoding="utf-8").splitlines()
                matches = [(number, line) for number, line in enumerate(lines, 1) if fragment in line]
                if len(matches) != 1:
                    raise ValueError(f"原文证据无法唯一定位：{item_id}/{fragment}")
                number, quote = matches[0]
                locator = {"type": "line", "line": number}
                # 表格必须同时保留列头，防止只命中数字而列归属错误。
                contexts = [line for line in lines[:4] if line.startswith("|") and "---" not in line]
                if item_id == "tableeval-2042":
                    contexts = [line for line in lines if line.startswith("| 处置方式") or "|         | 2021" in line]
            evidence.append({"document_ids": equivalent, "document_id": item_id,
                             "quote": quote, "context_quotes": contexts, "locator": locator,
                             "source_path": item["path"], "source_sha256": item["sha256"]})
        rows.append({"question_id": f"accept-v1-{key}", "query": query, "gold_answer": answer,
                     "accepted_answers": [answer], "answerable": True,
                     "question_type": kind, "split": "acceptance-new-v1",
                     "relevant_document_groups": groups,
                     "relevant_document_ids": list(dict.fromkeys(doc for group in groups for doc in group)),
                     "evidence": evidence})
    for key, query, answer, item_id, reason in REFUSALS:
        item = manifest[item_id]
        rows.append({"question_id": f"accept-v1-{key}", "query": query, "gold_answer": answer,
                     "accepted_answers": [answer], "answerable": False, "split": "acceptance-new-v1",
                     "question_type": "refusal", "relevant_document_ids": [], "evidence": [],
                     "scope_document_id": item_id, "scope_source_path": item["path"],
                     "reason": reason, "absence_review": "已检查指定原文全文；拒答范围限于问题明确指定的资料"})
    return rows


def main() -> None:
    rows = build()
    previous = read_jsonl(ANN / "retrieval_questions.jsonl") + read_jsonl(ANN / "table_qa_cases.jsonl")
    if {row["query"] for row in rows} & {row["query"] for row in previous}:
        raise ValueError("新增题与既有题完全重复")
    output = ANN / "acceptance_questions_v1.jsonl"
    if output.exists() and read_jsonl(output) != rows:
        raise ValueError("验收题已经冻结，修改时需新建版本，禁止覆盖旧结果对应的题目")
    write_jsonl(output, rows)
    frozen = {"version": "acceptance-new-v1", "frozen_on": "2026-10-14",
              "questions_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
              "question_count": len(rows), "answerable_count": sum(row["answerable"] for row in rows),
              "locked_profiles": ["dense", "bm25_mixed", "hybrid_mixed", "hybrid_mixed_qwen37"],
              "vector_k": 20, "bm25_k": 20, "rrf_k": 60, "candidate_k": 40, "final_k": 6,
              "note": "同一资料上的新增未调参问题；非外部盲测，看结果后转为回归集，拒答不计算Recall"}
    (ANN / "acceptance_v1.lock.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")
    content = ["# 节点6新增验收题", "", "这些问题在新一轮检索前冻结。共19道有答案题、4道资料不足题。",
               "已有55题保持独立；本轮不是外部盲测，结果查看后用于回归测试。", ""]
    for row in rows:
        content.extend([f"## {row['question_id']}", "", row["query"], "", f"标准答案：{row['gold_answer']}", ""])
        for evidence in row["evidence"]:
            content.extend([f"来源：{evidence['document_id']}；定位：{evidence['locator']}。", "",
                            f"证据原文：{evidence['quote']}", ""])
        if not row["answerable"]:
            content.extend([f"拒答核对范围：{row['scope_source_path']}。", ""])
    (ROOT / "docs/节点6新增验收题.md").write_text("\n".join(content), encoding="utf-8")
    print(json.dumps(frozen, ensure_ascii=False))


if __name__ == "__main__":
    main()
