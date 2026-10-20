"""聊天页单次流程查看组件；仅渲染已有结果，不触发摘要、检索或模型调用。"""

import streamlit as st

from generation.observability import settings_snapshot


STAGE_LABELS = {"history_query": "历史查询消歧", "query": "查询处理", "recall": "候选召回（管线输出）",
                "fusion": "候选融合", "rerank": "重排", "evidence": "最终证据"}


def render_settings() -> None:
    settings = settings_snapshot()
    with st.expander("当前运行配置", expanded=False):
        st.caption("配置首选模型不等于本次实际调用模型；实际模型见回答后的流程记录。此处只读，不修改策略。")
        st.dataframe([
            {"项目": "向量库 / 集合", "配置": f"{settings['vector_store_backend']} / {settings['collection_name']}"},
            {"项目": "嵌入模型 / 维度", "配置": f"{settings['embedding_model_name']} / {settings['embedding_dimensions']}"},
            {"项目": "召回 / 分词", "配置": f"{settings['retrieval_strategy']} / {settings['bm25_tokenizer']}"},
            {"项目": "向量 / BM25候选数", "配置": f"{settings['hybrid_vector_k']} / {settings['hybrid_bm25_k']}"},
            {"项目": "最终块数 / 重排", "配置": f"{settings['hybrid_final_k']} / {settings['rerank_strategy']}"},
            {"项目": "查询消歧 / 证据扩展", "配置": f"{settings['query_resolution_strategy']} / {settings['evidence_expansion']}"},
            {"项目": "配置首选回答模型 / 备用", "配置": f"{settings['chat_model_name']} / {settings['chat_fallback_provider']}"},
            {"项目": "历史 / 摘要预算", "配置": f"{settings['memory_history_max_tokens']} / {settings['memory_summary_max_tokens']} 估算Token"},
        ], hide_index=True, use_container_width=True)


def render_trace(report: dict | None) -> None:
    if not report:
        st.caption("提问后可查看本次流程。旧聊天记录没有历史流程快照，不会重新调用模型补生成。")
        return
    with st.expander("最近一次问答流程", expanded=False):
        status = {"completed": "已完成", "failed": "失败", "interrupted": "中断"}.get(report.get("status"), "进行中")
        columns = st.columns(3)
        columns[0].metric("请求状态", status)
        first = report.get("first_text_seconds")
        columns[1].metric("收到首段正文", f"{first:.2f} 秒" if first is not None else "尚未输出")
        columns[2].metric("本次总耗时", f"{report.get('total_seconds', 0):.2f} 秒")
        st.caption("总耗时含历史组装、检索与输出；首字为客户端收到首个文本片段的时间，不是服务端精确首Token时间。")
        if report.get("error_type"):
            st.warning(f"本次请求未完成：{report['error_type']}。下方仅展示已采集阶段，不代表流程成功。")
        prepared = report.get("prepared")
        if prepared is None:
            st.info("尚未采集到检索准备完成记录，不能把它理解成知识库没有资料。")
            return
        query_tab, stages_tab, evidence_tab, model_tab = st.tabs(["检索问题", "阶段耗时", "证据原文", "模型与历史"])
        with query_tab:
            decision = prepared["query_resolution"]
            st.text_area("原始问题", decision["original_query"], disabled=True, key=f"original_{report['request_id']}")
            st.text_area("实际检索问题", decision["query"], disabled=True, key=f"resolved_{report['request_id']}")
            st.caption(f"消歧动作：{decision['action']}；本次配置：{report['settings']['query_resolution_strategy']}")
            if decision.get("clarification"):
                st.info(decision["clarification"])
            if decision.get("anchors"):
                st.json(decision["anchors"])
        with stages_tab:
            st.dataframe([{"阶段": STAGE_LABELS.get(stage["stage"], stage["stage"]),
                           "耗时（毫秒）": stage["elapsed_ms"], "块数": len(stage.get("chunk_ids", [])),
                           "实际查询": "；".join(stage.get("queries", []))}
                          for stage in prepared["stages"]], hide_index=True, use_container_width=True)
            st.caption("此表为检索管线阶段；生成和历史处理包含在总耗时中，不能将表内之和当作整次请求耗时。")
        with evidence_tab:
            evidence = prepared["evidence"]
            if not evidence:
                st.info("本次没有进入回答的证据；可能是澄清分支或检索无命中，请结合检索问题查看。")
            else:
                index = st.selectbox("选择进入回答的证据", range(len(evidence)),
                    format_func=lambda i: f"[{evidence[i]['citation_id']}] {evidence[i]['source']}",
                    key=f"evidence_{report['request_id']}")
                item = evidence[index]
                st.caption(f"切块ID：{item['chunk_id']}；文档ID：{item['document_id']}")
                st.text(item["text"])
                st.json(item["locations"])
                st.caption("这是进入回答的证据，不表示模型一定引用或正确理解了每一块。")
        with model_tab:
            history = prepared["history"]
            st.write({"历史消息条数": history["message_count"], "历史估算Token": history["estimated_tokens"],
                      "本次包含摘要": history["has_summary"]})
            st.caption("只统计送入本次回答的历史，不含当前问题、系统提示与知识库证据；不展示完整历史或摘要正文。")
            calls = report.get("model_calls", [])
            if calls:
                st.dataframe([{"阶段": "回答" if c["phase"] == "answer" else "其他模型调用（阶段未细分）",
                    "状态": {"completed": "已完成", "failed": "失败", "started": "未结束"}.get(c["status"], c["status"]),
                    "模型": c.get("model_name", "未采集"),
                    "输入Token": c.get("usage", {}).get("prompt_tokens", c.get("usage", {}).get("input_tokens")),
                    "输出Token": c.get("usage", {}).get("completion_tokens", c.get("usage", {}).get("output_tokens")),
                    "结束原因": c.get("finish_reason", "未采集")}
                    for c in calls], hide_index=True, use_container_width=True)
                st.caption("Token为接口返回用量；空白表示未采集，不代表零。其他模型调用可能包含摘要或查询消歧，本批尚未细分。")
            else:
                st.info("未采集到模型调用记录。不能据此推断费用为零；澄清或无证据分支可能不调用回答模型。")
