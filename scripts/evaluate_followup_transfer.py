"""补充同一追问、不同历史对象的真实检索对照，复用向量而不重复已有回答。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import config
from generation.prompts import answer_prompt_sha256
from memory.session_context import fingerprint
from scripts.evaluate_memory_behaviors import run_retrieval
from scripts.run_node6_rerank_eval import save_cache


def main(source: Path, live: bool) -> None:
    previous = json.loads(source.read_text(encoding="utf-8"))
    fixture = json.loads((ROOT / "data/eval/session_memory_v1/followup_transfer.json").read_text(encoding="utf-8"))
    baseline = previous["identity"]
    if (baseline["model"], baseline["base_url"], baseline["rag_prompt_sha256"]) != (
        config.zhipu_chat_model, config.zhipu_base_url, answer_prompt_sha256()):
        raise ValueError("模型或回答提示词已改变，不能直接比较已有结果")
    if any(getattr(config, key) != value for key, value in baseline["retrieval_config"].items()):
        raise ValueError("检索配置已改变，不能复用基线")
    identity = {"protocol": "followup-entity-transfer-v1", "fixture": fixture,
                "baseline_path": str(source.resolve()), "baseline_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "model": baseline["model"], "base_url": baseline["base_url"], "reasoning_effort": "low",
                "code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
                    "scripts/evaluate_followup_transfer.py", "scripts/evaluate_memory_behaviors.py")}}
    output = source.parent / f"followup_transfer_{fingerprint(identity)[:12]}.json"
    report = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {
        "identity": identity, "followup_vectors": previous["followup_vectors"],
        "catalog_sha256": previous["catalog_sha256"]}
    if report["identity"] != identity:
        raise ValueError("补充实验身份与缓存不同")
    print(f"report={output}; live={live}; planned_answer_calls=2; reused_followup_vectors=True", flush=True)
    if live:
        save_cache(output, report)
        run_retrieval(report, output, fixture)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "storage/evaluations/session_memory_v1/reports/behaviors_fe4fc7ec3865.json")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    main(args.source, args.live)
