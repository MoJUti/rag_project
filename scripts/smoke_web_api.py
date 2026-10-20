"""新版工作台最小验收；默认仅检查服务，--live才创建测试会话并真实提问一次。"""

import argparse
import json
from pathlib import Path
import sys
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.run_node6_rerank_eval import save_cache


def main(base_url: str, live: bool) -> None:
    with requests.Session() as client:
        client.trust_env = False
        health = client.get(base_url + "/api/health", timeout=10)
        health.raise_for_status()
        page = client.head(base_url + "/", timeout=10)
        page.raise_for_status()
        print({"health": health.json(), "frontend_status": page.status_code, "live": live}, flush=True)
        if not live:
            return
        response = client.post(base_url + "/api/sessions", timeout=10)
        response.raise_for_status()
        session = response.json()["id"]
        query = "3月员工生日会通知中的活动地点在哪里？"
        started = time.perf_counter()
        events, event_name, parts = [], "", []
        with client.post(base_url + f"/api/sessions/{session}/chat", json={"query": query, "provider": "zhipu"}, stream=True, timeout=(10, 120)) as stream:
            stream.raise_for_status()
            stream.encoding = "utf-8"
            for line in stream.iter_lines(decode_unicode=True):
                if line.startswith("event: "):
                    event_name = line[7:]
                elif line.startswith("data: "):
                    data = json.loads(line[6:])
                    events.append({"event": event_name, "data": data})
                    if event_name == "token":
                        parts.append(data["text"])
        history = client.get(base_url + f"/api/sessions/{session}/messages", timeout=10)
        history.raise_for_status()
        report = {"session_id": session, "query": query, "events": events, "answer": "".join(parts),
                  "elapsed_seconds": time.perf_counter() - started, "messages": history.json()}
        report["passed"] = bool(events and events[-1]["event"] == "done" and "公司活动室" in report["answer"]
                                and len(report["messages"]) == 2 and any(e["event"] == "prepared" for e in events))
        output = ROOT / f"storage/evaluations/web_ui_v1/reports/{session}.json"
        save_cache(output, report)
        print(json.dumps({"passed": report["passed"], "answer": report["answer"], "report": str(output)}, ensure_ascii=False), flush=True)
        if not report["passed"]:
            raise RuntimeError("新版问答烟测未通过，已保存记录；不自动重复付费请求")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    main(args.base_url.rstrip("/"), args.live)
