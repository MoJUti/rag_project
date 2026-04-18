import os
import sys
import tempfile
from contextlib import contextmanager
from unittest.mock import patch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from langchain_core.messages import AIMessage, HumanMessage

from core import config
from memory import history_store as hs


@contextmanager
def temp_config(**kwargs):
    backup = {}
    for key, value in kwargs.items():
        backup[key] = getattr(config, key)
        setattr(config, key, value)
    try:
        yield
    finally:
        for key, value in backup.items():
            setattr(config, key, value)


def build_round_messages(rounds: int, chars_per_msg: int):
    messages = []
    for idx in range(1, rounds + 1):
        messages.append(HumanMessage(content=f"用户第{idx}轮:" + ("U" * chars_per_msg)))
        messages.append(AIMessage(content=f"助手第{idx}轮:" + ("A" * chars_per_msg)))
    return messages


def show_case(title: str, messages):
    compressed = hs._compress_messages(messages)
    total_tokens = hs._estimate_messages_tokens(compressed)
    rounds = sum(1 for m in compressed if m.type == "human")
    has_summary = any(hs._is_summary_message(m) for m in compressed)

    print("=" * 72)
    print(f"[CASE] {title}")
    print(f"压缩后消息数: {len(compressed)}")
    print(f"压缩后轮次数: {rounds}")
    print(f"是否包含摘要: {has_summary}")
    print(f"估算总 token: {total_tokens}")
    print("压缩后消息类型:", [m.type for m in compressed])


def run_file_history_demo():
    with tempfile.TemporaryDirectory() as tmpdir:
        with temp_config(chat_history_directory=tmpdir):
            history = hs.FileChatMessageHistory("chat_demo", tmpdir)
            # 先写入 7 轮，触发摘要。
            history.add_messages(build_round_messages(7, 40))
            # 再追加 1 轮，观察摘要与最近轮次保留策略。
            history.add_messages(build_round_messages(1, 30))

            loaded = history.messages
            print("=" * 72)
            print("[FILE HISTORY DEMO] 本地历史文件写入后状态")
            print(f"历史消息条数: {len(loaded)}")
            print("消息类型:", [m.type for m in loaded])
            print("是否存在摘要:", any(hs._is_summary_message(m) for m in loaded))
            print("会话文件:", os.path.join(tmpdir, "chat_demo"))


def main():
    with temp_config(
        memory_keep_recent_rounds=4,
        memory_summary_trigger_rounds=6,
        memory_history_max_tokens=1500,
        memory_summary_max_chars=800,
        memory_summary_enabled=True,
        memory_summary_tag="[SESSION_SUMMARY]",
    ):
        # 通过 mock 固定摘要输出，避免验证时依赖外部模型服务。
        with patch("memory.history_store._summarize_messages", return_value="[mock] 这是压缩摘要"):
            show_case("短会话，不触发摘要", build_round_messages(rounds=3, chars_per_msg=20))
            show_case("达到阈值，触发摘要", build_round_messages(rounds=8, chars_per_msg=20))

        with patch("memory.history_store._summarize_messages", return_value="[mock] 短摘要"):
            with temp_config(memory_history_max_tokens=180):
                show_case("超预算，触发删旧轮次", build_round_messages(rounds=10, chars_per_msg=120))

        with patch("memory.history_store._summarize_messages", return_value="[mock] 文件摘要"):
            run_file_history_demo()

    print("=" * 72)
    print("验证完成：你可以根据输出确认摘要触发、轮次保留、预算裁剪行为。")


if __name__ == "__main__":
    main()
