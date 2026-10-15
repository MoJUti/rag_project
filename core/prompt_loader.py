"""统一加载提示词文件，启动后缓存；模板数据只替换一次，不执行代码。"""

from functools import lru_cache
from pathlib import Path
from string import Template
from collections.abc import Mapping


PROMPT_ROOT = Path(__file__).resolve().parents[1] / "prompts"


@lru_cache(maxsize=None)
def load_prompt(relative_path: str, expected_variables: tuple[str, ...] = ()) -> str:
    """限定提示词目录，并严格校验文件内容及完整变量集合。"""
    path = (PROMPT_ROOT / relative_path).resolve()
    if not path.is_relative_to(PROMPT_ROOT.resolve()) or path.suffix != ".md":
        raise ValueError(f"提示词路径必须是 prompts 目录内的 Markdown 文件：{relative_path}")
    try:
        text = path.read_text(encoding="utf-8").rstrip("\r\n")
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(f"提示词文件读取失败：{relative_path}（{type(exc).__name__}）") from exc
    if not text.strip():
        raise ValueError(f"提示词文件为空：{relative_path}")
    if "\x00" in text:
        raise ValueError(f"提示词文件包含非法空字符：{relative_path}")
    template = Template(text)
    if not template.is_valid():
        raise ValueError(f"提示词变量语法错误：{relative_path}，请使用 ${{variable}}，字面美元符号写为 $$")
    actual = set(template.get_identifiers())
    if actual != set(expected_variables):
        raise ValueError(f"提示词变量不匹配：{relative_path}；预期{sorted(expected_variables)}，实际{sorted(actual)}")
    return text


def render_prompt(relative_path: str, values: Mapping[str, str], *, literal_newlines: bool = False) -> str:
    """只替换模板变量，不二次解释资料中的花括号、美元符号或模板语句。"""
    text = load_prompt(relative_path, tuple(sorted(values)))
    if literal_newlines:
        # 旧摘要任务使用字面的反斜线n；本次拆分保留历史行为，不同时改压缩逻辑。
        text = text.replace("\n", "\\n")
    return Template(text).substitute(values)
