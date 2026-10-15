"""可切换的中文词法处理，供正式检索和评测共用。"""

import re


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[._/%+-][A-Za-z0-9]+)*|[\u4e00-\u9fff]+")


def char_tokenize(text: str) -> list[str]:
    """使用字符级词元，兼容旧索引行为。"""
    return [char for char in text if not char.isspace()]


def mixed_tokenize(text: str) -> list[str]:
    """保留英文数字整体，并组合相邻汉字的单字和双字词元。"""
    tokens: list[str] = []
    for match in TOKEN_PATTERN.finditer(text.lower()):
        value = match.group()
        if "\u4e00" <= value[0] <= "\u9fff":
            tokens.extend(value)
            tokens.extend(value[index:index + 2] for index in range(len(value) - 1))
        else:
            tokens.append(value)
    return tokens


TOKENIZERS = {"char": char_tokenize, "mixed": mixed_tokenize}
