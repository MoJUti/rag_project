import re

# 行级匹配。用于状态机式扫描正文结构（编/章/节/条）。
LINE_PART_PATTERN = re.compile(r"^第[一二三四五六七八九十百千零〇0-9]+编\s*.*$")
LINE_CHAPTER_PATTERN = re.compile(r"^第[一二三四五六七八九十百千零〇0-9]+章\s*.*$")
LINE_SECTION_PATTERN = re.compile(r"^第[一二三四五六七八九十百千零〇0-9]+节\s*.*$")
# 只把“条号后有空白”的行识别为法条标题，避免误识别“第X条规定”。
LINE_ARTICLE_PATTERN = re.compile(r"^第[一二三四五六七八九十百千零〇0-9]+\s*条[\s　]+")
INLINE_ARTICLE_BREAK_PATTERN = re.compile(r"(?<!\n)(第[一二三四五六七八九十百千零〇0-9]+\s*条)(?=[\s　])")


def preprocess_legal_text(text: str) -> str:
	"""清洗法律文本中的目录/页眉等噪声，保留正文结构。"""
	if not text:
		return text

	normalized = text.replace("\r\n", "\n")

	# 优先从首个“编”标题开始，其次从首个法条开始。
	part_start = re.search(r"\n第[一二三四五六七八九十百千零〇0-9]+编", normalized)
	if part_start:
		normalized = normalized[part_start.start() + 1 :]
	else:
		article_start = re.search(r"\n第[一二三四五六七八九十百千零〇0-9]+\s*条", normalized)
		if article_start:
			normalized = normalized[article_start.start() + 1 :]

	noise_patterns = [
		r"^目\s*录\s*$",
		r"^中华人民共和国刑法\s*$",
		r"^\（.*?\）$",
	]
	noise_res = [re.compile(p) for p in noise_patterns]

	cleaned_lines = []
	for raw in normalized.split("\n"):
		line = raw.strip()
		if not line:
			continue
		if any(pattern.match(line) for pattern in noise_res):
			continue
		cleaned_lines.append(line)

	# 删除目录区块污染：仅保留正文首条法条前最近一次出现的编/章/节上下文。
	if cleaned_lines:
		pre_context = {"part": "", "chapter": "", "section": ""}
		body_lines = []
		found_first_article = False
		for line in cleaned_lines:
			if not found_first_article:
				if LINE_PART_PATTERN.match(line):
					pre_context["part"] = line
					pre_context["chapter"] = ""
					pre_context["section"] = ""
					continue
				if LINE_CHAPTER_PATTERN.match(line):
					pre_context["chapter"] = line
					pre_context["section"] = ""
					continue
				if LINE_SECTION_PATTERN.match(line):
					pre_context["section"] = line
					continue
				if LINE_ARTICLE_PATTERN.match(line):
					found_first_article = True
					if pre_context["part"]:
						body_lines.append(pre_context["part"])
					if pre_context["chapter"]:
						body_lines.append(pre_context["chapter"])
					if pre_context["section"]:
						body_lines.append(pre_context["section"])
					body_lines.append(line)
			else:
				body_lines.append(line)

		if body_lines:
			cleaned_lines = body_lines

	cleaned = "\n".join(cleaned_lines)
	cleaned = re.sub(r"\n{2,}", "\n", cleaned)
	return cleaned.strip()


__all__ = [
	"preprocess_legal_text",
	"LINE_PART_PATTERN",
	"LINE_CHAPTER_PATTERN",
	"LINE_SECTION_PATTERN",
	"LINE_ARTICLE_PATTERN",
	"INLINE_ARTICLE_BREAK_PATTERN",
]
