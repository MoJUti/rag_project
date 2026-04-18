import re
from datetime import datetime

from ingestion.legal_preprocess import (
	INLINE_ARTICLE_BREAK_PATTERN,
	LINE_ARTICLE_PATTERN,
	LINE_CHAPTER_PATTERN,
	LINE_PART_PATTERN,
	LINE_SECTION_PATTERN,
)

# 结构标题匹配（多行模式）。
PART_PATTERN = re.compile(r"(?m)^第[一二三四五六七八九十百千零〇0-9]+编\s*[^\n]*$")
CHAPTER_PATTERN = re.compile(r"(?m)^第[一二三四五六七八九十百千零〇0-9]+章\s*[^\n]*$")
SECTION_PATTERN = re.compile(r"(?m)^第[一二三四五六七八九十百千零〇0-9]+节\s*[^\n]*$")
ARTICLE_PATTERN = re.compile(r"第[一二三四五六七八九十百千零〇0-9]+\s*条")
ARTICLE_LINE_EXTRACT_PATTERN = re.compile(r"^(第[一二三四五六七八九十百千零〇0-9]+\s*条)[\s　]+(.*)$")


def _last_match_text(pattern: re.Pattern, text: str):
	matches = list(pattern.finditer(text))
	if not matches:
		return None
	return matches[-1].group(0).strip()


def _all_match_texts(pattern: re.Pattern, text: str):
	return [m.group(0).strip() for m in pattern.finditer(text)]


def parse_legal_article_units(cleaned_text: str):
	"""将法律正文解析为按“条”组织的结构化单元。"""
	normalized_text = INLINE_ARTICLE_BREAK_PATTERN.sub(r"\n\1", cleaned_text)

	units = []
	state = {"part": "", "chapter": "", "section": ""}
	current = None

	def flush_current():
		if not current:
			return
		content = "\n".join(current["lines"]).strip()
		if not content:
			return
		units.append(
			{
				"part": current["part"],
				"chapter": current["chapter"],
				"section": current["section"],
				"article_no": current["article_no"],
				"text": content,
			}
		)

	for raw_line in normalized_text.split("\n"):
		line = raw_line.strip()
		if not line:
			continue

		if LINE_PART_PATTERN.match(line):
			state["part"] = line
			state["chapter"] = ""
			state["section"] = ""
			continue
		if LINE_CHAPTER_PATTERN.match(line):
			state["chapter"] = line
			state["section"] = ""
			continue
		if LINE_SECTION_PATTERN.match(line):
			state["section"] = line
			continue

		article_match = ARTICLE_LINE_EXTRACT_PATTERN.match(line)
		if article_match:
			flush_current()
			article_no = article_match.group(1).strip()
			remain = article_match.group(2).strip()
			current = {
				"part": state["part"],
				"chapter": state["chapter"],
				"section": state["section"],
				"article_no": article_no,
				"lines": [line] if remain else [article_no],
			}
			continue

		if current is not None:
			current["lines"].append(line)

	flush_current()
	return units


def build_chunks_from_article_units(article_units: list[dict], max_chars: int, overlap_articles: int = 0):
	"""按条合并 chunk，且严格不跨章、不跨节。"""
	if not article_units:
		return [], []

	chunks = []
	metadata_list = []
	now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

	chapter_end_map = {}
	for u in article_units:
		if u["chapter"] and u["article_no"]:
			chapter_end_map[u["chapter"]] = u["article_no"]

	scoped_groups = {}
	scoped_order = []
	for u in article_units:
		chapter_key = u["chapter"] or "__UNKNOWN_CHAPTER__"
		section_key = u["section"] or "__NO_SECTION__"
		scope_key = (chapter_key, section_key)
		if scope_key not in scoped_groups:
			scoped_groups[scope_key] = []
			scoped_order.append(scope_key)
		scoped_groups[scope_key].append(u)

	for scope_key in scoped_order:
		units = scoped_groups[scope_key]
		i = 0
		n = len(units)

		while i < n:
			start = i
			end = i
			current_len = 0

			while end < n:
				t = units[end]["text"]
				add_len = len(t) + (1 if current_len > 0 else 0)
				if current_len > 0 and current_len + add_len > max_chars:
					break
				current_len += add_len
				end += 1
				if current_len >= max_chars:
					break

			if end == start:
				end = start + 1

			selected = units[start:end]
			chunk_text = "\n".join([u["text"] for u in selected]).strip()
			first = selected[0]
			last = selected[-1]

			metadata_list.append(
				{
					"source": "",
					"create_time": now,
					"operator": "",
					"part": first["part"],
					"chapter": first["chapter"],
					"section": first["section"],
					"article_no": first["article_no"],
					"article_end": chapter_end_map.get(first["chapter"], last["article_no"]),
					"chunk_article_end": last["article_no"],
				}
			)
			chunks.append(chunk_text)

			next_i = end - overlap_articles
			i = next_i if next_i > start else start + 1

	return chunks, metadata_list


def extract_chapter_article_end_map(cleaned_text: str):
	"""提取每一章对应的末条条号。"""
	chapter_end_map = {}
	current_chapter = ""

	for raw_line in cleaned_text.split("\n"):
		line = raw_line.strip()
		if not line:
			continue

		if LINE_CHAPTER_PATTERN.match(line):
			current_chapter = line
			continue

		if not current_chapter:
			continue

		article_list = _all_match_texts(ARTICLE_PATTERN, line)
		if article_list:
			chapter_end_map[current_chapter] = article_list[-1]

	return chapter_end_map


def build_legal_chunk_metadata(
	chunks: list[str],
	source: str,
	operator: str,
	chapter_article_end_map: dict[str, str] | None = None,
):
	"""为每个 chunk 构建法律层级元数据（历史兼容路径）。"""
	now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	chapter_article_end_map = chapter_article_end_map or {}
	state = {"part": None, "chapter": None, "section": None, "article_no": None}

	metadata_list = []
	for chunk in chunks:
		part = _last_match_text(PART_PATTERN, chunk)
		chapter = _last_match_text(CHAPTER_PATTERN, chunk)
		section = _last_match_text(SECTION_PATTERN, chunk)
		article_list = _all_match_texts(ARTICLE_PATTERN, chunk)

		if part:
			state["part"] = part
			state["chapter"] = ""
			state["section"] = ""
		if chapter:
			state["chapter"] = chapter
			state["section"] = ""
		if section:
			state["section"] = section
		if article_list:
			state["article_no"] = article_list[-1]

		chapter_value = state["chapter"] or ""
		chunk_article_end = article_list[-1] if article_list else ""
		chapter_article_end = chapter_article_end_map.get(chapter_value, "")

		metadata = {
			"source": source,
			"create_time": now,
			"operator": operator,
			"part": state["part"] or "",
			"chapter": chapter_value,
			"section": state["section"] or "",
			"article_no": article_list[0] if article_list else (state["article_no"] or ""),
			"article_end": chapter_article_end or chunk_article_end,
			"chunk_article_end": chunk_article_end,
		}
		metadata_list.append(metadata)

	return metadata_list


__all__ = [
	"parse_legal_article_units",
	"build_chunks_from_article_units",
	"extract_chapter_article_end_map",
	"build_legal_chunk_metadata",
]
