"""知识库入库模块。

本模块负责三件事：
1) 对原始法律文本做预处理（去目录噪声、统一换行）。
2) 将文本解析为“条”级结构，再按长度组装为 chunk。
3) 为 chunk 生成可过滤的元数据并写入 Chroma 向量库。

说明：本文件中保留了两类元数据构建路径：
- build_chunks_from_article_units：当前主路径（推荐，结构化更强）。
- build_legal_chunk_metadata：历史兼容路径（仍可用于对照或迁移）。
"""
import os
import  config_data as config
import  hashlib
import re
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from datetime import datetime


# 结构标题匹配（多行模式）。用于从 chunk 中回溯最近一次结构标题。
PART_PATTERN = re.compile(r"(?m)^第[一二三四五六七八九十百千零〇0-9]+编\s*[^\n]*$")
CHAPTER_PATTERN = re.compile(r"(?m)^第[一二三四五六七八九十百千零〇0-9]+章\s*[^\n]*$")
SECTION_PATTERN = re.compile(r"(?m)^第[一二三四五六七八九十百千零〇0-9]+节\s*[^\n]*$")
ARTICLE_PATTERN = re.compile(r"第[一二三四五六七八九十百千零〇0-9]+\s*条")

# 行级匹配。用于“状态机式”扫描正文结构（编/章/节/条）。
LINE_PART_PATTERN = re.compile(r"^第[一二三四五六七八九十百千零〇0-9]+编\s*.*$")
LINE_CHAPTER_PATTERN = re.compile(r"^第[一二三四五六七八九十百千零〇0-9]+章\s*.*$")
LINE_SECTION_PATTERN = re.compile(r"^第[一二三四五六七八九十百千零〇0-9]+节\s*.*$")
# 只把“条号后有空白”的行识别为法条标题，避免把“第X条规定”误识别为新法条。
LINE_ARTICLE_PATTERN = re.compile(r"^第[一二三四五六七八九十百千零〇0-9]+\s*条[\s　]+")
ARTICLE_LINE_EXTRACT_PATTERN = re.compile(r"^(第[一二三四五六七八九十百千零〇0-9]+\s*条)[\s　]+(.*)$")
INLINE_ARTICLE_BREAK_PATTERN = re.compile(r"(?<!\n)(第[一二三四五六七八九十百千零〇0-9]+\s*条)(?=[\s　])")


def add_texts_in_batches(chroma, texts, metadatas, batch_size: int = 10):
    """按 DashScope 允许的上限分批写入 Chroma。"""
    if len(texts) != len(metadatas):
        raise ValueError("texts 和 metadatas 数量必须一致")

    for start in range(0, len(texts), batch_size):
        end = start + batch_size
        chroma.add_texts(
            texts[start:end],
            metadatas=metadatas[start:end],
        )


def preprocess_legal_text(text: str) -> str:
    """清洗法律文本中的目录/页眉等噪声，保留正文结构。

    处理策略：
    - 统一换行符，便于后续正则匹配。
    - 尝试从首个“编”或“条”开始截断，剔除前置噪声区块。
    - 去除目录、主标题、修订说明等非正文行。
    - 在正文首个法条前，保留最近一次编/章/节上下文，防止结构丢失。
    """
    if not text:
        return text

    normalized = text.replace("\r\n", "\n")

    # 优先从首个“编”标题开始，其次从首个法条开始，尽量跳过目录与前言信息
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
        r"^\（.*?\）$",  # 标题下方的修订说明行
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

    # 删除目录区块污染：仅保留正文首条法条前最近一次出现的编/章/节上下文
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

    # 压缩连续空行，统一文本形态
    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)
    return cleaned.strip()


def _last_match_text(pattern: re.Pattern, text: str):
    """返回文本中最后一次匹配内容，常用于“就近结构继承”场景。"""
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    return matches[-1].group(0).strip()


def _all_match_texts(pattern: re.Pattern, text: str):
    """提取文本中全部匹配结果并去空白。"""
    return [m.group(0).strip() for m in pattern.finditer(text)]


def parse_legal_article_units(cleaned_text: str):
    """将法律正文解析为按“条”组织的结构化单元。

    Returns:
        list[dict]: 每个元素包含：
            - part/chapter/section: 当前法条所在层级
            - article_no: 法条号
            - text: 法条完整文本（含后续续行）

    关键点：
    - 先做行内条号断行，降低来源文本排版不规范带来的漏切风险。
    - 用状态机维护当前层级，确保每条都继承正确结构上下文。
    """
    normalized_text = INLINE_ARTICLE_BREAK_PATTERN.sub(r"\n\1", cleaned_text)

    units = []
    state = {"part": "", "chapter": "", "section": ""}
    current = None

    def flush_current():
        """将当前聚合中的法条刷入结果列表。"""
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
    """按条合并 chunk，且严格不跨章、不跨节。

    Args:
        article_units: 条级结构化单元。
        max_chars: 单个 chunk 目标字符上限。
        overlap_articles: 条级重叠数量。默认 0 表示无重叠。

    Returns:
        tuple[list[str], list[dict]]:
            - chunks: 分块后的文本
            - metadata_list: 与 chunks 一一对应的元数据

    元数据语义约定：
    - article_no: 当前 chunk 首条
    - chunk_article_end: 当前 chunk 末条
    - article_end: 当前 chapter 末条（章级真值）
    """
    if not article_units:
        return [], []

    chunks = []
    metadata_list = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 预计算每章最后一条，后续给每个 chunk 回填章级上界。
    chapter_end_map = {}
    for u in article_units:
        if u["chapter"] and u["article_no"]:
            chapter_end_map[u["chapter"]] = u["article_no"]

    # 先按 (chapter, section) 分组，天然保证 chunk 不跨节。
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
    """提取每一章对应的末条条号，例如：第一章 -> 第十二条。"""
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
    """为每个 chunk 构建法律层级元数据（编/章/节/条）。

    注意：该函数是历史兼容路径。
    当前主流程已迁移到 parse_legal_article_units + build_chunks_from_article_units，
    但该函数仍保留，便于做新旧策略对照和回归测试。
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    chapter_article_end_map = chapter_article_end_map or {}
    state = {
        "part": None,
        "chapter": None,
        "section": None,
        "article_no": None,
    }

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
            # article_end 使用“章节末条”语义，便于章级过滤与展示。
            "article_end": chapter_article_end or chunk_article_end,
            # 额外保留分片真实末条，便于调试切片覆盖范围。
            "chunk_article_end": chunk_article_end,
        }
        metadata_list.append(metadata)

    return metadata_list

def check_md5(md5_str:str):
    """检查文本哈希是否已入库。

    Returns:
        bool: True 表示已处理过；False 表示未处理。
    """
    if not os.path.exists(config.md5_path):
        # if 进入表示文件不存在，表示没有处理过这个MD5
        open(config.md5_path,'w',encoding='utf-8').close()
        return False
    else:
        for line in open(config.md5_path,'r',encoding='utf-8').readlines():
            line=line.strip()   # 处理字符串前后的空格和回车
            if line == md5_str:
                return True     # 已处理过
        return False

def save_md5(md5_str:str):
    """保存已入库文本的 md5 哈希。"""
    with open(config.md5_path,'a',encoding="utf-8")as f:
        f.write(md5_str + '\n')

def get_string_md5(input_str:str ,encoding='utf-8'):
    """将传入字符串转换为 md5 十六进制摘要。"""

    # 将字符串转换为bytes字节数组
    str_bytes = input_str.encode(encoding=encoding)

    # 创建md5 对象
    md5_obj =hashlib.md5()      # 得到md5对象
    md5_obj.update(str_bytes)   # 更新内容（传入即将要转换的字节数组）
    md5_hex=md5_obj.hexdigest() # 得到md5的十六进制字符串

    return md5_hex


class KnowledgeBaseService(object):
    """知识库入库服务。

    当前入库主流程：
    1) 预处理文本
    2) 条级解析
    3) 条级组块
    4) 写入向量库并落地去重哈希
    """
    def __init__(self):
        # 如果文件夹不存在则创建，如果存在则跳过
        os.makedirs(config.persist_directory,exist_ok=True)

        self.chroma=Chroma(          # 向量存储的示例 Chroma向量库对象
            collection_name=config.collection_name,      #数据库表名
            embedding_function=DashScopeEmbeddings(model="text-embedding-v4"),
            persist_directory=config.persist_directory,   #数据库本地存储文件夹
        )      # 向量存储的实例，Chroma向量库对象
        self.chunk_size = config.chunk_size
        # 默认关闭条级重叠，避免测试/检索时出现明显重复段。
        self.chunk_overlap_articles = 0

    def upload_by_str(self,data:str,filename):
        """将传入字符串向量化并写入向量数据库。"""
        data = preprocess_legal_text(data)

        # 先得到出传入的字符串的md5值
        md5_hex=get_string_md5(data)
        if check_md5(md5_hex):
            return "[Repeat] 内容已存在知识库"
        # 先做条级结构解析，再组块，避免字符硬切引发的章节漂移。
        article_units = parse_legal_article_units(data)
        knowledge_chunks, metadata_list = build_chunks_from_article_units(
            article_units=article_units,
            max_chars=self.chunk_size,
            overlap_articles=self.chunk_overlap_articles,
        )

        if not knowledge_chunks:
            return "[Warn] 未解析到可入库的内容"

        for metadata in metadata_list:
            metadata["source"] = filename

        # DashScope 的 embeddings 一次最多只能处理 10 条 input.contents。
        # 这里改成分批写入，避免大文件在入库时直接报 400。
        add_texts_in_batches(
            self.chroma,
            knowledge_chunks,
            metadata_list,
            batch_size=10,
        )
        save_md5(md5_hex)
        return "[Success]内容已经成功载入向量库"

if __name__ =='__main__':

    service= KnowledgeBaseService()
    r=service.upload_by_str("流星","testfile")
    print(r)
