import hashlib
import os

from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings

from core import config
from ingestion.legal_chunker import build_chunks_from_article_units, parse_legal_article_units
from ingestion.legal_preprocess import preprocess_legal_text


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


def check_md5(md5_str: str):
	if not os.path.exists(config.md5_path):
		open(config.md5_path, "w", encoding="utf-8").close()
		return False

	for line in open(config.md5_path, "r", encoding="utf-8").readlines():
		line = line.strip()
		if line == md5_str:
			return True
	return False


def save_md5(md5_str: str):
	with open(config.md5_path, "a", encoding="utf-8") as f:
		f.write(md5_str + "\n")


def get_string_md5(input_str: str, encoding="utf-8"):
	str_bytes = input_str.encode(encoding=encoding)
	md5_obj = hashlib.md5()
	md5_obj.update(str_bytes)
	return md5_obj.hexdigest()


class KnowledgeBaseService:
	"""知识库入库服务。"""

	def __init__(self):
		os.makedirs(config.persist_directory, exist_ok=True)

		self.chroma = Chroma(
			collection_name=config.collection_name,
			embedding_function=DashScopeEmbeddings(model=config.embedding_model_name),
			persist_directory=config.persist_directory,
		)
		self.chunk_size = config.chunk_size
		self.chunk_overlap_articles = config.chunk_overlap_articles

	def upload_by_str(self, data: str, filename: str):
		data = preprocess_legal_text(data)

		md5_hex = get_string_md5(data)
		if check_md5(md5_hex):
			return "[Repeat] 内容已存在知识库"

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

		add_texts_in_batches(
			self.chroma,
			knowledge_chunks,
			metadata_list,
			batch_size=10,
		)
		save_md5(md5_hex)
		return "[Success]内容已经成功载入向量库"


__all__ = [
	"KnowledgeBaseService",
	"add_texts_in_batches",
	"check_md5",
	"save_md5",
	"get_string_md5",
]
