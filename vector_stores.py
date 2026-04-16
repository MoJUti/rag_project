from langchain_chroma import Chroma
import config_data as config

class VectorStoreService(object):
    def __init__(self,embedding):
        """
        :param embedding: 嵌入模型的传入
        """
        self.embedding= embedding

        self.vector_store = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory,
        )

    def get_retriever(self):
            """返回向量检索器，方便加入chain"""
            search_kwargs = {"k": config.retrieval_top_k}
            if config.retrieval_source_filter:
                search_kwargs["filter"] = {"source": config.retrieval_source_filter}

            return self.vector_store.as_retriever(search_kwargs=search_kwargs)


