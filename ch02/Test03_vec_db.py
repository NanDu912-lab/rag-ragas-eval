import os, sys

from httpx import stream
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.indices.base import BaseIndex
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.readers.file import PyMuPDFReader
from openai import embeddings
import pymupdf
import logging
from load_nltk import load_nltk
sys.path.insert(0, "../util")
import tools

logging.basicConfig(level=logging.ERROR)

# 先禁用 nltk
os.environ["LLAMA_INDEX_DISABLE_NLTK"] = "1"


# 1. 解析文档
def parse_docs(dir = 'docs'):
    # 1. 解析文档
    print('解析文档...')
    documents = SimpleDirectoryReader(r'../docs',
                                      file_extractor={'.pdf': PyMuPDFReader()}).load_data()
    # 2. 创建向量索引：指定文档对象 和 嵌入式模型（最新版）
    print('创建向量索引...')
    index = VectorStoreIndex.from_documents(documents,
                                            embed_model=DashScopeEmbedding(
                                                model_name='text-embedding-v4',
                                                embed_batch_size=10))
    return index

# 2. 存储向量库
def store_index(index: BaseIndex, dir = '../knowledge/test'):
    index.storage_context.persist(dir)
    print('存储向量库成功。')

# 3. 加载向量库
def load_index(dir = '../knowledge/test'):
    storage_context = StorageContext.from_defaults(persist_dir=dir)
    index = load_index_from_storage(storage_context,
                            embed_model=DashScopeEmbedding(
                                model_name='text-embedding-v4',
                                embed_batch_size=10
                            ))
    print(f'成功从路径 {dir} 加载索引。')
    return index

# 4. 获取 rag 引擎
def create_rag_engine(dir = '../knowledge/test'):
    index = load_index(dir)
    # 创建查询引擎
    query_engine = index.as_query_engine(
        streaming=True,
        llm=OpenAILike(
            model='qwen-max',
            api_base=os.environ['LLM_BASE_URL_QWEN'],
            api_key=os.environ['DASHSCOPE_API_KEY'],
            is_chat_model=True
        )
    )
    return query_engine

# 5. rag 查询
def rag_query(question, dir = '../knowledge/test'):
    # 创建查询引擎
    query_engine = create_rag_engine(dir)
    stream_response = query_engine.query(question)
    stream_response.print_response_stream()

if __name__ == "__main__":
    # 1. 解析文档并存储向量库
    # index = parse_docs()
    # store_index(index)

    # 2. rag 查询
    rag_query('张伟是那个部门的')
