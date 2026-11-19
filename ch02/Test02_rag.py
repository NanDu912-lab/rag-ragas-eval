import os, sys

from httpx import stream
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
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

if __name__ == "__main__":
    # 1. 解析文档
    print('解析文档...')
    documents = SimpleDirectoryReader(r'..\docs', file_extractor={'.pdf': PyMuPDFReader()}).load_data()
    # 2. 创建向量索引：指定文档对象 和 嵌入式模型（最新版）
    print('创建向量索引...')
    index = VectorStoreIndex.from_documents(documents,  embed_model= DashScopeEmbedding(
                                                model_name= 'text-embedding-v4', embed_batch_size=10))
    # 3. 创建新查询引擎
    print('创建 rag 查询引擎...')
    query_engine = index.as_query_engine(
        streaming = True,
        llm = OpenAILike(
            model = 'qwen-max',
            api_base = os.environ['LLM_BASE_URL_QWEN'],
            api_key = os.environ['DASHSCOPE_API_KEY'],
            is_chat_model = True    # 多轮对话模式
        )
    )
    # 4. 流式查询
    stream_response = query_engine.query('我们公司内容管理工程师进行项目管理应该用什么工具')
    # 5. 输出结果
    stream_response.print_response_stream()

