import os, sys
from llama_index.core import PromptTemplate
from httpx import stream
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage, Settings
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

# 配置千问大模型和文本向量模型
Settings.llm = OpenAILike(
    model="qwen-max",
    api_base=os.getenv("DASHSCOPE_BASE_URL"),
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    is_chat_model=True
)
# 配置文本向量模型，设置批处理大小和最大输入长度
Settings.embed_model = DashScopeEmbedding(
    model_name= 'text-embedding-v4', # DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3,
    embed_batch_size=6,
    embed_input_length=8192
)


# 先禁用 nltk
os.environ["LLAMA_INDEX_DISABLE_NLTK"] = "1"


# 1. 解析文档
def parse_docs(dir = 'docs'):
    # 1. 解析文档
    print('解析文档...')
    documents = SimpleDirectoryReader('docs',
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

# 5. rag 查询并打印
def rag_query(question, dir = '../knowledge/test'):
    # 创建查询引擎
    query_engine = create_rag_engine(dir)
    stream_response = query_engine.query(question)
    stream_response.print_response_stream()

# 6. rag 查询：返回响应对象
def rag_ask(question, dir = '../knowledge/test'):
    query_engine = create_rag_engine(dir)
    return query_engine.query(question)

# 7. 传入查询引擎和问题
def ask(question, engine):
    stream_response = engine.query(question)
    return stream_response

	
def update_prompt_tpl(query_engine, qa_template_str = (
        "你叫公司小蜜，是公司的答疑机器人。你需要仔细阅读参考信息，然后回答大家提出的问题。"
        "注意事项：\n"
        "1. 根据上下文信息而非先验知识来回答问题。\n"
        "2. 如果是工具咨询类问题，请务必给出下载地址链接。\n"
        "3. 如果员工部门查询问题，请务必注意有同名员工的情况，可能有2个、3个甚至更多同名的人\n"
        "以下是参考信息。"
        "---------------------\n"
        "{context_str}\n"  # 占位符：由检索器填充的上下文文本
        "---------------------\n"
        "问题：{query_str}\n。"
        "回答："
)):
    # 创建提示词模板对象
    qa_prompt_template = PromptTemplate(qa_template_str)
    # 更新查询引擎中的提示词模板
    query_engine.update_prompts(
        {"response_synthesizer:text_qa_template": qa_prompt_template}
    )
    return query_engine
