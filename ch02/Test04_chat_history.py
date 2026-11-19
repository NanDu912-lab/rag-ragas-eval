'''多轮对话'''

import sys,os
from llama_index.core import PromptTemplate
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.indices.base import BaseIndex
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.readers.file import PyMuPDFReader
from openai import embeddings
sys.path.insert(0, "../util")
import tools
import rag

if __name__ == '__main__':
    # 1. 创建 rag 引擎
    rag_engine = rag.create_rag_engine()

    # 2. 创建提示词模板
    prompt_template = PromptTemplate('''
        给定一段对话历史（人类与助手之间）和人类的后续问题，
        请将该问题改写为一个独立的问题，包含对话中所有相关的上下文信息。
    
        <对话历史>
        {chat_history}
    
        <后续问题>
        {question}
    
        <改写后的独立问题>
    ''')

    # 3. 会话历史
    chat_history = [
        ChatMessage(role=MessageRole.USER, content = '内容开发工程师有哪些细分类型？'),
        ChatMessage(role=MessageRole.ASSISTANT, content = '内容开发工程师是综合性技术岗位。')
    ]

    # 创建多轮会话引擎 Condense...
    # 1. 创建 rag 引擎
    # 2. 创建提示词模板
    # 3. 会话历史
    # 4. llm：用于浓缩会话
    chat_engine = CondenseQuestionChatEngine.from_defaults(
        query_engine=rag_engine,
        condense_question_prompt=prompt_template,
        chat_history = chat_history,
        llm=OpenAILike(
            model='qwen-max',
            api_base=os.environ['LLM_BASE_URL_QWEN'],
            api_key=os.environ['DASHSCOPE_API_KEY'],
            is_chat_model=True
        ),
        verbose= True
    )

    # 查询问题
    stream_response = chat_engine.stream_chat("核心职责是什么？")
    for token in stream_response.response_gen:
        print(token, end='')














