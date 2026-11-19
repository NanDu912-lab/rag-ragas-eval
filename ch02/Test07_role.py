
import sys,os
from llama_index.core import PromptTemplate
sys.path.insert(0, "../util")
import tools
import rag

question = 'Qwen-VL'

llm_engineer = ("你是资深的大模型算法工程师，请根据用户的问题进行解答\n"
"---------------------\n"
"{context_str}\n"
"---------------------\n"
"问题：{query_str}\n。"
"回答：")

pupil_teacher = ("你是一名小学老师，请根据用户的问题进行解答\n"
"---------------------\n"
"{context_str}\n"
"---------------------\n"
"问题：{query_str}\n。"
"回答：")

if __name__ == "__main__":
    query_engine = rag.create_rag_engine()
    query_engine = rag.update_prompt_tpl(query_engine, llm_engineer)
    stream_response = query_engine.query(question)
    stream_response.print_response_stream()
    print('-' * 80)
    query_engine = rag.update_prompt_tpl(query_engine, pupil_teacher)
    stream_response = query_engine.query(question)
    stream_response.print_response_stream()