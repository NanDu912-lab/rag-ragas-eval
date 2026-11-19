'''更新提示词模板'''
import sys,os
from llama_index.core import PromptTemplate
sys.path.insert(0, "../util")
import tools
import rag

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

if __name__ == "__main__":
    query_engine = rag.create_rag_engine()
    query_engine = update_prompt_tpl(query_engine)

    stream_response = query_engine.query('我们公司内容管理工程师进行项目管理应该用什么工具')
    stream_response.print_response_stream()













