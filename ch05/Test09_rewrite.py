'''改写问题'''
'''切片测试'''
from llama_index.core import SimpleDirectoryReader, Settings, PromptTemplate
from llama_index.core.node_parser import SentenceSplitter
import pandas as pd
import sys
sys.path.insert(0, "../util")
import rag
from evaluate import evaluate_splitter, ground_truth

documents = SimpleDirectoryReader('../docs/2_5').load_data()

def generate_query(query: str):
    prompt = '''
系统角色设定:
你是一个专业的问题改写助手。你的任务是将用户的原始问题扩充为一个更完整、更全面的问题。

规则：
1. 将可能的歧义、相关概念和上下文信息整合到一个完整的问题中
2. 使用括号对歧义概念进行补充说明
3. 添加关键的限定词和修饰语
4. 确保改写后的问题清晰且语义完整
5. 对于模糊概念，在括号中列举主要可能性

原始问题:
{query}

请生成一个综合的改写问题，确保：
- 包含原始问题的核心意图
- 涵盖可能的歧义解释
- 使用清晰的逻辑关系词连接不同方面
- 必要时使用括号补充说明

输出格式：
[综合改写] - 改写后的问题
    '''
    prompt_template = PromptTemplate(prompt)
    response = Settings.llm.predict(prompt_template,
                                    query = query)
    return response

def rewrite_query(query: str):
    print(f'原始问题：{query}')
    new_query = generate_query(query)
    print(f'改写后问题：{new_query}')
    return new_query

if __name__ == '__main__':
    # 1. 句子切片
    splitter_name = '句子切片'
    splitter = SentenceSplitter(
        chunk_size= 256,
        chunk_overlap= 64
    )

    # 2. 提问
    # 2.1 原始提问
    question = '找张伟'

    # 2.2 提示词改写
    question = rewrite_query(question)

    df = evaluate_splitter(splitter, documents,
                           question, ground_truth,
                           splitter_name)
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.width', None,
                           'display.max_colwidth', 50):
        print(df)











