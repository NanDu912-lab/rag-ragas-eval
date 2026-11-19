import pandas as pd
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter


from ch05.evaluate import evaluate_splitter

documents = SimpleDirectoryReader(r'../docs/2-2').load_data()

question = '张伟是哪个部门的'
ground_truth = '''公司有三名张伟，分别是：
- 教研部的张伟：职位是教研专员，邮箱 zhangwei@educompany.com。
- 课程开发部的张伟：职位是课程开发专员，邮箱 zhangwei01@educompany.com。
- IT部的张伟：职位是IT专员，邮箱 zhangwei036@educompany.com。
'''

def generate_query(query: str):
    # 设置生成提示词模型
    response = Settings.llm.predict(
        query_gen_prompt, query=query
    )
    return response # 根据改写模板和问题生成新的提示词

def rewrite_query(query: str):
    print(f'原始问题：{query}')
    new_query = generate_query(query)
    print(f'改写后的问题：{new_query}')
    return new_query

if __name__ == '__main__':
    splitter_name = '句子切片'
    splitter = SentenceSplitter(
        chunk_size = 256,
        chunk_overlap = 64
    )

    question = '找张伟'

    question = rewrite_query(question)

    df = evaluate_splitter(splitter, documents, question, ground_truth, splitter_name)

    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.width', None,
                           'display.max_colwidth', 50):
        print(df)