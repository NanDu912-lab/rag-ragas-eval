'''评估切片方式'''
from datasets import Dataset
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.llms.tongyi import Tongyi
from llama_index.core import VectorStoreIndex, Settings, SimpleDirectoryReader
import os, sys

from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.embeddings.dashscope import DashScopeEmbedding
from ragas import evaluate
from ragas.metrics import context_precision, answer_correctness, context_recall

sys.path.insert(0, '../util')
import  tools, rag
import pandas as pd

# 准备测试数据
question = '张伟是哪个部门的'
ground_truth = '''公司有三名张伟，分别是：
- 教研部的张伟：职位是教研专员，邮箱 zhangwei@educompany.com。
- 课程开发部的张伟：职位是课程开发专员，邮箱 zhangwei01@educompany.com。
- IT部的张伟：职位是IT专员，邮箱 zhangwei036@educompany.com。
'''

# 加载文档
documents = SimpleDirectoryReader('./docs').load_data()

# 定义评估函数
def evaluate_result(question, response, ground_truth,
                    embedding_name):
    # 获取回答内容
    if hasattr(response, 'response_txt'):
        answer = response.response_txt
    else:
        answer = str(response)
    # 获取检索到的上下文
    context = [source_node.get_content()
               for source_node in response.source_nodes]

    # 构造评估数据集
    data_samples = {
        'question': [question],
        'answer': [answer],
        'ground_truth':[ground_truth],
        'contexts' : [context],
    }
    dataset = Dataset.from_dict(data_samples)

    # 使用Ragas进行评估
    score = evaluate(
        dataset = dataset,
        metrics=[answer_correctness,
                 context_recall,
                 context_precision],
        llm=Tongyi(model_name="qwen-max"),
        embeddings=DashScopeEmbeddings(model=embedding_name)
    )
    return score.to_pandas()

def evaluate_embeddings(splitter, documents, question, ground_truth,
                        embedding_name, embedding_model):
    print(f'{"=" * 80}')
    nodes = splitter.get_nodes_from_documents(documents)

    print(f'1. 使用指定嵌入式模型创建索引...')
    index = VectorStoreIndex(nodes, embed_model = embedding_model)
    print(f'2. 创建查询引擎...')
    query_engine = index.as_query_engine(
        streaming = True,
        similarity_top_k = 10
    )
    print('3. 查询结果')
    response = query_engine.query(question)
    response.print_response_stream()

    # 输出参考片段
    # print(f"\n📚 {splitter_name} 召回的参考片段:")
    # for i, node in enumerate(response.source_nodes, 1):
    #     print(f"\n文档片段 {i}:")
    #     print("-" * 40)
    #     print(node)

    # 评估结果
    print(f"\n📊 评估结果:")
    print("-" * 40)
    return evaluate_result(question,
        response, ground_truth,
                           embedding_name)

if __name__ == '__main__':
    splitter = MarkdownNodeParser()
    embeddings = {
        'text-embedding-v2' : DashScopeEmbedding(
            model_name= 'text-embedding-v2',
            embed_batch_size=6,
            embed_input_length=8192
        ),
        'text-embedding-v3': DashScopeEmbedding(
            model_name= 'text-embedding-v3',
            embed_batch_size=6,
            embed_input_length=8192
        ),
        'text-embedding-v4': DashScopeEmbedding(
            model_name= 'text-embedding-v4',
            embed_batch_size=6,
            embed_input_length=8192
        )
    }

    for model_name, embedding in embeddings.items():
        print(f'{"=" * 80}')
        print(f'嵌入式模型：{model_name}')
        df = evaluate_embeddings(splitter, documents,
        question, ground_truth, model_name, embedding)
        with pd.option_context('display.max_rows', None,
                               'display.max_columns', None,
                               'display.width', None,
                               'display.max_colwidth', 50):
            print(df)


