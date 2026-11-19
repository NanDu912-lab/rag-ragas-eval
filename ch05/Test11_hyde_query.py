'''假设文档查询'''
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.indices.query.query_transform import StepDecomposeQueryTransform, HyDEQueryTransform
from llama_index.core.node_parser import SentenceSplitter
import sys
from evaluate import evaluate_result
from llama_index.core.query_engine import MultiStepQueryEngine, TransformQueryEngine
import pandas as pd
sys.path.insert(0, '../util')
import rag

def analysis_hyde(question):
    '''分析假设文档'''
    hyde_transform = HyDEQueryTransform(include_original=True)
    result = hyde_transform(question)
    doc = result.embedding_strs[0]
    print(f'问题：{question}')
    print(f'假设文档：{doc}')

if __name__ == '__main__':
    # 分析假设文档
    question = '找张伟'
    analysis_hyde(question)


    # # 0. 准备工作
    # # 0.1 加载文档
    # documents = SimpleDirectoryReader('../docs/2_5').load_data()
    # # 0.2 句子切片器
    # splitter = SentenceSplitter(
    #     chunk_size = 256,
    #     chunk_overlap = 32
    # )
    # # 0.3 切片
    # print(f'{"=" * 80}')
    # nodes = splitter.get_nodes_from_documents(documents)
    # # 0.4 创建索引
    # print(f'1. 创建索引...')
    # index = VectorStoreIndex(nodes,
    #                          embed_model=Settings.embed_model)
    #
    # # 1. 创建假设文档查询转换器
    # hyde_transform = HyDEQueryTransform(include_original=True)
    # # 2. 原始查询引擎
    # print(f'2. 创建查询引擎...原始引擎...查询转换引擎')
    # query_engine = index.as_query_engine(
    #     streaming=True,
    #     similarity_top_k=10
    # )
    # # 3. 创建转换查询引擎
    # query_engine = TransformQueryEngine(query_engine, query_transform=hyde_transform)
    #
    # # 4. 查询
    # print(f'查询：{question}')
    # response = query_engine.query(question)
    # print(f'{"=" * 80}')
    # print(f'回答：')
    # response.print_response_stream()
    #
    # # 5. 输出参考片段
    # print(f"\n📚召回的参考片段:")
    # for i, node in enumerate(response.source_nodes, 1):
    #     print(f"\n文档片段 {i}:")
    #     print("-" * 40)
    #     print(node)
    #
    # # 6. 显示评估结果
    # print(f"\n📊 评估结果:")
    # print("-" * 40)
    # df = evaluate_result(question, response)
    # with pd.option_context('display.max_rows', None,
    #                        'display.max_columns', None,
    #                        'display.width', None,
    #                        'display.max_colwidth', 50):
    #     print(df)








































