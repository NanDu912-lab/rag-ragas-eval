'''分步查询'''
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.indices.query.query_transform import StepDecomposeQueryTransform
from llama_index.core.node_parser import SentenceSplitter
import sys
from evaluate import evaluate_result
from llama_index.core.query_engine import MultiStepQueryEngine
import pandas as pd
sys.path.insert(0, '../util')
import rag


if __name__ == '__main__':
    # 0. 准备工作
    # 0.1 加载文档
    documents = SimpleDirectoryReader('../docs/2-5').load_data()
    # 0.2 句子切片器
    splitter = SentenceSplitter(
        chunk_size = 256,
        chunk_overlap = 32
    )
    # 0.3 切片
    print(f'{"=" * 80}')
    nodes = splitter.get_nodes_from_documents(documents)
    # 0.4 创建索引
    print(f'1. 创建索引...')
    index = VectorStoreIndex(nodes,
                             embed_model=Settings.embed_model)

    # 1. 创建多步解构查询转换器
    step_decompose_transform = StepDecomposeQueryTransform(
        verbose=True)

    # 2. 原始查询引擎
    print(f'2. 创建查询引擎...原始引擎...多步查询引擎')
    query_engine = index.as_query_engine(
        streaming=True,
        similarity_top_k=10
    )
    # 3. 创建多步查询引擎
    multi_step_query = MultiStepQueryEngine(
        query_engine = query_engine,
        query_transform = step_decompose_transform,
        index_summary= '公司人员信息'
    )
    # 4. 查询
    question = '找张伟'
    print(f'查询：{question}')
    response = multi_step_query.query(question)
    print(f'{"=" * 80}')

    # 5. 输出参考片段
    print(f"\n📚召回的参考片段:")
    for i, node in enumerate(response.source_nodes, 1):
        print(f"\n文档片段 {i}:")
        print("-" * 40)
        print(node)

    # 6. 显示评估结果
    print(f"\n📊 评估结果:")
    print("-" * 40)
    df = evaluate_result(question, response)
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.width', None,
                           'display.max_colwidth', 50):
        print(df)








































