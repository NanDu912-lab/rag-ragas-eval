'''重建索引'''
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.llms.openai_like import OpenAILike
import os, sys
sys.path.insert(0, '../util')
import tools, rag

if __name__ == '__main__':
    print('1. 加载文档')
    documents = SimpleDirectoryReader('../docs/2_5').load_data()

    print('2. 创建索引')
    index = VectorStoreIndex.from_documents(documents)

    print('3. 创建引擎')
    query_engine = index.as_query_engine(
        streaming=True,
        similarity_top_k = 20
    )

    print('4. 检索生成')
    response = query_engine.query('张伟是哪个部门的？')
    response.print_response_stream()
    print()
    # 输出参考文档
    print('' + '-' * 50)  # 使用乘法生成分割线
    print('📚 参考文档：')
    for i, source_node in enumerate(response.source_nodes, start=1):
        print(f'文档 {i}:')
        print(source_node)
        print()
    print('-' * 50)  # 使用乘法生成分割线





