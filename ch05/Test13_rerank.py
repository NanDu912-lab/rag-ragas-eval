import sys,os

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, response, Settings
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.postprocessor.dashscope_rerank import DashScopeRerank

sys.path.insert(0, '../util')
import tools, rag

print('1. 加载 md 文件')
documents = SimpleDirectoryReader(r'../docs/2-5').load_data()

print('2. 创建索引向量...')
index = VectorStoreIndex.from_documents()(documents, embed_model = Settings.embed_model)

query_engine = index.as_query_engine(
    streaming = True,
    similarity_matrix = [
        DashScopeRerank(top_n = 3, model = 'qwen3-rerank'),
        SimilarityPostprocessor(similarity_cutoff=0)
    ]
)

question = '张伟是哪个部门的？'
response = rag.ask(question, query_engine)
print('回答')
response.print_response_stream()

print(f"\n📚 召回的参考片段:")
for i, node in enumerate(response.source_nodes, 1):
    print(f"\n文档片段 {i}:")
    print("-" * 40)
    print(node)