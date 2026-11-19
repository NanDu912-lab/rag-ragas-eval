'''初步优化 rag，增加切片数'''
import sys, os
from llama_index.llms.openai_like import OpenAILike

sys.path.insert(0, '../util')
import tools, rag

if __name__ == '__main__':
    index = rag.load_index()
    query_engine = index.as_query_engine(
        streaming=True,
        llm=OpenAILike(
            model='qwen-max',
            api_base=os.environ['LLM_BASE_URL_QWEN'],
            api_key=os.environ['DASHSCOPE_API_KEY'],
            is_chat_model=True
        ),
        similarity_top_k = 20   # 召回 k 个切片
    )
    response = rag.ask('张伟是哪个部门的？', query_engine)
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


