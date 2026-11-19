from llama_index.embeddings.dashscope import DashScopeTextEmbeddingModels, DashScopeEmbedding
import sys
sys.path.insert(0, '../util')
import tools
import numpy as np
from Test06_similarity import cosine_similarity

def compare_embeddings(query, chunks, embeddings):
    print(f'问题：{query}')
    print('-' * 80)
    print(f'召回片段：{chunks}')
    for i, chunk in enumerate(chunks, 1):
        print(f'{i}. {chunk}')
    print('-' * 80)

    for model_name, model in embeddings.items():
        print(f'模型：{model_name}')
        query_embedding = (model.get_query_embedding(query)
                           if hasattr(model, 'get_query_embedding')
                            else model.get_text_embedding(query))
        for i, chunk in enumerate(chunks, 1):
            chunk_embedding = (model.get_chunk_embedding(chunk)
                               if hasattr(model, 'get_chunk_embedding')
                               else model.get_text_embedding(chunk))
            print(f'{i}. {cosine_similarity(np.array(query_embedding), np.array(chunk_embedding))}')
        print('-' * 80)

if __name__ == '__main__':
    # 准备测试数据
    query = "张伟是哪个部门的"
    chunks = [
        "核，提供⾏政管理与协调⽀持，优化⾏政⼯作流程。 ⾏政部 秦⻜ 蔡静 G705 034 ⾏政 ⾏政专员 13800000034 qinf@educompany.com 维护公司档案与信息系统，负责公司通知及公告的发布",
        "组织公司活动的前期准备与后期评估，确保公司各项⼯作的顺利进⾏。 IT部 张伟 ⻢云 H802 036 IT⽀撑 IT专员 13800000036 zhangwei036@educompany.com 进⾏公司⽹络及硬件设备的配置"
    ]
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

    compare_embeddings(query, chunks, embeddings)





