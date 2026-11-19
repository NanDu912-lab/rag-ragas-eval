import sys
sys.path.insert(0, '../util')
import tools, rag

if __name__ == '__main__':
    response = rag.rag_ask('张伟是哪个部门的？')
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