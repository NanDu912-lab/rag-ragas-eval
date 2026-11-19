'''切片测试'''
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core.node_parser import TokenTextSplitter, SentenceSplitter, SentenceWindowNodeParser, \
    SemanticSplitterNodeParser, MarkdownNodeParser
import pandas as pd
import sys
sys.path.insert(0, "../util")
import rag
from evaluate import evaluate_splitter, question, ground_truth

documents = SimpleDirectoryReader('../docs/2_5').load_data()

if __name__ == '__main__':
    # 1. token 切片
    # splitter_name = 'Token 切片'
    # splitter = TokenTextSplitter(chunk_size= 256,
    #                                    chunk_overlap=32)

    # 2. 句子切片
    # splitter_name = '句子切片'
    # splitter = SentenceSplitter(
    #     chunk_size= 256,
    #     chunk_overlap= 64
    # )

    # 3. 句子窗口切片：采用默认句子大小和 chunk_overlap
    # splitter_name = 'sentence window'
    # splitter = SentenceWindowNodeParser.from_defaults(
    #     window_size=3,
    #     window_metadata_key='window',
    #     original_text_metadata_key='original_text'
    # )

    # 4. 语义切片
    # splitter_name = '语义切片'
    # splitter = SemanticSplitterNodeParser(
    #     buffer_size=1,
    #     breakpoint_percentile_threshold= 60, # 95,
    #     embed_model= Settings.embed_model
    # )

    # 5. markdown 切片
    splitter_name = 'Markdown 切片'
    splitter = MarkdownNodeParser()

    df = evaluate_splitter(splitter, documents,
                           question, ground_truth,
                           splitter_name)
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.width', None,
                           'display.max_colwidth', 50):
        print(df)





