'''自定义解析器'''
import json
import os, sys

import nest_asyncio
from llama_index.readers.dashscope import DashScopeParse, ResultType
import logging

sys.path.insert(0, '../util')
import tools
os.environ['DASHSCOPE_BASE_URL'] = "https://dashscope.aliyuncs.com"

silent_logging = logging.getLogger(__name__)
silent_logging.setLevel(logging.ERROR)

class SilentDashScopeParser(DashScopeParse):
    '''自定义文档解析器'''
    def __init__(self, *args, **kwargs):
        import llama_index.readers.dashscope.base as base_module
        import llama_index.readers.dashscope.domain.lease_domains as lease_domains_module
        import llama_index.readers.dashscope.utils as utils_module
        base_module.logger = silent_logging
        lease_domains_module.logger = silent_logging
        utils_module.logger = silent_logging
        super().__init__(*args, **kwargs)

def file_2_md(file, category_id):
    parser = SilentDashScopeParser(
        result_type=ResultType.DASHSCOPE_DOCMIND,
        category_id=category_id
    )
    documents = parser.load_data(file_path = file)
    md_content = ''
    for doc in documents:
        doc_json = json.loads(json.loads(doc.text))
        for item in doc_json['layouts']:
            if item['text'] in item['markdownContent']:
                md_content += item['markdownContent']
            else:
                md_content += item['text'] + '\n'
    return md_content

def md_polisher(data):
    question = '''下面这段文本是由pdf转为markdown的，格式和内容可能存在一些问题，需要你帮我优化下：
        1、目录层级，如果目录层级顺序不对请以markdown形式补全或修改；
        2、内容错误，如果存在上下文不一致的情况，请你修改下；
        3、如果有表格，注意上下行不一致的情况；
        4、输出文本整体应该与输入没有较大差异，不要自己制造内容，我是需要对原文进行润色；
        5、输出格式要求：markdown文本，你的所有回答都应该放在一个markdown文件里面。
        特别注意：只输出转换后的 markdown 内容本身，不输出任何其他信息。
        需要处理的内容是：''' + data

    os.environ['DASHSCOPE_BASE_URL'] = os.environ["LLM_BASE_URL_QWEN"]
    result = tools.llm_ask(question)    # qwen-max

    # result = tools.llm_ask(question, model='deepseek-v4-flash')
    return result

if __name__ == '__main__':
    # 1. pdf 转换为 md
    md = file_2_md('../docs/2_2/内容公司各部门职责与关键角色联系信息汇总.pdf', 'default')
    # print(md)

    # 2. 润色 md
    md = md_polisher(md)

    # 3. 保存到文件
    with open('docs/内容公司各部门职责与关键角色联系信息汇总.md', 'w', encoding='utf-8') as f:
        f.write(md)

















