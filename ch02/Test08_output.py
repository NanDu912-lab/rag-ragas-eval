
import sys,os
sys.path.insert(0, "../util")
import tools

question_task = """
【任务要求】
你将看到一句话或一段话。你需要审查这段话中有没有错别字。如果出现了错别字，你要指出错误，并给出解释。
    “的” 和 “地” 混淆不算错别字，没有错误
---
【输出要求】
请你只输出json格式，不要输出代码段
其中，label只能取0或1，0代表有错误，1代表没有错误
reason是错误的原因
correct是修正后的文档内容
---
【用户输入】
以下是用户输入，请审阅：
"""
question_doc = "分隔符是特殊的符号，它们帮助大语言模形 (LLM) 识别提示中哪些部分应当被视为一个完整的意思单元。"

question = question_task + question_doc

if __name__ == '__main__':
    client = tools.qwen_client()
    stream_response = tools.stream_response(client,
                                              sys_prompt='',
                                              user_prompt=question)
    for token in stream_response:
        print(token, end='')