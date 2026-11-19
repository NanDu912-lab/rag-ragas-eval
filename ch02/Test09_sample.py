import sys,os
sys.path.insert(0, "../util")
import tools

question_task = """
【任务要求】
请你根据用户的主题，创作内容。
---
【输出要求】
最终输出需要以Markdown格式呈现，请注意，在你的回答中包含所有必要的Markdown元素，如标题、列表、链接、图片引用、加粗等，以便于阅读、后续编辑和保存。
---
【用户输入】
以下是用户的要求创作的主题：
"""
question_doc = "手工钥匙扣制作教程"

question_task_sample= """
【任务要求】
请根据用户的主题，结合下面【样例】给的例子，理解和使用一致的风格和结构继续创作内容，不要输出多余的内容。
---
【输出要求】
最终输出需要以Markdown格式呈现，请注意，在你的回答中包含所有必要的Markdown元素，如标题、列表、链接、图片引用、加粗等，以便于阅读、后续编辑和保存。
---
【样例】
### 示例1: 制作简易书签
# 简易书签制作教程

## 材料清单
- 彩色卡纸
- 剪刀
- 装饰贴纸
- 铅笔

## 步骤
1. 选择一张彩色卡纸。
2. 用铅笔在卡纸上画出一个长方形，尺寸约为2英寸 x 6英寸。
3. 沿着铅笔线剪下长方形。
4. 使用装饰贴纸对书签进行个性化装饰。
5. 完成！现在你有了一个独一无二的书签。

## 结束语
希望这个教程能帮助你制作出满意的书签！

---
【用户输入】
以下是用户的要求创作的主题：
"""
question_doc = "手工钥匙扣制作教程"

if __name__ == '__main__':
    # question = question_task + question_doc
    question = question_task_sample + question_doc

    client = tools.qwen_client()
    stream_response = tools.stream_response(client,
                                              sys_prompt='',
                                              user_prompt=question)
    for token in stream_response:
        print(token, end='')