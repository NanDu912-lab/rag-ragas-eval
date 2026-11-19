import sys,os
sys.path.insert(0, '../util')
import tools

sys_prompt = """你是一个标签提取专家。请从文本中提取结构化信息，并按要求输出标签。
---
【支持的标签类型】
- 人名
- 部门名称
- 职位名称
- 技术领域
- 产品名称
---
【输出要求】
1. 请用 JSON 格式输出，如：[{"key": "部门名称", "value": "教研部"}]
2. 如果某类标签未识别到，则不输出该类
---
待分析文本如下：
"""

def extract_tags(text: str):
    tools.llm_ask(sys_prompt + text)

if __name__ == '__main__':
    print('-' * 80)
    hr_text = """张明是我们AI研发部的技术主管，他带领团队开发了新一代智能对话平台 ChatMax，在自然语言处理领域有着丰富经验。如果您需要了解项目细节，可以直接联系他。"""
    print("人事文档标签提取结果：")
    print(f'原文：{hr_text}')
    extract_tags(hr_text)
    print('\n\n','-' * 80)

    # 示例2：技术文档
    tech_text = """本论文提出了一种基于深度学习的图像识别算法，在医疗影像分析中取得了突破性进展。该算法已在北京协和医院的CT诊断系统中得到应用。"""
    print("技术文档标签提取结果：")
    print(f'原文：{tech_text}')
    extract_tags(tech_text)