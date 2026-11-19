import sys,os
sys.path.insert(0, "../util")
import tools

# 无推理
question = """
【背景信息】
某教育培训机构（以下简称“公司”）在2023年度发生了以下主要支出：
为了给不同城市的学校学生上课，公司的老师全年共出差了5次，每次出差时间为一周，具体费用如下：
    - 交通费及住宿费：平均1600元/次
    - 教学用具采购费用：公司在年初一次性购买了一批教学用具，总价为10000元，预计可以使用4年。

【问题描述】
请根据上述背景信息，完成以下任务：
计算全年因教师出差而产生的差旅总费用，包括摊销的教学用具。

【输出要求】
"""

# 直接回答
# question += '直接给出总差旅费用，不要其他信息'
# 使用 COT
question += ' 请你一步步推导，计算总差旅费用'

if __name__ == '__main__':
    client = tools.qwen_client()
    stream_response = tools.stream_response(client,
                                              sys_prompt='',
                                              user_prompt=question)
    for token in stream_response:
        print(token, end='')