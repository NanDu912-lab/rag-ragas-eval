from pathlib import Path
"""最小化验证 ragas 0.4.3 + DashScope(qwen) 是否可跑通。"""
import asyncio
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meal_agent import config

print(config.describe())
print()

import openai
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    AnswerCorrectness,
)

client = openai.AsyncOpenAI(
    api_key=config.get("DASHSCOPE_API_KEY"),
    base_url=config.get("DASHSCOPE_BASE_URL"),
)

llm = llm_factory("qwen-plus", provider="openai", client=client)
print("llm ok:", type(llm).__name__)
emb = embedding_factory("openai", model="text-embedding-v4", client=client)
print("emb ok:", type(emb).__name__)

sample = dict(
    user_input="请提供一周的健康早餐菜单。",
    response="周一：牛奶燕麦（421大卡）+ 水煮蛋；周二：桂圆红枣粥（617大卡）。",
    retrieved_contexts=[
        "菜名：牛奶燕麦；分类：早餐；预估卡路里：421 大卡/份；食材：牛奶、燕麦、鸡蛋；操作要点：牛奶倒入早餐杯，燕麦煮 2 分钟后倒入牛奶。",
        "菜名：桂圆红枣粥；分类：早餐；预估卡路里：617 大卡/份；食材：糯米、红枣、桂圆。",
    ],
    reference="一周健康早餐应包含优质蛋白与膳食纤维，例如牛奶燕麦、水煮蛋、桂圆红枣粥、全麦吐司等，单餐 300-500 大卡。",
)


async def main():
    for metric in [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=emb),
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm),
        AnswerCorrectness(llm=llm, embeddings=emb),
    ]:
        try:
            score = await metric.ascore(**{k: v for k, v in sample.items() if k != "reference"} | {"reference": sample["reference"]})
            print(f"OK   {metric.name}: {score}")
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {metric.name}: {type(e).__name__}: {str(e)[:300]}")


asyncio.run(main())
