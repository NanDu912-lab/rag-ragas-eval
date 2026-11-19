from pathlib import Path
"""legacy metric 类 + evaluate() 顶层接口验证（ragas 0.4.3 + DashScope）。"""
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meal_agent import config
import openai
from datasets import Dataset
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._context_recall import ContextRecall
from ragas.metrics._answer_correctness import AnswerCorrectness

client = openai.AsyncOpenAI(api_key=config.get("DASHSCOPE_API_KEY"), base_url=config.get("DASHSCOPE_BASE_URL"))
llm = llm_factory("qwen-plus", provider="openai", client=client)
emb = embedding_factory("openai", model="text-embedding-v4", client=client)

metrics = [
    Faithfulness(llm=llm),
    AnswerRelevancy(llm=llm, embeddings=emb),
    ContextPrecision(llm=llm),
    ContextRecall(llm=llm),
    AnswerCorrectness(llm=llm, embeddings=emb),
]
print("metrics:", [m.name for m in metrics])

data = {
    "user_input": [
        "请提供一周的健康早餐菜单。",
        "周末请客想吃鱼，应该提供哪些菜单？",
    ],
    "response": [
        "周一早餐：牛奶燕麦（约421大卡）配水煮蛋，蛋白质充足；周二：桂圆红枣粥（约617大卡）；周三：全麦吐司配太阳蛋。",
        "推荐：清蒸鲈鱼（385大卡）为主菜，搭配蒜蓉虾、白灼虾、清蒸生蚝，再配素菜与汤羹。",
    ],
    "retrieved_contexts": [
        [
            "菜名：牛奶燕麦；分类：早餐；预估卡路里：421 大卡/份；食材：牛奶、燕麦、鸡蛋。",
            "菜名：桂圆红枣粥；分类：早餐；预估卡路里：617 大卡/份；食材：糯米、红枣、桂圆。",
        ],
        [
            "菜名：清蒸鲈鱼；分类：水产；预估卡路里：385 大卡/份；难度：★★★；食材：鲈鱼、香葱、姜。",
            "菜名：蒜蓉虾；分类：水产；预估卡路里：456 大卡/份；食材：海虾、蒜蓉酱。",
        ],
    ],
    "reference": [
        "一周健康早餐菜单应覆盖 7 天，每天以优质蛋白+主食+蔬果为主，单餐约 300-500 大卡，例如牛奶燕麦、水煮蛋、全麦吐司、桂圆红枣粥、蒸蛋等。",
        "周末宴客鱼类菜单应包含鱼类主菜（清蒸鲈鱼/红烧鱼/糖醋鲤鱼等）、海鲜、素菜、汤羹与主食，荤素搭配。",
    ],
}

ds = Dataset.from_dict(data)
res = evaluate(ds, metrics=metrics, llm=llm, embeddings=emb, show_progress=False, raise_exceptions=False)
print(res)
print(res.to_pandas().to_string())
