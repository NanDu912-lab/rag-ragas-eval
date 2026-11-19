import inspect
import warnings

warnings.filterwarnings("ignore")
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory

print("llm_factory:", inspect.signature(llm_factory))
print("embedding_factory:", inspect.signature(embedding_factory))

import os

os.environ["DASHSCOPE_API_KEY"] = ""
from meal_agent import config

print(config.describe())
