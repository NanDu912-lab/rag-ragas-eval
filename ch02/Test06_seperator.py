import sys,os
from llama_index.core import PromptTemplate
sys.path.insert(0, "../util")
import tools
import rag

question = """
把下列【】括起来的文本进行扩写和润色，让文案生动且富有创造力，并且受到公司新员工的喜欢。
【新员工训练营活动】
"""

rag.rag_query(question)

