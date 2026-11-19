import inspect
import warnings

warnings.filterwarnings("ignore")
import ragas
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    AnswerCorrectness,
)

print("ragas", ragas.__version__)
print("evaluate:", inspect.signature(ragas.evaluate))
for C in (Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall, AnswerCorrectness):
    print("==", C.__name__)
    try:
        print("   __init__:", inspect.signature(C.__init__ or C))
    except Exception as e:
        print("   sig err", e)
    src = inspect.getsource(C)
    print("   bases:", [b.__name__ for b in C.__mro__[1:4]])
    for line in src.splitlines()[:22]:
        print("     ", line)
