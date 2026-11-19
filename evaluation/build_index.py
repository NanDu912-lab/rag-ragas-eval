from pathlib import Path
"""构建菜品向量索引（后台运行，输出进度）。"""
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

t0 = time.time()
from meal_agent.retrieval import DishRetriever, Embedder
from meal_agent.ingest import load_library

dishes = load_library()
print(f"菜品数: {len(dishes)}", flush=True)
r = DishRetriever(dishes=dishes, embedder=Embedder())
print(f"索引完成，用时 {time.time()-t0:.1f}s, 向量数 {len(r.vectors)}", flush=True)

for q in ["低脂高蛋白减脂早餐", "宴客清蒸鱼", "红烧鱼"]:
    print(f"\n查询：{q}", flush=True)
    for it in r.search(q, top_k=5):
        print(f"  {it.score:.5f} v={it.vector_score:+.4f} b={it.bm25_score:7.3f} {it.dish.brief()}", flush=True)
