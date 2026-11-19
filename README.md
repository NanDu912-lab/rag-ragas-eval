# RAG 检索链路与 RAGAS 自动化评测

围绕「文档答疑」场景做的 RAG 检索质量实验集：切块 → Embedding 选型 → Query 改写 → Rerank →
RAGAS 自动化评测，每一环都是可单独跑的脚本。

## 技术栈
LlamaIndex（`VectorStoreIndex`、`SentenceSplitter` / `SemanticSplitterNodeParser`、
`SimpleDirectoryReader`）、DashScope Embedding + 通义千问、RAGAS
（`context_recall` / `context_precision` / `answer_correctness`）、`datasets`。

## 目录
```
ch02/                      基础链路：向量库、对话历史、Prompt / 结构化输出
ch05/
  Test01~03                朴素 RAG、增量切块、建索引
  Test05_splitter.py       切块策略对比（解决长上下文漏答的起点）
  Test07/08                Embedding 模型横向对比
  Test09~11                Query 改写：直接改写 / step-back / HyDE
  Test12_tags_filter.py    元数据标签过滤
  Test13_rerank.py         检索后重排
  evaluate.py              RAGAS 评测入口：同一问题在不同切块下打分对比
  docs/                    被测语料（部门职责与联系人文档）
evaluation/                RAGAS 版本探测、索引重建、指标可用性排查脚本
util/                      模型调用与 RAG 公共封装
```

## 运行
```bash
pip install -r requirements.txt
cd ch05
python Test05_splitter.py      # 单个实验
python evaluate.py             # RAGAS 评测
```
API Key 走环境变量 `DASHSCOPE_API_KEY`，或首运行时写入 `util/config/keys.json`
（已在 `.gitignore` 中，不会被提交）。

## 关于评测指标
`evaluate.py` 对同一问题在不同切块 / 检索配置下分别计算 `context_recall` 等指标并打印对比，
调参过程的可重复部分是脚本本身；单次跑分产生的输出记录没有随仓库归档，
重跑 `python evaluate.py` 即可复现完整的打分链路。

> 语料用的是脱敏的示例公司文档（含「张伟是哪个部门的」一类问答样例），不含真实企业数据。
