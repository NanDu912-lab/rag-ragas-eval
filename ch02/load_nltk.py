# ============================================================
# 导入所需标准库和第三方库
# ============================================================
import os  # 操作系统接口，用于文件和路径操作、环境变量设置
import zipfile  # 用于解压 ZIP 压缩包
import requests  # 用于发送 HTTP 请求，从网络下载资源文件
from pathlib import Path  # 面向对象的路径操作，更优雅地处理路径
import nltk  # NLTK 自然语言处理库，本段用于管理其离线语料资源

def load_nltk():
    """
    检查并加载 NLTK 离线资源。
    核心逻辑：将必要的语料资源（分词模型、停用词等）下载并解压到
    LlamaIndex 强制要求的 _static/nltk_cache 子目录下。

    由于 LlamaIndex 内部在调用 NLTK 时，会硬编码从特定路径查找资源，
    因此我们必须将资源放置在该路径，否则会触发 LookupError。
    """
    # 1. 定义基础根目录（存放所有 NLTK 数据的顶层文件夹）
    #    此处根据实际部署环境指定，可灵活修改
    base_dir = Path("../data/nltk_data")

    # 2. 【关键】LlamaIndex 真正搜索的物理深层目录
    #    根据 LlamaIndex 的设计，它会自动在缓存根目录下拼接 "_static/nltk_cache"
    #    如果不把资源解压到这里，LlamaIndex 在调用 nltk.data.find() 时就找不到
    actual_cache_dir = base_dir / "_static" / "nltk_cache"

    # 3. 定义需要准备的资源列表：子路径（相对 actual_cache_dir） 和 下载 URL
    #    - punkt: 用于英文分句的预训练模型（标点符号分词器）
    #    - punkt_tab: punkt 的制表符版本，用于扩展分句功能
    #    - stopwords: 常用停用词表（如 "the", "a" 等）
    #    所有资源均从 ModelScope（魔搭社区）的公开数据源下载
    resources = {
        "tokenizers/punkt": "https://www.modelscope.cn/datasets/haoznic/nltk_data_4_llm_learn/resolve/master/punkt.zip",
        "tokenizers/punkt_tab": "https://www.modelscope.cn/datasets/haoznic/nltk_data_4_llm_learn/resolve/master/punkt_tab.zip",
        "corpora/stopwords": "https://www.modelscope.cn/datasets/haoznic/nltk_data_4_llm_learn/resolve/master/stopwords.zip"
    }

    # 4. 设置环境变量，让 LlamaIndex 和 NLTK 能够识别我们的自定义路径
    #    - LLAMA_INDEX_CACHE_DIR: LlamaIndex 会读取此变量作为缓存根目录（它内部会拼接 _static/nltk_cache）
    os.environ["LLAMA_INDEX_CACHE_DIR"] = str(base_dir.resolve())
    #    - NLTK_DATA: NLTK 会读取此变量作为额外的数据搜索路径（直接指向 base_dir）
    os.environ["NLTK_DATA"] = str(base_dir.resolve())

    # 同时，将 base_dir 加入 nltk.data.path 列表（NLTK 的运行时搜索路径）
    # 这样即使环境变量未生效，NLTK 也能找到资源
    if str(base_dir) not in nltk.data.path:
        nltk.data.path.insert(0, str(base_dir.resolve()))

    # 5. 确保实际缓存目录存在（如果不存在则创建）
    actual_cache_dir.mkdir(parents=True, exist_ok=True)
    any_downloaded = False  # 标记是否有任何资源被下载

    # 6. 遍历每个资源，检查是否存在，不存在则下载并解压
    for sub_path, url in resources.items():
        # 检查点：例如 .../nltk_data/_static/nltk_cache/tokenizers/punkt
        # 如果该目录存在，说明该资源已就绪，跳过
        check_point = actual_cache_dir / sub_path

        if check_point.exists():
            continue  # 资源已存在，无需重复下载

        any_downloaded = True
        zip_name = Path(url).name  # 从 URL 中提取 ZIP 文件名，如 "punkt.zip"
        print(f"📦 正在初始化离线资源: {zip_name}...")

        try:
            # 建立该资源对应的父目录（如 tokenizers/ 或 corpora/）
            parent_dir = check_point.parent
            parent_dir.mkdir(parents=True, exist_ok=True)

            # 下载 ZIP 文件到父目录下
            zip_file_path = parent_dir / zip_name
            response = requests.get(url, timeout=60, stream=True)
            response.raise_for_status()  # 如果 HTTP 状态码非 200，则抛出异常

            # 以二进制方式写入文件，分块写入大文件避免内存溢出
            with open(zip_file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 每次 1MB
                    f.write(chunk)

            # 解压 ZIP 到父目录（即 parent_dir）
            # 由于 ZIP 内部结构通常包含与子路径同名的文件夹（如 punkt/），
            # 解压后会自动放置在 parent_dir 下，满足 NLTK 的目录结构要求
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(parent_dir)

            # 删除 ZIP 文件，释放磁盘空间
            zip_file_path.unlink()
            print(f"✅ {zip_name} 处理成功")

        except Exception as e:
            print(f"❌ {zip_name} 初始化失败: {e}")

    # 7. 输出最终状态信息
    if not any_downloaded:
        print(f"💡 环境检查：离线资源已在 {actual_cache_dir} 就绪。")
    else:
        print(f"✅ 资源已成功解压至对齐目录。")
