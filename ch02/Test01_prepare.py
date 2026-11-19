import os, sys
from load_nltk import load_nltk
sys.path.insert(0, '../util')
import tools

if __name__ == '__main__':
    # 生产环境中请勿将 API Key 输出到日志中，避免泄露
    print(f'''你配置的 API Key 是：{os.environ["DASHSCOPE_API_KEY"][:5]+"*"*5}''')

    load_nltk()