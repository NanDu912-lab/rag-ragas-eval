import os
import json
import dashscope

# 1. 判断配置文件是否存在？
# 1.1 存在，读取
#   1.1.1 没有该 key，输入；保存
#   1.1.2 有该 key；读取
# 1.2 不存在，输入；保存
def load_keys(filename = '../util/config/keys.json'):
    keys = {}
    if os.path.exists(filename): # 文件存在则读取
        with open(filename) as f:
            keys = json.load(f)

    if not 'DASHSCOPE_API_KEY' in keys: # 没有 key，则输入
        key = input('请输入 dashscope api-key:')
        keys['DASHSCOPE_API_KEY'] = key
        with open(filename, 'w') as f: # 将 key 写入文件
            json.dump(keys, f)
    # 把读到的所有 key 写入环境变量
    for k, v in keys.items():
        os.environ[k] = v
    # 对 dashscope key 进行特殊处理
    dashscope.api_key = keys['DASHSCOPE_API_KEY']

if __name__ == "__main__":
    load_keys()
    print(os.environ['DASHSCOPE_API_KEY'])
    print(os.environ['DEEPSEEK_API_KEY'])