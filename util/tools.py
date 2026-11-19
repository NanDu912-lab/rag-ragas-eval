from openai import OpenAI
from load_env import load_keys
import os

load_keys()

model_company = {
    'deepseek-v4-pro' : 'DEEPSEEK',
    'deepseek-v4-flash' : 'DEEPSEEK',
    'qwen-max' : 'DASHSCOPE',
	'qwen-flash' : 'DASHSCOPE'
}

def create_client(model):
    company = model_company[model]
    if not company:
        raise Exception('无法找到该模型！')
    return OpenAI(
        api_key=os.getenv(company + "_API_KEY"),
        base_url=os.getenv(company + "_BASE_URL"),
    )

# 创建会话客户端
def qwen_client():
    return OpenAI(
        api_key = os.environ['DASHSCOPE_API_KEY'],
        base_url = os.environ['LLM_BASE_URL_QWEN']
    )

def stream_response(client, sys_prompt, user_prompt,
                    temperature = 0.7, top_p = 0.8,
                    model = 'qwen-max'):
    '''流式响应
    :param sys_prompt:  系统提示词
    :param user_prompt: 用户提示词
    :return: 生成器
    '''
    response = client.chat.completions.create(
        model = model,
        messages = [
            {'role': 'system', 'content': sys_prompt},
            {'role': 'user', 'content': user_prompt},
        ],
        stream = True,
        temperature = temperature,
        top_p = top_p,
    )
    for chunk in response:
        if chunk.choices and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content
		
def llm_ask(user_prompt, sys_prompt = '', 
            is_show = True, model = 'qwen-max'):
    client = create_client(model)
    resp = stream_response(client, sys_prompt, 
                           user_prompt, model=model)
    result = ''
    for token in resp:
        if is_show:
            print(token, end='')
        result += token
    return result
		
		
		
		
		
		