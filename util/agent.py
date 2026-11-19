from agentscope.agent import ReActAgent, AgentBase
from agentscope.formatter import DashScopeChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.model import DashScopeChatModel
from agentscope.tool import Toolkit
import sys, os

sys.path.insert(0, '../util')
import tools

def create_react_agent(name: str, sys_prompt: str, **kwargs):
    model_name = kwargs.pop("model_name", 'qwen-max')
    stream = kwargs.pop("stream", False)
    multi_agent = kwargs.pop("multi_agent", False)
    model = DashScopeChatModel(
        model_name= model_name,
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        stream=stream,
        multi_agent=multi_agent
    )
    toolkit = kwargs.pop("toolkit", Toolkit())
    memory = kwargs.pop("memory", InMemoryMemory())
    formatter = kwargs.pop("formatter", DashScopeChatFormatter())
    plan_notebook=kwargs.pop("plan_notebook", None)
    max_iters = kwargs.pop("max_iters", 5)

    return ReActAgent(
        name= name,
        sys_prompt=sys_prompt,
        model = model,
        toolkit=toolkit,
        memory= memory,
        formatter= formatter,
        plan_notebook = plan_notebook,
        max_iters= max_iters,
        **kwargs
    )

def set_console_output_enabled(agents: [AgentBase], flag : bool = True):
    for agent in agents:
        agent.set_console_output_enabled(flag)
