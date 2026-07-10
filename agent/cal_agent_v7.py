# 基于 v6 的进阶版本，引入 Multi-Agent 架构：
# 1. Orchestrator：阅读用户的复合请求，拆成若干独立子任务，标注每个子任务该交给哪个 Sub-Agent
# 2. Sub-Agent：各自独立的小型 agent，只负责自己领域的问题（价格 / 天气），复用 v2 的
#    function calling 模式
# 3. 并行执行：所有 Sub-Agent 用 asyncio.gather 并行跑（复用 v2 的并行思路，只是
#    这次并行的对象从"多个工具调用"变成了"多个 Sub-Agent"）
# 4. Aggregator：把所有 Sub-Agent 的回答汇总成一句连贯的最终回复
#
# 跟 v5/v6 的核心区别：v5/v6 是"一个 Agent 自己按步骤调用工具"，v7 是"一个 Orchestrator
# 把任务分发给多个专职 Agent"——每个 Sub-Agent 更专注、更容易维护，且天然支持并行。


import asyncio
import json
import openai
from dotenv import load_dotenv

load_dotenv()
client = openai.OpenAI()

# ===========================
# 工具定义（分别属于不同 Sub-Agent，互不共享）
# ===========================

def get_price(item_name: str):
    """查询商品单价"""
    mock_db = {"react_book": 128, "vue_book": 100, "coffee": 35}
    return mock_db.get(item_name.strip(), "未知商品")

def calculate(expression: str):
    """执行数学运算"""
    try:
        return eval(expression)
    except Exception as e:
        return f"计算错误: {e}"

def get_weather(city: str):
    """查询城市天气（mock 数据）"""
    mock_db = {"北京": "晴，25°C", "上海": "多云，28°C", "深圳": "小雨，30°C"}
    return mock_db.get(city.strip(), "暂无该城市的天气数据")

PRICE_TOOLS = {"get_price": get_price, "calculate": calculate}
PRICE_TOOLS_CONFIG = [
    {
        "type": "function",
        "function": {
            "name": "get_price",
            "description": "根据商品名称查询单价",
            "parameters": {
                "type": "object",
                "properties": {"item_name": {"type": "string", "description": "商品名称，例如 react_book"}},
                "required": ["item_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "执行数学计算",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "数学表达式，例如 1 + 1"}},
                "required": ["expression"],
            },
        },
    },
]

WEATHER_TOOLS = {"get_weather": get_weather}
WEATHER_TOOLS_CONFIG = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的天气",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "城市名称，例如 北京"}},
                "required": ["city"],
            },
        },
    },
]

# ===========================
# Sub-Agent：只负责自己领域的问题，内部是标准的 OpenAI function calling 两段式调用
# （跟 v2 的单个 agent 结构一样，只是现在会有多个这样的实例并存）
# ===========================

class SubAgent:
    def __init__(self, name: str, system_prompt: str, tools: dict, tools_config: list):
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools
        self.tools_config = tools_config

    async def run(self, query: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]
        response = client.chat.completions.create(
            model="gpt-4o", messages=messages, tools=self.tools_config
        )
        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append(msg)
            for call in msg.tool_calls:
                fn = self.tools[call.function.name]
                args = json.loads(call.function.arguments)
                result = fn(**args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": str(result),
                })
            response = client.chat.completions.create(model="gpt-4o", messages=messages)
            return response.choices[0].message.content

        return msg.content


price_agent = SubAgent(
    name="price_agent",
    system_prompt="你是价格计算助手，负责查询商品单价、计算总价，用一句话简洁回答。",
    tools=PRICE_TOOLS,
    tools_config=PRICE_TOOLS_CONFIG,
)

weather_agent = SubAgent(
    name="weather_agent",
    system_prompt="你是天气助手，负责回答天气相关问题，用一句话简洁回答。",
    tools=WEATHER_TOOLS,
    tools_config=WEATHER_TOOLS_CONFIG,
)

AGENTS = {"price_agent": price_agent, "weather_agent": weather_agent}

# ===========================
# Orchestrator：阅读用户请求，拆解成子任务并分派给对应 Sub-Agent
# ===========================

ORCHESTRATOR_PROMPT = """
你是一个任务调度器，负责阅读用户的请求，把它拆分成若干独立子任务，
并指明每个子任务应该交给哪个 Sub-Agent 处理。如果用户的请求只涉及一个领域，
就只拆出一个子任务。

可用的 Sub-Agent：
- price_agent：负责商品价格查询、算术计算
- weather_agent：负责天气查询

返回 JSON 格式：
{"tasks": [{"agent": "<agent名>", "query": "<交给该 agent 的子任务文本>"}]}
"""

def orchestrate(user_input: str) -> list:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": ORCHESTRATOR_PROMPT}, {"role": "user", "content": user_input}],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)["tasks"]

# ===========================
# Aggregator：把所有 Sub-Agent 的回答汇总成一句连贯的话
# ===========================

def aggregate(user_input: str, results: list) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "你是总结助手，把多个子任务的回答汇总成一句连贯的话回复用户。"},
            {"role": "user", "content": f"用户问题：{user_input}\n各子任务结果：{results}"},
        ],
    )
    return response.choices[0].message.content

# ===========================
# 主流程：拆解 → 并行执行 Sub-Agent → 汇总
# ===========================

async def run(user_input: str) -> str:
    tasks = orchestrate(user_input)
    print(f"Orchestrator 拆分结果: {tasks}")

    # 并行执行所有子任务（跟 v2 的 asyncio.gather 是同一个套路，
    # 只是这次并行跑的是多个 Sub-Agent，而不是多个工具调用）
    coros = [AGENTS[t["agent"]].run(t["query"]) for t in tasks]
    results = await asyncio.gather(*coros)
    print(f"各 Sub-Agent 结果: {results}")

    final_answer = aggregate(user_input, results)
    print(f"Question: {user_input}")
    print(f"Answer: {final_answer}")
    return final_answer


if __name__ == "__main__":
    asyncio.run(run("帮我查一下 react_book 多少钱，另外北京今天天气怎么样？"))
