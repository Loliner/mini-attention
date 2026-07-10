# 练习：给 SubAgent 加一个多轮 function calling 循环，再接入一个需要多步工具调用的 order_agent
#
# v7 例子里 SubAgent.run() 只处理"一轮"工具调用：模型请求调用工具 → 执行 → 再问一次
# 拿最终答案，就结束了。price_agent/weather_agent 一步到位没问题，但如果某个 Sub-Agent
# 需要连续好几步工具调用才能完成任务（比如：先查库存，库存不够就多下单顶替，再结账），
# 现在的 run() 处理不了。
#
# 任务：
#   1. 把 SubAgent.run() 改造成一个循环：只要模型还在请求 tool_calls，就继续执行工具、
#      把结果塞回 messages，再问模型一次，直到模型不再请求工具调用为止（加一个最大轮数
#      保护，比如 5 轮，避免死循环）
#   2. 用下面已经写好的 check_stock / add_to_order / checkout 三个工具，组一个新的
#      order_agent
#   3. 把 order_agent 注册进 AGENTS，更新 ORCHESTRATOR_PROMPT 加入它的说明
#
# 测试：run("帮我买 2 本 react_book、1 本 vue_book、3 杯 coffee，另外北京天气怎么样")
# 观察点：这次 order_agent 完全没有走"先 plan 再 execute 再 replan"的自定义 JSON 流程，
# 而是让模型在 OpenAI 原生的多轮 function calling 循环里自己一步步决定"先查库存、
# 库存不够就多下单顶替、最后结账"——对比一下这种方式跟自己维护 plan/replan JSON 相比，
# 稳定性有没有变化。


import asyncio
import json
import openai
from dotenv import load_dotenv

load_dotenv()
client = openai.OpenAI()

# ===========================
# price_agent / weather_agent 用到的工具（照抄 v7 例子）
# ===========================

def get_price(item_name: str):
    mock_db = {"react_book": 128, "vue_book": 100, "coffee": 35}
    return mock_db.get(item_name.strip(), "未知商品")

def calculate(expression: str):
    try:
        return eval(expression)
    except Exception as e:
        return f"计算错误: {e}"

def get_weather(city: str):
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
# order_agent 用到的工具（照抄 execise2 里已经修好的版本，直接可用）
# ===========================

_order_state = {}  # item_name -> 累计数量，供 add_to_order / checkout 共用

def check_stock(item_name: str):
    mock_db = {"react_book": 0, "vue_book": 5, "coffee": 3}
    return mock_db.get(item_name.strip(), 0)

def add_to_order(item_name: str, quantity: int):
    _order_state[item_name] = _order_state.get(item_name, 0) + quantity
    return _order_state[item_name]

def checkout():
    return sum(get_price(item) * qty for item, qty in _order_state.items())

ORDER_TOOLS = {"check_stock": check_stock, "add_to_order": add_to_order, "checkout": checkout}
ORDER_TOOLS_CONFIG = [
    {
        "type": "function",
        "function": {
            "name": "check_stock",
            "description": "查询商品剩余库存数量，用于判断是否需要用其他商品顶替",
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
            "name": "add_to_order",
            "description": "把商品加入订单，同一商品多次调用会自动累加数量",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_name": {"type": "string", "description": "商品名称，例如 react_book"},
                    "quantity": {"type": "integer", "description": "购买数量"},
                },
                "required": ["item_name", "quantity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "checkout",
            "description": "结算当前订单里所有商品的总价",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

# ===========================
# Sub-Agent
# ===========================

class SubAgent:
    def __init__(self, name: str, system_prompt: str, tools: dict, tools_config: list):
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools
        self.tools_config = tools_config

    async def run(self, query: str) -> str:
        # TODO: 把下面这段"只处理一轮 tool_calls"的逻辑改造成循环，
        # 直到模型不再请求 tool_calls（或达到最大轮数）为止
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]
        response = client.chat.completions.create(
            model="gpt-4o", messages=messages, tools=self.tools_config
        )
        msg = response.choices[0].message
        content = msg.content
        tool_calls = msg.tool_calls
        loop_count = 0
        
        while tool_calls:
            loop_count += 1
            if loop_count >= 5:
                break
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
            response = client.chat.completions.create(
                model="gpt-4o", messages=messages, tools=self.tools_config
            )
            msg = response.choices[0].message
            content = msg.content
            tool_calls = msg.tool_calls
            if not tool_calls:
                messages.append(msg)

            
        print(f"Sub-Agent: {self.name}, messages: {messages}")
        return content


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

# TODO: 用 ORDER_TOOLS / ORDER_TOOLS_CONFIG 创建 order_agent，system_prompt 里要写清楚
# "先查库存，react_book 不足时用等量 vue_book 顶替，最后调用 checkout 结算"这条业务规则
order_agent = SubAgent(
    name="order_agent",
    system_prompt="你是订单助手，负责下单并返回每个商品的数量，用一句话简洁回答。工作方式：先查询所需商品库存，调用add_to_order添加到订单（商品数量不可超过商品库存），最后调用checkout结算",
    tools=ORDER_TOOLS,
    tools_config=ORDER_TOOLS_CONFIG,
)

AGENTS = {
    "price_agent": price_agent,
    "weather_agent": weather_agent,
    # TODO: 把 order_agent 注册进来
    "order_agent": order_agent,
}

# ===========================
# Orchestrator
# ===========================

# TODO: 在下面加入 order_agent 的说明
ORCHESTRATOR_PROMPT = """
你是一个任务调度器，负责阅读用户的请求，把它拆分成若干独立子任务，
并指明每个子任务应该交给哪个 Sub-Agent 处理。如果用户的请求只涉及一个领域，
就只拆出一个子任务。

可用的 Sub-Agent：
- price_agent：负责商品价格查询、算术计算
- weather_agent：负责天气查询
- order_agent：负责根据用户需求下单，并计算最终价格

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
# Aggregator
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
# 主流程
# ===========================

async def run(user_input: str) -> str:
    tasks = orchestrate(user_input)
    print(f"Orchestrator 拆分结果: {tasks}")

    coros = [AGENTS[t["agent"]].run(t["query"]) for t in tasks]
    results = await asyncio.gather(*coros)
    print(f"各 Sub-Agent 结果: {results}")

    final_answer = aggregate(user_input, results)
    print(f"Question: {user_input}")
    print(f"Answer: {final_answer}")
    return final_answer


if __name__ == "__main__":
    asyncio.run(run("帮我买 2 本 react_book、1 本 vue_book、3 杯 coffee，另外北京天气怎么样"))
