# 练习：给 v6 加一个"必须触发 replan"的场景
#
# v6 里的 need_replan 从来没被真正触发过。原因：单纯"数值要等前面步骤算完才知道"
# 不需要 replan（steps 里本来就能用 stepN 占位符引用，执行时统一替换即可）。
# 真正需要 replan 的是"分支"——下一步该走哪条路取决于一个只有执行后才知道的判断结果。
#
# 任务：新增工具 check_stock(item_name) -> int，返回库存数量，把 react_book 设成 0（缺货）。
# 业务规则：买东西前先查库存，库存不够就用等量 vue_book 顶替缺货部分。
# 需要你做的：
#   1. 实现 check_stock，注册进 tools / tools_config
#   2. plan/replan 的 prompt 里加上 check_stock 的说明和使用时机
#   3. SYSTEM_PROMPT 里写清楚顶替规则
#
# 测试：agent("帮我买 2 本 react_book，1 本 vue_book，3 杯 coffee，总共多少钱")
# 验证：react_book 缺货 2 本 → 用 2 本 vue_book 顶替 → 实际 vue_book x3 + coffee x3
#      = 3*100 + 3*35 = 405


import openai
import json
import asyncio
from dotenv import load_dotenv

# ===========================
# 工具定义
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

# TODO: 实现 check_stock，返回商品库存数量（把 react_book 设成 0）
def check_stock(item_name: str):
    # raise NotImplementedError("TODO: 实现库存查询")
    mock_db = {"react_book": 0, "vue_book": 5, "coffee": 3}
    return mock_db.get(item_name.strip(), 0)

tools = {
    "get_price": get_price,
    "calculate": calculate,
    # TODO: 把 check_stock 注册进来
    "check_stock": check_stock
}

tools_config = [
    {
        "type": "function",
        "function": {
            "name": "get_price",
            "description": "根据商品名称查询单价",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_name": {
                        "type": "string",
                        "description": "商品的名称，例如 react_book",
                    }
                },
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
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，例如 1 + 1",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    # TODO: 补充 check_stock 的 tools_config 定义
    {
        "type": "function",
        "function": {
            "name": "check_stock",
            "description": "检查库存",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_name": {
                        "type": "string",
                        "description": "商品的名称，例如 react_book",
                    }
                },
                "required": ["item_name"],
            },
        },
    },
]

load_dotenv()
client = openai.OpenAI()

# ===========================
# Agent 主体
# ===========================

class SimpleAgent:
    def __init__(self, plan_prompt):
        self.plan_prompt = plan_prompt
        self.execute_cache = {}

    def plan(self, user_input):
        # TODO: 加入 check_stock 的说明，告诉 LLM 库存不足时要 need_replan
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": self.plan_prompt}, {"role": "user", "content": user_input}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)["steps"]

    def replan(self, user_input, execute_cache):
        # TODO: 同样加入 check_stock 说明
        user_prompt = f"用户问题是：{user_input}，当前执行结果是：{execute_cache}"
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": self.plan_prompt}, {"role": "user", "content": user_prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)["steps"]

    def execute(self, plan):
        for step in plan:
            tool, args, step_n = step["tool"], step["args"], step["step"]
            if tool == "need_replan":
                return None, self.execute_cache, True
            if tool == "calculate":
                for key, val in self.execute_cache.items():
                    args["expression"] = args["expression"].replace(f"step{key}", str(val["result"]))
            if tool in tools:
                result = tools[tool](**args)
                self.execute_cache[step_n] = {"tool": tool, "args": args, "result": result}
        print(f"Execute cache: {self.execute_cache}")
        last_step = plan[-1]["step"]
        return self.execute_cache[last_step], self.execute_cache, False

    def respond(self, user_input, execute_cache, result):
        respond_prompt = "你是一个财务助理，用自然语言回答用户问题。"
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": respond_prompt},
                      {"role": "user", "content": f"用户问题是：{user_input}，执行过程产物是：{execute_cache}，最终结果是：{result}，请直接给出一句话的简洁回答，不要推导过程"}]
        )
        return response.choices[0].message.content

    async def async_run(self, user_input):
        plan = self.plan(user_input)
        print(f"Plan: {plan}")
        result, execute_cache, need_replan = self.execute(plan)
        if need_replan:
            plan = self.replan(user_input, execute_cache)
            print(f"Replan: {plan}")
            result, execute_cache, need_replan = self.execute(plan)
        response = self.respond(user_input, execute_cache, result)
        print(f"Question: {user_input}")
        print(f"Answer: {response}")
        return response

    def __call__(self, user_input):
        return asyncio.run(self.async_run(user_input))


# TODO: 写清楚"库存不足用 vue_book 顶替"的规则
# SYSTEM_PROMPT = """
# 你是一个财务助理，负责根据用户需求查询商品库存、单价、总价。

# 工作方式：
# - 需要库存时调用 check_stock; 需要单价时调用 get_price; 需要算数时调用 calculate; 
# - 先调用 check_stock 查询库存，再查询价格；商品库存不足时，不再调用 get_price 获取该商品价格
# - 某书籍库存不足时，用另一种书籍代替
# - 得到最终数字后，不要继续调用工具，直接在回复中给出结论。

# 最终回复格式（必须遵守）：
# 在最后一段用「Final Answer:」开头写出你的结论。
# """

PLAN_PROMPT ="""
你是一个任务规划师，负责阅读用户需求，产出分步骤实施计划。
若本次规划无法一次性完成，请在最后一步告知需要重新规划。
得到最终数字后，不要继续调用工具，直接在回复中给出结论。

工作方式：
- 需要库存时调用 check_stock; 需要单价时调用 get_price; 需要算数时调用 calculate; 
- 先调用 check_stock 查询库存，再查询价格；商品库存不足时，不再调用 get_price 获取该商品价格
- 若 react_book 不足时，用等量 vue_book 代替

顶替规则举例：
用户要买 3 本 react_book、2 本 vue_book、1 杯 coffee。
若 check_stock 显示 react_book 库存为 0（不足 3 本），则改为多买 3 本 vue_book 顶替，
此时 vue_book 的最终购买数量 = 原本的 2 本 + 顶替的 3 本 = 5 本，而不是只算顶替的 3 本 或 原本的 2 本。
最终总价 = 5 * vue_book单价 + 1 * coffee单价。

格式要求：
1. 返回一个JSON格式数组，数组每一项表示一个步骤 \{"steps":[{"step": <n>, "tool": <工具名>, "args": {<步骤内容>}}]\}
2. 如果该步骤依赖之前步骤，请使用 stepn 来表示之前的内容。
3. 如果库存不足，需要 need_replan
4. 一个计划只有两种合法的结尾：
   - 最后一步是 calculate，算出的是用户需要支付的最终总价
   - 最后一步是 need_replan（比如库存结果还没查到，无法确定最终总价）
   除此之外都不算规划完成，禁止计划以 check_stock 或 get_price 结尾。
   在没有实际执行 check_stock 之前，你不知道库存是否充足，所以只要计划里包含
   check_stock，且后续步骤依赖它的结果，就必须以 need_replan 结尾，等下一轮拿到
   真实库存数字后再继续规划 get_price / calculate。

工具说明：
- get_price (查询商品单价，用于计算总价): args 格式为 {"item_name": "<商品名>"}
- check_stock (查询商品剩余库存数量，用于判断是否需要用其他商品顶替) : args 格式为 {"item_name": "<商品名>"}
- calculate 计算: args 格式为 {"expression": "<数学表达式>"}
- need_replan (当本次规划无法一次性完成时，用于重新规划): args 格式为 {"reason": "<需要重新规划的原因>"}
表达式中可以用 step1、step2 等引用前面步骤的结果

示例输出：
{"steps": [
    {"step": 1, "tool": "get_price", "args": {"item_name": "react_book"}},
    {"step": 2, "tool": "calculate", "args": {"expression": "step1 * 0.5"}}
]}
"""


agent = SimpleAgent(PLAN_PROMPT)
result = agent("帮我买 2 本 react_book，1 本 vue_book，3 杯 coffee，总共多少钱")
