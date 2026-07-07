# 练习二：把"数量合并计算"从 LLM 手里拿走，交给代码
#
# execise 里发现的问题：LLM 现场拼 calculate 表达式做数量加法，经常忘记把
# "顶替买的数量"和"原本就要买的数量"加在一起，正确率不稳定。
#
# 任务：去掉 get_price / calculate 这两个"让 LLM 做算术"的工具，换成：
#   - add_to_order(item_name, quantity)：把商品加入订单，同一商品多次调用会自动累加
#     （累加逻辑在代码里用 dict += 实现，不依赖 LLM 记住之前买了多少）
#   - checkout()：结账，遍历订单里每个商品，代码自己查单价、乘数量、求和，返回总价
#
# LLM 之后的职责变成：check_stock 判断分支 → 每次购买决策调用一次 add_to_order（数量
# 只是一个数字，不用做加法）→ 最后调用 checkout。所有加法乘法都在代码里完成。
#
# 需要你做的：
#   1. 实现 add_to_order，注册进 tools / tools_config
#   2. 实现 checkout（内部调用 _get_price 查单价），注册进 tools / tools_config
#   3. 更新 PLAN_PROMPT：删掉 get_price/calculate 的说明，换成 add_to_order/checkout
#      的用法，并把"合法结尾"规则改成"必须以 checkout 或 need_replan 结尾"
#
# 测试：agent("帮我买 2 本 react_book，1 本 vue_book，3 杯 coffee，总共多少钱")
# 验证：react_book 缺货 2 本 → 用 2 本 vue_book 顶替 → 实际 vue_book x3 + coffee x3
#      = 3*100 + 3*35 = 405

# 这个例子完成后有问题，就是 agent 在 plan 总会忘记需要以 need_replan 结尾，或直接一次 plan 就完成任务，导致最终结果不正确。
# claude 建议我们依赖 v8 的 eval + 自我修正重试。我们先拭目以待


import openai
import json
import asyncio
from dotenv import load_dotenv

# ===========================
# 工具定义
# ===========================

def _get_price(item_name: str):
    """内部函数：查询商品单价，不直接暴露给 LLM 调用"""
    mock_db = {"react_book": 128, "vue_book": 100, "coffee": 35}
    return mock_db.get(item_name.strip(), 0)

def check_stock(item_name: str):
    """查询商品库存（react_book 缺货，制造分支）"""
    mock_db = {"react_book": 0, "vue_book": 5, "coffee": 3}
    return mock_db.get(item_name.strip(), 0)


# TODO: 实现 checkout()
# 提示：同样需要访问 self.order，写成 SimpleAgent 的方法；内部调用 _get_price 算总价。

tools_config = [
    {
        "type": "function",
        "function": {
            "name": "check_stock",
            "description": "查询商品剩余库存数量，用于判断是否需要用其他商品顶替",
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
    # TODO: 补充 add_to_order 的 tools_config 定义（两个参数：item_name, quantity）
    # TODO: 补充 checkout 的 tools_config 定义（无参数，parameters.properties 留空即可）
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
        self.order = {}  # item_name -> 累计数量
        self.tools = {
            "check_stock": check_stock,
            # TODO: 把 add_to_order / checkout 注册进来（注意这两个是 self 的方法）
            "add_to_order": self.add_to_order,
            "checkout": self.checkout,
            
        }

    # TODO: 实现 add_to_order(item_name, quantity)    
    def add_to_order(self, item_name, quantity):
        """把某商品加入订单，数量会自动累加（同一商品调用多次会叠加，不会覆盖）"""
        count = self.order.get(item_name) or 0
        count += int(quantity)
        self.order[item_name] = count
        
    def checkout(self) -> int:
        """结账：用当前累计的订单，代码里循环算总价，不经过 LLM 拼表达式"""
        total: int = 0
        for key, count in self.order.items():
            price: int = _get_price(key)
            total += price * count
        return total

    def plan(self, user_input):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": self.plan_prompt}, {"role": "user", "content": user_input}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)["steps"]

    def replan(self, user_input, execute_cache):
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
            if tool in self.tools:
                result = self.tools[tool](**args)
                self.execute_cache[step_n] = {"tool": tool, "args": args, "result": result}
        print(f"Execute cache: {self.execute_cache}\n")
        last_step = plan[-1]["step"]
        return self.execute_cache[last_step]["result"], self.execute_cache, False

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
        print(f"Plan: {plan}\n")
        result, execute_cache, need_replan = self.execute(plan)
        if need_replan:
            plan = self.replan(user_input, execute_cache)
            print(f"Replan: {plan}\n")
            result, execute_cache, need_replan = self.execute(plan)
        response = self.respond(user_input, execute_cache, result)
        print(f"Question: {user_input}")
        print(f"Answer: {response}")
        return response

    def __call__(self, user_input):
        return asyncio.run(self.async_run(user_input))


# TODO: 把 get_price/calculate 的说明换成 add_to_order/checkout，
# 并把"合法结尾"规则改成"必须以 checkout 或 need_replan 结尾"
PLAN_PROMPT = """
你是一个任务规划师，负责阅读用户购买需求，产出分步骤实施计划。
若本次规划无法一次性完成，请在最后一步告知需要重新规划。

工作方式：
- 需要库存时调用 check_stock
- 需要将物品加入购物车时调用 add_to_order
- 需要计算总价时调用 checkout (只有通过 add_to_order 添加到购物车的商品才会 checkout)
- 若 react_book 不足时，用等量 vue_book 代替

格式要求：
1. 返回一个JSON格式数组，数组每一项表示一个步骤 \\{"steps":[{"step": <n>, "tool": <工具名>, "args": {<步骤内容>}}]\\}
2. 如果库存不足，需要 need_replan
3. [重要]必须以 checkout 或 need_replan 结尾，禁止以 check_stock / add_to_order 结尾

工具说明：
- check_stock (查询商品剩余库存数量): args 格式为 {"item_name": "<商品名>"}
- add_to_order (添加商品到购物车): args 格式为 {"item_name": "<商品名>", quantity: "<数量>"}
- checkout (结算购物车中的商品): args 格式为 {}
- need_replan (当本次规划无法一次性完成时，用于重新规划): args 格式为 {"reason": "<原因>"}
"""


agent = SimpleAgent(PLAN_PROMPT)
result = agent("帮我买 2 本 react_book，1 本 vue_book，3 杯 coffee，总共多少钱")
