# 这是基于 v1 的优化版本，新增了：
# 1. 使用 function calling  替换 v1 的 Action: 和 Observation: 正则匹配
# 2. function calling 改为并行（非 Action: 中的串行）
import openai
import re
import json
import asyncio
from dotenv import load_dotenv

# 工具 1: 模拟数据库查询
def get_price(item_name: str):
    """查询商品单价"""
    mock_db = {"react_book": 128, "vue_book": 100, "coffee": 35}
    return mock_db.get(item_name.strip(), "未知商品")

# 工具 2: 数学计算
def calculate(expression: str):
    """执行数学运算"""
    try:
        # 学习提示：在生产环境建议使用 math 库或专门的解析器
        return eval(expression)
    except Exception as e:
        return f"计算错误: {e}"

# 将工具组织成字典，方便动态调用
tools = {
    "get_price": get_price,
    "calculate": calculate
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
    }
]

load_dotenv()
client = openai.OpenAI()



class SimpleAgent:
    def __init__(self, system_prompt):
        self.system_prompt = system_prompt
        self.messages = [{"role": "system", "content": system_prompt}]

    def __call__(self, user_input):
        return asyncio.run(self.async_run(user_input))
    
    async def async_run(self, user_input):
        self.messages.append({"role": "user", "content": user_input})
        
        # 最多循环 5 次，防止思考掉入死循环
        for i in range(5):
            print(f"\n--- 🤖 Agent 正在思考 (第 {i+1} 轮) ---")
            print(f"input messages: {self.messages}\n")
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=self.messages,
                tools=tools_config,
                tool_choice="auto",
            )
            
            content = response.choices[0].message.content
            tool_calls = response.choices[0].message.tool_calls
            
            print(f"output content: {content}\n")
            print(f"tool_calls: {tool_calls}\n")
            
                        # 无 tool_calls 时才是最终文字答案
            if not tool_calls:
                if "Final Answer:" in content:
                    return content.split("Final Answer:")[1].strip()
                continue   # 没有工具调用也没 Final Answer，继续下一轮
            
            # 必须：先追加带 tool_calls 的 assistant 消息（有 tool_calls 时 API 要求这条）
            assistant_msg = {"role": "assistant", "content": content}
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in tool_calls
                ]
                self.messages.append(assistant_msg)
                # 使用 list comprehension 创建任务
                tasks = [self.async_execute_tool(tc) for tc in tool_calls]
                # 类似 Promise.all()
                results = await asyncio.gather(*tasks) 
                self.messages.extend(results)
            else:
                self.messages.append(assistant_msg)

    async def async_execute_tool(self, tool_call):
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)
        print(f"tool_name: {tool_name}, tool_args: {tool_args}\n")
        if tool_name in tools:
            observation = tools[tool_name](**tool_args)
            return {"tool_call_id": tool_call.id, "role": "tool", "name": tool_name, "content": str(observation)}
        else:
            error_msg = f"Observation: 错误，找不到工具 {tool_name}"
            return {"role": "user", "content": error_msg}

# --- 系统提示词 (System Prompt) ---
SYSTEM_PROMPT = """
你是一个财务助理，负责根据用户需求查询商品单价、计算总价或折扣等。

工作方式：
- 需要单价时调用 get_price；需要算数时调用 calculate。
- 先查清所有涉及的单价，再计算总价或折扣。
- 得到最终数字后，不要继续调用工具，直接在回复中给出结论。

最终回复格式（必须遵守）：
在最后一段用「Final Answer:」开头写出你的结论，例如：
Final Answer: 两本 react_book 加一杯 coffee 打 9 折后总价是 xxx 元。

若某商品查不到或无法计算，在 Final Answer 中说明原因并用中文回复用户。
"""

# --- 运行测试 ---
agent = SimpleAgent(SYSTEM_PROMPT)
result = agent("买两本 react_book，加一杯 coffee，总价打 9 折是多少钱？")
print(f"\n✅ 最终结果: {result}")