import openai
import re
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

load_dotenv()
client = openai.OpenAI()

class SimpleAgent:
    def __init__(self, system_prompt):
        self.system_prompt = system_prompt
        self.messages = [{"role": "system", "content": system_prompt}]

    def __call__(self, user_input):
        self.messages.append({"role": "user", "content": user_input})
        
        # 最多循环 5 次，防止思考掉入死循环
        for i in range(5):
            print(f"\n--- 🤖 Agent 正在思考 (第 {i+1} 轮) ---")
            # print(f"input messages: {self.messages}\n")
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=self.messages,
                # 核心技巧：看到 Observation 就停，等工具执行结果
                # stop 可以设置多个值，如 ["Observation:", "User:", "###"]，看到 User: 就停（防止 AI 模拟用户说话）
                stop=["Observation:"]
            )
            
            content = response.choices[0].message.content
            
            print(f"{content}\n")
            
            # 将 AI 的思考加入上下文
            self.messages.append({"role": "assistant", "content": content})

            # 1. 检查是否得到最终答案
            if "Final Answer:" in content:
                return content.split("Final Answer:")[1].strip()

            # 2. 解析 Action (正则提取)
            # 匹配格式如：Action: get_price("react_book")
            match = re.search(r"Action:\s*(\w+)\((.*)\)", content)
            if match:
                func_name = match.group(1)
                func_args = match.group(2).strip("'\"") # 去掉参数的引号
                
                # 3. 执行工具 (Action)
                if func_name in tools:
                    print(f"🛠️  执行工具: {func_name}({func_args})\n")
                    observation = tools[func_name](func_args)
                    
                    # 4. 反馈结果 (Observation)
                    obs_text = f"Observation: {observation}"
                    print(f"obs_text: {obs_text}\n")
                    self.messages.append({"role": "user", "content": obs_text})
                else:
                    error_msg = f"Observation: 错误，找不到工具 {func_name}"
                    self.messages.append({"role": "user", "content": error_msg})

# --- 系统提示词 (System Prompt) ---
SYSTEM_PROMPT = """
你是一个财务助理。你可以调用工具来解决问题。
可用工具：
- get_price(item_name): 获取商品单价。参数是字符串。
- calculate(expression): 执行数学计算。参数是数学表达式字符串。

输出格式必须遵循：
Thought: 你的思考过程
Action: 工具名(参数)
Observation: (这里由系统填入)
... (重复)
Final Answer: 最终结论
"""

# --- 运行测试 ---
agent = SimpleAgent(SYSTEM_PROMPT)
result = agent("买两本 react_book，加一杯 coffee，总价打 9 折是多少钱？")
print(f"\n✅ 最终结果: {result}")