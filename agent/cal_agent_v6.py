# 基于 v4 的进阶版本，修改了：
# 1. 将原有 ReAct 架构换成 Plan-Execute 架构
# 
# 但这里的优缺点也需要注意：
# 优点：
# 1. ReAct 做一步想一步容易出现偏差，先规划好路径再做就好很多
# 2. 一开始就让 LLM 告知我们用什么工具进行，然后就可以在 agent 自己调用解决，减少调用 LLM 频次和 token 
# 3. 耗时更少
# 
# 缺点：
# 1. Plan 无法根据中间结果进行调整，所以现实中往往用 混合模式，先执行 Plan，执行中发现异常再重新 Plan



import uuid

import openai
import json
import asyncio
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
from dotenv import load_dotenv

# ===========================
# 工具定义（与 v2 相同）
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

SCRIPT_DIR = Path(__file__).resolve().parent
db_path = SCRIPT_DIR / "chroma_db"

load_dotenv()
client = openai.OpenAI()


# ===========================
# 短期记忆模块
# ===========================

class ShortTermMemory:
    """
    Agent 的短期记忆，存储本次对话中已查询过的信息。

    设计思路：
    - key = "price:{item_name}"，value = 价格数字
    - 每次执行 get_price 工具前先查记忆，命中则跳过 API 调用
    - 对话结束后记忆自动消失（非持久化）

    类比：就像你去超市，第一次看到矿泉水 3 块，
    下次再看价签时你直接记得 3 块，不用重新看牌子。
    """

    def __init__(self):
        self.price_cache: dict = {}

    def remember_price(self, item_name: str, price):
        """存入价格记忆"""
        self.price_cache[item_name] = price
        print(f"📝 [记忆] 已记住 {item_name} 的价格: {price}")

    def recall_price(self, item_name: str):
        """
        尝试从记忆中读取价格。
        返回 (hit, value)：hit=True 表示命中缓存
        """
        key = item_name
        if key in self.price_cache:
            print(f"💡 [记忆命中] {item_name} 的价格已在记忆中: {self.price_cache[key]}，跳过工具调用")
            return True, self.price_cache[key]
        return False, None

    def summary(self) -> str:
        """输出当前记忆内容，方便调试"""
        if not self.price_cache:
            return "（记忆为空）"
        lines = [f"  {k}: {v}" for k, v in self.price_cache.items()]
        return "\n".join(lines)

# ===========================
# 长期记忆模块
# ===========================

class LongTermMemory:
    """长期记忆，存储所有对话摘要"""
    def __init__(self):
        # 创建持久化客户端，数据会存在本地目录
        chroma_client = chromadb.PersistentClient(path=db_path)
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        # 获取或创建集合, 自动 embedding
        self.collection = chroma_client.get_or_create_collection(name="conversation_memory", embedding_function=ef)
        
    def remember(self, summary: str):
        """存入对话摘要"""
        self.collection.add(
            ids=[str(uuid.uuid4())],
            documents=[summary],
            metadatas=[{"source": "conversation"}]
        )
        
    def retrieve(self, query: str):
        """检索对话摘要"""
        results = self.collection.query(
            query_texts=[query],
            n_results=3
        )
        return results['documents'][0]

# ===========================
# Agent 主体
# ===========================

class SimpleAgent:
    def __init__(self, system_prompt):
        self.system_prompt = system_prompt
        # messages 维持多轮对话上下文，不在每次 __call__ 时重置
        # 区别于 v2：v2 每次调用都是全新对话
        self.messages = [{"role": "system", "content": system_prompt}]
        # 新增：短期记忆
        self.memory = ShortTermMemory()
        self.long_memory = LongTermMemory()
        self.execute_cache = {}

    def __call__(self, user_input):
        """注意，在这里重设了所有输入 LLM 的上下文，为了抹去第一轮对话的信息，因为第一轮对话有 react_book 和 coffee 的价格"""
        """用这样的方式让 LLM 在遇到 react_book 时重新向 agent 提问"""
        self.messages = [{"role": "system", "content": self.system_prompt}]
        summaries = self.long_memory.retrieve(user_input)
        self.messages.append({"role": "system", "content": f"根据之前的对话，{user_input} ，的摘要是：{summaries}"})
        # return asyncio.run(self.async_run(user_input))
        return asyncio.run(self.async_run(user_input))

    def reset(self):
        """显式重置对话（开始全新话题时调用）"""
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.memory = ShortTermMemory()
        print("🔄 对话已重置")
        
    def plan(self, user_input):
        plan_prompt = """
            你是一个任务规划师，负责阅读用户需求，产出分步骤实施计划。
            若本次规划无法一次性完成，请在最后一步告知需要重新规划。

            格式要求：
            1. 返回一个JSON格式数组，数组每一项表示一个步骤 \{"steps":[{"step": <n>, "tool": <工具名>, "args": {<步骤内容>}}]\}
            2. 如果该步骤依赖之前步骤，请使用 stepn 来表示之前的内容。
            
            工具说明：
            - get_price: args 格式为 {"item_name": "<商品名>"}
            - calculate: args 格式为 {"expression": "<数学表达式>"}
            - need_replan: args 格式为 {"reason": "<需要重新规划的原因>"}
            表达式中可以用 step1、step2 等引用前面步骤的结果
            
            示例输出：
            {"steps": [
                {"step": 1, "tool": "get_price", "args": {"item_name": "react_book"}},
                {"step": 2, "tool": "calculate", "args": {"expression": "step1 * 0.5"}}
                {"step": 3, "tool": "need_replan", "args": {"reason": "需要根据价格筛选符合条件的商品后再计算"}} // 如果需要
            ]}
        """
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": plan_prompt}, {"role": "user", "content": user_input}],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        plan = json.loads(content)["steps"]
        return plan
    
    def replan(self, user_input, execute_cache):
        replan_prompt = """
            你是一个任务规划师，负责阅读用户需求，并根据当前执行结果产出下一阶段的分步骤实施计划。

            格式要求：
            1. 返回一个JSON格式数组，数组每一项表示一个步骤 \{"steps":[{"step": <n>, "tool": <工具名>, "args": {<步骤内容>}}]\}
            2. 如果该步骤依赖之前步骤，请使用 stepn 来表示之前的内容。
            
            工具说明：
            - get_price: args 格式为 {"item_name": "<商品名>"}
            - calculate: args 格式为 {"expression": "<数学表达式>"}
            表达式中可以用 step1、step2 等引用前面步骤的结果
            
            示例输出：
            {"steps": [
                {"step": 1, "tool": "get_price", "args": {"item_name": "react_book"}},
                {"step": 2, "tool": "calculate", "args": {"expression": "step1 * 0.5"}}
            ]}
        """
        user_prompt = f"用户问题是：{user_input}，当前执行结果是：{execute_cache}"
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": replan_prompt}, {"role": "user", "content": user_prompt}],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        plan = json.loads(content)["steps"]
        return plan
        
    def execute(self, plan):
        
        for step in plan:
            # print(f"Step {step}")
            tool, args, step_n = step["tool"], step["args"], step["step"]
            if tool == "need_replan":
                return None, self.execute_cache, True
            if tool == "calculate":
                for key, val in self.execute_cache.items():
                        args["expression"] = args["expression"].replace(f"step{key}", str(val))
                        # print(f"After replace: {args}")
            if tool in tools:
                result = tools[tool](**args)
                self.execute_cache[step_n] = result
        print(f"Execute cache: {self.execute_cache}")
        last_step = plan[-1]["step"]
        return self.execute_cache[last_step], self.execute_cache, False
    
    def respond(self, user_input, execute_cache, result):
        respond_prompt = """
            你是一个财务助理，用自然语言回答用户问题。
        """
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": respond_prompt}, 
                      {"role": "user", "content": f"用户问题是：{user_input}，执行过程产物是：{execute_cache}，最终结果是：{result}，请直接给出一句话的简洁回答，不要推导过程"}]
        )
        content = response.choices[0].message.content
        return content
    
    async def async_run(self, user_input):
        plan = self.plan(user_input=user_input)
        print(f"Plan: {plan}")
        result, execute_cache, need_replan = self.execute(plan)
        if need_replan:
            plan = self.replan(user_input, execute_cache)
            print(f"Replan: {plan}")
            result, execute_cache, need_replan = self.execute(plan)
        response = self.respond(user_input, execute_cache, result)
        print(f"Answer: {response}")
        return response

# ===========================
# 系统提示词
# ===========================

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


# ===========================
# 多轮对话测试
# 演示记忆的效果：
#   第 1 轮：查询 react_book + coffee 价格（正常调用工具）
#   第 2 轮：再次询问 react_book 相关问题
#             → 观察 react_book 价格直接从记忆读取，不重复调用 get_price
# ===========================

agent = SimpleAgent(SYSTEM_PROMPT)

# result1 = agent("买两本 react_book 加一杯 coffee，打九折多少钱？")
result2 = agent("帮我买所有价格不超过 110 元的商品（react_book、vue_book、coffee），总共花多少钱？")

