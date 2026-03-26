# 基于 v3 的进阶版本，新增了：
# 1. Long-term Memory（长期记忆）：使用 ChromaDB 持久化存储对话摘要，跨会话可读取


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

    def __call__(self, user_input):
        """注意，在这里重设了所有输入 LLM 的上下文，为了抹去第一轮对话的信息，因为第一轮对话有 react_book 和 coffee 的价格"""
        """用这样的方式让 LLM 在遇到 react_book 时重新向 agent 提问"""
        self.messages = [{"role": "system", "content": self.system_prompt}]
        summaries = self.long_memory.retrieve(user_input)
        self.messages.append({"role": "system", "content": f"根据之前的对话，{user_input} ，的摘要是：{summaries}"})
        return asyncio.run(self.async_run(user_input))

    def reset(self):
        """显式重置对话（开始全新话题时调用）"""
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.memory = ShortTermMemory()
        print("🔄 对话已重置")

    async def async_run(self, user_input):
        self.messages.append({"role": "user", "content": user_input})

        for i in range(5):
            print(f"\n--- 🤖 Agent 正在思考 (第 {i+1} 轮) ---")
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=self.messages,
                tools=tools_config,
                tool_choice="auto",
            )

            content = response.choices[0].message.content
            tool_calls = response.choices[0].message.tool_calls

            print(f"output content: {content}")
            print(f"tool_calls: {tool_calls}")

            if not tool_calls:
                if content and "Final Answer:" in content:
                    summary = f"用户问:{user_input}，结论:{content}"
                    self.long_memory.remember(summary)
                    print(f"📝 [长期记忆] 已记住 {summary}")
                    return content.split("Final Answer:")[1].strip()
                continue

            # 追加 assistant 消息（带 tool_calls 字段，API 要求）
            assistant_msg = {
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in tool_calls
                ]
            }
            self.messages.append(assistant_msg)

            # 并行执行所有工具（与 v2 相同）
            tasks = [self.async_execute_tool(tc) for tc in tool_calls]
            results = await asyncio.gather(*tasks)
            self.messages.extend(results)

    async def async_execute_tool(self, tool_call):
        """
        执行工具，新增记忆层：
        - get_price：先查记忆，命中直接返回缓存；未命中则调用工具并存入记忆
        - calculate：直接执行（结果不缓存，因为表达式每次不同）
        """
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)
        print(f"\n🔧 工具调用: {tool_name}({tool_args})")

        # --- 记忆层拦截 ---
        if tool_name == "get_price":
            item_name = tool_args.get("item_name", "")
            hit, cached_price = self.memory.recall_price(item_name)
            if hit:
                # 命中记忆：直接返回，不调用真实工具
                print(f"命中短期记忆，{item_name}价格是{cached_price}")
                observation = cached_price
            else:
                # 未命中：调用工具，并把结果存入记忆
                observation = tools["get_price"](**tool_args)
                self.memory.remember_price(item_name, observation)
        elif tool_name in tools:
            observation = tools[tool_name](**tool_args)
        else:
            observation = f"错误：找不到工具 {tool_name}"

        print(f"📦 工具结果: {observation}")
        return {
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": tool_name,
            "content": str(observation)
        }


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

print("=" * 50)
print("第 1 轮提问")
print("=" * 50)
result1 = agent("买两本 react_book 加一杯 coffee，打九折多少钱？")
print(f"\n✅ 第 1 轮结果: {result1}")

print("\n" + "=" * 50)
print("第 2 轮提问（react_book 价格应命中记忆）")
print("=" * 50)
# 注意：实际上由于第一轮对话
result2 = agent("如果再加一本 vue_book，总共多少钱？不打折了")
print(f"\n✅ 第 2 轮结果: {result2}")

print("\n" + "=" * 50)
print("当前记忆内容：")
print(agent.memory.summary())
