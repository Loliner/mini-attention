# =========================
# 0. 原理
# 我们分了好多个 chunk（4个），把它们转成 4 个 1536 维张量。
# 当我们输入问题的时候，用同样一个 embedding 算法将问题转换为一个 1536 维张量。
# 然后用这个问题张量去和4个chunk的张量进行点积运算，找到最靠近的那个（比如如果问题和对应chunk都包含某个关键字或相似语义，那么这两个张量就考的更近）。
# 然后根据匹配到的张量的 chunk 和 问题扔给LLM，这样就可以限制 LLM 搜索范围。
# =========================
import os
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv

# =========================
# 1. 初始化
# =========================

load_dotenv()
client = OpenAI()

EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

# =========================
# 2. 文档加载
# =========================

def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# =========================
# 3. 切分文本
# chunk_size 应该更合理的划分，比如遇到句号或 \n\n 时切分，避免切断句子。
# overlap 即使生硬的切断了某个句子，也可以让下一chunk包含这个完整的句子。
# =========================

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

# =========================
# 4. 生成 embedding
# =========================

def get_embedding(text):
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    # print('embedding response:', response)
    return response.data[0].embedding

# =========================
# 5. 构建向量数据库
# =========================

def build_faiss_index(chunks):
    print("正在生成 embedding...")

    vectors = []
    for i, chunk in enumerate(chunks):
        print(f"Embedding chunk {i+1}/{len(chunks)}")
        vec = get_embedding(chunk)
        print("chunk:", len(chunk), "vec:", len(vec))
        # 此时 vectors 存储的是每个 vec 的指针，而不是 vec 本身，所以后续需要转换成 numpy 数组
        vectors.append(vec)

    # 将在内存中分散的每个 vec 组恒合成一个二维 numpy 数组，才能被 faiss 正确处理
    # faais 只支持 float32
    vectors = np.array(vectors).astype("float32")
    print("vectors shape:", vectors.shape) # (num_chunks, embedding_dim) = (4, 1536)

    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)

    return index, vectors

# =========================
# 6. 检索
# =========================

def search(index, chunks, query, top_k=3):
    q_vec = get_embedding(query)
    print("query vec length:", len(q_vec)) # 1536
    query_vec = np.array([q_vec]).astype("float32")
    D, I = index.search(query_vec, top_k)

    results = []
    for idx in I[0]:
        results.append(chunks[idx])
    print("top_k chunks:", results)
    return results

# =========================
# 7. 构造 Prompt
# =========================

def build_prompt(contexts, question):
    context_text = "\n\n".join(contexts)

    return f"""
你只能基于以下内容回答问题。
如果无法从中找到答案，请回答：不知道。

内容：
{context_text}

问题：
{question}
"""

# =========================
# 8. 调用 LLM
# =========================

def ask_llm(prompt):
    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# =========================
# 9. 主流程
# =========================

def main():
    print("加载文档...")
    text = load_text("./rag/knowledge_base.txt")

    print("切分文本...")
    chunks = chunk_text(text)

    print(f"共生成 {len(chunks)} 个 chunks")

    index, _ = build_faiss_index(chunks)

    print("\nRAG 系统已启动！输入问题（输入 exit 退出）\n")

    while True:
        question = input("你的问题：")

        if question.lower() == "exit":
            break

        contexts = search(index, chunks, question, top_k=3)

        prompt = build_prompt(contexts, question)

        answer = ask_llm(prompt)

        print("\n回答：")
        print(answer)
        print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    main()