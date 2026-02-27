# =========================
# 基于 v1 版本优化，新增了:
# 1. 更合理的切分文本（基于语义跳变点）
# 2. 检索时增加了距离阈值，过滤掉一些不相关的 chunk
# =========================
import os
import re
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

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

def semantic_chunking(text, client, model="text-embedding-3-small", breakpoint_percentile=50):
    # 1. 粗暴但有效的按句切分
    sentences = re.split(r'--------------------------------', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    print(f"共切分出 {len(sentences)} 个句子。")
    for i, sentence in enumerate(sentences):
        print(f"--- Sentence {i+1} ---")
        print(sentence[:200])
    
    # 2. 为每一句生成 Embedding
    # 既然你懂原理，这里可以考虑做“滑动窗口聚合”，但我们先做最基础的
    print(f"正在分析 {len(sentences)} 个句子的语义...")
    res = client.embeddings.create(input=sentences, model=model)
    embeddings = [d.embedding for d in res.data]
    
    # 3. 计算相邻句子的余弦距离
    distances = []
    for i in range(len(embeddings) - 1):
        similarity = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
        distances.append(1 - similarity) # 距离 = 1 - 相似度
    
    print(f"相邻句子之间的距离：", distances)
    # 4. 寻找异常点（语义跳变点）
    threshold = np.percentile(distances, breakpoint_percentile)
    
    chunks = []
    current_chunk = sentences[0]
    
    for i, dist in enumerate(distances):
        if dist > threshold:
            # 距离过大，在此处切断
            chunks.append(current_chunk)
            current_chunk = sentences[i+1]
        else:
            current_chunk += "\n\n" + sentences[i+1]
            
    chunks.append(current_chunk)
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
    print("D:", D)
    print("I:", I)

    results = []
    for distance, idx in zip(D[0], I[0]):
        if distance < 1.3: # 这个距离阈值可以根据实际情况调整
            results.append(chunks[idx]) 
    # print("top_k chunks:", results)
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
    chunks = semantic_chunking(text, client=client)

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

        print("\n回答: ")
        print(answer)
        print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    main()