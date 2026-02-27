# =========================
# 基于 v2 版本优化，新增了:
# 1. 段落-句子双层结构，检索到相关句子后可以回溯到对应段落，提供更丰富的上下文给 LLM。
# 这主要是用于解决当信息零散在不同段落（实际生产环境有可能是不同文章），但他们都属于同一主题时，单纯检索句子可能无法提供足够的上下文信息。
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

def sub_sentence_chunking(text, client, model="text-embedding-3-small", breakpoint_percentile=50):

    paragraphs = re.split(r'--------------------------------', text)
    paraDict = {i: para for i, para in enumerate(paragraphs)}
    paraToSentences = {}
    chunks = []
    for i, para in enumerate(paragraphs):
        sentences = re.split(r'(?<=[。！？])\s*', para)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        
        sentenceArr = paraToSentences.get(i)
        if sentenceArr is None:
            paraToSentences[i] = []

        for j, sentence in enumerate(sentences):
            paraToSentences[i].append(sentence)
            chunks.append(sentence)

    print(f"共切分出 {len(paragraphs)} 个段落。")
    print(f"共切分出 {len(chunks)} 个句子。")

    return paraDict, paraToSentences, chunks

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
        if distance < 1.5: # 这个距离阈值可以根据实际情况调整
            results.append(chunks[idx]) 
    # print("top_k chunks:", results)
    return results

def getParagraphsFromChunks(chunks, paraDict, paraToSentences):
    paragraphs = set()
    for chunk in chunks:
        for paraId, sentences in paraToSentences.items():
            if chunk in sentences:
                paragraphs.add(paraDict[paraId])
    return list(paragraphs)

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
    text = load_text("./rag/knowledge_base_entangled.txt")

    print("切分文本...")
    paraDict, paraToSentences, chunks = sub_sentence_chunking(text, client=client)

    # print("paraDict:", paraDict)
    # print("paraToSentences:", paraToSentences)

    index, _ = build_faiss_index(chunks)

    print("\nRAG 系统已启动！输入问题（输入 exit 退出）\n")

    while True:
        question = input("你的问题：")

        if question.lower() == "exit":
            break

        searchChunks = search(index, chunks, question, top_k=5)
        
        # print("检索到的相关句子：")
        # for i, chunk in enumerate(searchChunks):
        #     print(chunk)
        
        contexts = getParagraphsFromChunks(searchChunks, paraDict, paraToSentences)

        # print("检索到的相关段落：")
        # for i, context in enumerate(contexts):
        #     print(context)
        
        prompt = build_prompt(contexts, question)

        answer = ask_llm(prompt)

        print("\n回答: ")
        print(answer)
        print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    main()