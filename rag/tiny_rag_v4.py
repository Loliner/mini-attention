# =========================
# 基于 v3 版本优化，新增了:
# 1. 返回引用的段落，增强回答的可解释性
# =========================
import os
import re
import uuid
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

def short_id():
    return uuid.uuid4().hex[:6]

def sub_sentence_chunking(text, client, model="text-embedding-3-small", breakpoint_percentile=50):
    raw_paragraphs = re.split(r'--------------------------------', text)

    paraDict = {} # para_id -> {"content": ..., "title": ...}
    sentenceDict = {} # sent_id -> {"content": ..., "para_id": ...}
    chunks = []  # 句子文本列表，用于 FAISS 位置索引

    for para in raw_paragraphs:
        para_id = short_id()
        lines = [l.strip() for l in para.splitlines() if l.strip()] # 提取出所有行，并去掉空行
        title = lines[0] if lines else para_id # 直接将第一行作为 title
        paraDict[para_id] = {"content": para, "title": title}

        sentences = re.split(r'(?<=[。！？])\s*', para)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

        for sentence in sentences:
            sent_id = short_id()
            sentenceDict[sent_id] = {"content": sentence, "para_id": para_id}
            chunks.append(sentence)

    print(f"共切分出 {len(paraDict)} 个段落。")
    print(f"共切分出 {len(sentenceDict)} 个句子。")
    # print(paraDict)
    return paraDict, sentenceDict, chunks

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
        # print(f"Embedding chunk {i+1}/{len(chunks)}")
        vec = get_embedding(chunk)
        # print("chunk:", len(chunk), "vec:", len(vec))
        # 此时 vectors 存储的是每个 vec 的指针，而不是 vec 本身，所以后续需要转换成 numpy 数组
        vectors.append(vec)

    # 将在内存中分散的每个 vec 组恒合成一个二维 numpy 数组，才能被 faiss 正确处理
    # faais 只支持 float32
    vectors = np.array(vectors).astype("float32")
    # print("vectors shape:", vectors.shape) # (num_chunks, embedding_dim) = (4, 1536)

    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)

    return index, vectors

# =========================
# 6. 检索
# =========================

def search(index, chunks, query, top_k=3):
    q_vec = get_embedding(query)
    # print("query vec length:", len(q_vec)) # 1536
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

def getParagraphsFromChunks(chunks, paraDict, sentenceDict):
    # 返回结构：[{"para_id": ..., "title": ..., "content": ..., "matched_sents": [{"sent_id": ..., "index": ..., "content": ...}]}]
    chunk_set = set(chunks)
    para_map = {}  # para_id -> matched sent_ids list
    for sent_id, sent_info in sentenceDict.items():
        if sent_info["content"] in chunk_set:
            para_id = sent_info["para_id"]
            if para_id not in para_map:
                para_map[para_id] = []
            para_map[para_id].append({"sent_id": sent_id, "content": sent_info["content"]})

    result = [] # {"para_id": ..., "title": ..., "content": ..., "matched_sents": [{"sent_id": ..., "index": ..., "content": ...}]}
    for pid, matched in para_map.items():
        para = paraDict[pid]
        # 重新切分段落，以确定每个匹配句子的真实序号
        all_sents = re.split(r'(?<=[。！？])\s*', para["content"])
        all_sents = [s.strip() for s in all_sents if len(s.strip()) > 5] # 过滤掉过短的句子，避免干扰序号统计
        sent_to_index = {s: i + 1 for i, s in enumerate(all_sents)} # sent_text -> 序号
        for m in matched:
            m["index"] = sent_to_index.get(m["content"], "?") # 如果没找到，标记为 "?"
        result.append({"para_id": pid, "title": para["title"], "content": para["content"], "matched_sents": matched})
    return result

def getCitationsFromContexts(contexts):
    citations = []
    for ctx in contexts:
        indices = ", ".join(str(s["index"]) for s in ctx["matched_sents"])
        citations.append(f"{ctx['title']}第 {indices} 条")
    return "，".join(citations)
# =========================
# 7. 构造 Prompt
# =========================

def build_prompt(contexts, question):
    # 先按照原始序号排序（先二再七）
    contexts.sort(key=lambda x: x["order"])
    
    context_sections = []
    citations_list = []
    
    for i, ctx in enumerate(contexts):
        ref_id = i + 1  # 生成 1, 2, 3... 的角标
        # 内容注入标题和角标
        section = f"--- [引用 {ref_id}] 出处：{ctx['title']} ---\n内容：{ctx['content']}"
        context_sections.append(section)
        
        # 记录对应的详细出处
        indices = ", ".join(str(s["index"]) for s in ctx["matched_sents"])
        citations_list.append(f"[{ref_id}] {ctx['title']}第 {indices} 条")
    
    context_text = "\n\n".join(context_sections)
    formatted_citations = "\n".join(citations_list)

    return f"""
你是一个严谨的助手。请严格基于提供的“内容”回答问题。

任务要求：
1. **核对事实**：如果内容中没有提到相关信息，直接回答“不知道”。
2. **使用角标**：在回答的每一句结论末尾，必须使用对应内容的角标（如 [1] 或 [1][2]）进行标注。
3. **结构化输出**：
   <你的回答内容>
   
   ---
   **参考出处：**
   {formatted_citations}

待分析内容：
{context_text}

用户问题：
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
    paraDict, sentenceDict, chunks = sub_sentence_chunking(text, client=client)

    # print("paraDict:", paraDict)
    # print("sentenceDict:", sentenceDict)

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
        
        contexts = getParagraphsFromChunks(searchChunks, paraDict, sentenceDict)

        citations = getCitationsFromContexts(contexts)

        # print(citations)

        # print("检索到的相关段落：")
        # for i, context in enumerate(contexts):
        #     print(context)
        
        prompt = build_prompt(contexts, question, citations)

        answer = ask_llm(prompt)

        print("\n回答: ")
        print(answer)

        print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    main()