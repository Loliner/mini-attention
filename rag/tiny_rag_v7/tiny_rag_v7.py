# =========================
# 基于 v5 版本优化:
# 1. 引入 Cross-Encoder (Sentence-Transformers) 进行重排，大幅提升相关度过滤精度。
# 2. 优化编号逻辑：由 Python 预先分配静态角标 [1][2]，LLM 只负责在句末标注，不再承担重新排序的任务。
# 3. 简化 Prompt：去除复杂的动态重编指令，提高生成结果的稳定性。
# =========================
import os
# 限制线程数，防止信号量冲突
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import re
import uuid
import numpy as np
import faiss
import torch
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

# =========================
# 1. 初始化
# =========================
load_dotenv()
client = OpenAI()

# 基础模型配置
EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
# 引入本地重排模型 (建议使用 BGE 或混合模型，会自动下载)
device = "mps" if torch.backends.mps.is_available() else "cpu"
RERANK_MODEL = CrossEncoder('BAAI/bge-reranker-base', device=device)

# =========================
# 2. 文档加载
# =========================
def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def short_id():
    return uuid.uuid4().hex[:6]

def sub_sentence_chunking(text):
    # 按分隔符切分
    raw_paragraphs = re.split(r'--------------------------------', text)

    paraDict = {} # para_id -> {"content": ..., "title": ..., "order": ...}
    sentenceDict = {} # sent_id -> {"content": ..., "para_id": ...}
    chunks = []  # 句子文本列表

    for i, para in enumerate(raw_paragraphs):
        para_id = short_id()
        # 提取标题，例如 【二、年假政策】
        title_match = re.search(r'【.*?】', para)
        title = title_match.group() if title_match else f"段落 {i+1}"
        
        paraDict[para_id] = {"content": para.strip(), "title": title, "order": i}

        # 切分句子
        sentences = re.split(r'(?<=[。！？])\s*', para)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

        for sentence in sentences:
            sent_id = short_id()
            sentenceDict[sent_id] = {"content": sentence, "para_id": para_id}
            chunks.append(sentence)

    print(f"共切分出 {len(paraDict)} 个段落，{len(sentenceDict)} 个句子。")
    return paraDict, sentenceDict, chunks

# =========================
# 3. 向量检索 (粗排)
# =========================
def get_embedding(text):
    response = client.embeddings.create(model=EMBED_MODEL, input=text)
    return response.data[0].embedding

def build_faiss_index(chunks):
    print("正在生成 embedding...")
    vectors = [get_embedding(chunk) for chunk in chunks]
    vectors = np.array(vectors).astype("float32")

    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    return index

# =========================
# 4. 重排逻辑 (精排)
# =========================
def rerank_search(index, chunks, query, top_k=15, final_k=5):
    """
    双层检索优化：
    1. 使用 FAISS 粗排拉取 top_k 个候选子块。
    2. 使用 Cross-Encoder 对候选块进行精排。
    """
    q_vec = get_embedding(query)
    query_vec = np.array([q_vec]).astype("float32")
    D, I = index.search(query_vec, top_k)
    
    candidate_chunks = []
    for distance, idx in zip(D[0], I[0]):
        if distance < 1.6: # 保证相关性召回
            candidate_chunks.append(chunks[idx]) 
    
    # 构造重排对 [ (query, chunk1), (query, chunk2), ... ]
    pairs = [[query, chunk] for chunk in candidate_chunks]
    raw_scores = RERANK_MODEL.predict(pairs)
    
    T = 0.01 # 温度参数，控制 sigmoid 函数的平滑度
    # 对分数进行 Sigmoid 归一化，使其映射到 0-1 之间，增强可读性
    # 公式: 1 / (1 + exp(-x))
    sig_scores = 1 / (1 + np.exp(-raw_scores/T))
    
    print("scores:", sig_scores)
    # 按重排分数排序
    ranked_results = sorted(zip(candidate_chunks, sig_scores), key=lambda x: x[1], reverse=True)
    print("ranked_results:", ranked_results)
    
    # 过滤掉分数过低的项并截取前 final_k
    final_results = [res[0] for res in ranked_results if res[1] > 0.8] # 阈值 0 过滤不相关
    return final_results[:final_k]

# =========================
# 5. 上下文构建 (回溯段落)
# =========================
def get_ordered_contexts(searchChunks, paraDict, sentenceDict):
    # 建立句子内容 -> 在 searchChunks 中的最高排名（最小索引）的映射
    chunk_rank = {chunk: i for i, chunk in enumerate(searchChunks)}

    chunk_set = set(searchChunks)
    para_map = {}  # { para_id: [...content] }

    # sentenceDict: sent_id -> {"content": ..., "para_id": ...}
    for sent_info in sentenceDict.values():
        if sent_info["content"] in chunk_set:
            pid = sent_info["para_id"]
            if pid not in para_map:
                para_map[pid] = []
            para_map[pid].append(sent_info["content"])

    # 构建并排序
    result = []
    for pid, matched in para_map.items():
        para = paraDict[pid]
        # 获取句子序号
        all_sents = re.split(r'(?<=[。！？])\s*', para["content"])
        all_sents = [s.strip() for s in all_sents if len(s.strip()) > 5]
        sent_to_index = {s: i + 1 for i, s in enumerate(all_sents)}  # { content: index }

        matched_info = [{"content": m, "index": sent_to_index.get(m, "?")} for m in matched]

        # 该段落在 searchChunks 中最早出现的句子排名，决定段落顺序
        best_rank = min(chunk_rank[m] for m in matched)

        result.append({
            "title": para["title"],
            "content": para["content"],
            "order": para["order"],
            "matched_sents": matched_info,
            "best_rank": best_rank,
        })

    # 按 searchChunks 的相关度顺序排序（排名越小越靠前）
    result.sort(key=lambda x: x["best_rank"])
    return result

# =========================
# 6. 构建 Prompt
# =========================
def build_prompt(contexts, question):
    context_text_list = []
    citations_list = []
    
    for i, ctx in enumerate(contexts):
        ref_id = i + 1 # Python 端直接分配序号
        indices = ", ".join(str(s["index"]) for s in ctx["matched_sents"])
        
        # 建立内容与角标的强制绑定
        context_text_list.append(f"--- [引用 {ref_id}] --- \n{ctx['content']}")
        citations_list.append(f"[{ref_id}] {ctx['title']} 第 {indices} 条")
    
    context_text = "\n\n".join(context_text_list)
    citations_str = "\n".join(citations_list)
    print('contexts', contexts)
    print("context_text", context_text)
    print("citations_str", citations_str)

    return f"""
你是一个专业的助手。请仅基于以下“内容”回答问题。

### 规则：
1. **直接标注**：在你的每一句回答末尾，必须标注其内容来源的引用编号。
2. **多重引用**：如果一句话结合了多个来源，请同时标注，如 [1][2]。
3. **引用说明**：在末尾表明出处来源。
4. **不知道**：如果内容未提及，请回答“不知道”。

### 内容：
{context_text}

### 参考出处：
{citations_str}

### 用户问题：
{question}

### 回答范例
员工每月享有 50 澳元餐补，并提供年度体检福利 [1]。
此外，晋升成功后年假会增加 [2]。

参考出处：
[1] 【八、福利补贴】第1,2条
[2] 【七、晋升与考核机制】第3条
"""

# =========================
# 7. 调用 LLM
# =========================
def ask_llm(prompt):
    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0, # 设为 0 以保证输出格式的稳定性
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# =========================
# 9. 主流程
# =========================
def main():
    text = load_text("./rag/knowledge_base_entangled.txt")
    paraDict, sentenceDict, chunks = sub_sentence_chunking(text)
    index = build_faiss_index(chunks)

    print("\nRAG 系统已启动 (Cross-Encoder 重排版)！\n")

    while True:
        question = input("你的问题：")
        if question.lower() == "exit": break

        # 调用精排搜索
        refined_chunks = rerank_search(index, chunks, question)
        contexts = get_ordered_contexts(refined_chunks, paraDict, sentenceDict)
        
        if not contexts:
            print("\n回答: 不知道。\n")
            continue

        prompt = build_prompt(contexts, question)
        answer = ask_llm(prompt)

        print(f"\n回答:\n{answer}")
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()