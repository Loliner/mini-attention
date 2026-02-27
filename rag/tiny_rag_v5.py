# =========================
# 基于 v4 版本优化:
# 1. 增加动态角标。
# 注意：
# 我们给LLM提供了段落与句子的引用，但由于提供给LLM的内容有可能是多余的，所以提供出来的引用角标顺序并不可以直接依赖。
# 所以最终是依靠 LLM 根据提供的引用角标，自己重新定义一个顺序。
# 这是不推荐的做法，理想的状态是加入重排(Reranking Step)，将 search 完后的内容再过滤一遍，尽量保证喂给 LLM 输入 100% 包含答案。
# =========================
import os
import re
import uuid
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
# 3. 切分文本 (增加 order 记录)
# =========================
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
# 4. 生成 embedding
# =========================
def get_embedding(text):
    response = client.embeddings.create(model=EMBED_MODEL, input=text)
    return response.data[0].embedding

# =========================
# 5. 构建向量数据库
# =========================
def build_faiss_index(chunks):
    print("正在生成 embedding...")
    vectors = [get_embedding(chunk) for chunk in chunks]
    vectors = np.array(vectors).astype("float32")

    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    return index

# =========================
# 6. 检索与回溯 (整合排序逻辑)
# =========================
def search(index, chunks, query, top_k=5):
    q_vec = get_embedding(query)
    query_vec = np.array([q_vec]).astype("float32")
    D, I = index.search(query_vec, top_k)

    results = []
    for distance, idx in zip(D[0], I[0]):
        if distance < 1.6: # 适当放宽以保证相关性召回
            results.append(chunks[idx]) 
    return results

def get_ordered_contexts(searchChunks, paraDict, sentenceDict):
    # 找到命中的句子所属的段落
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
        sent_to_index = {s: i + 1 for i, s in enumerate(all_sents)} # { [content]: index }
        
        matched_info = [{"content": m, "index": sent_to_index.get(m, "?")} for m in matched] # [{content: ..., index: ...}]
        
        result.append({
            "title": para["title"],
            "content": para["content"],
            "order": para["order"],
            "matched_sents": matched_info
        })

    print("result", result)
    # 核心：按原始文档顺序排序（先二再七）
    result.sort(key=lambda x: x["order"])
    return result

# =========================
# 4. 构建动态角标 Prompt
# =========================
def build_prompt(contexts, question):
    context_text_list = []
    for ctx in contexts:
        # 依然使用标题作为唯一标识
        indices = ", ".join(str(s["index"]) for s in ctx["matched_sents"])
        context_text_list.append(f"{ctx['title']} (第 {indices} 条)\n内容：{ctx['content']}")
    
    context_text = "\n\n".join(context_text_list)

    return f"""
你是一个专业的公司政策助手。请基于提供的“内容列表”回答用户问题。

### 强制执行要求：
1. **事实核对**：如果内容中没有答案，直接回答“不知道”。
2. **资源级角标合并 (核心)**：
   - 每一个《资源标签》在你的回答中对应且仅对应一个唯一的角标。
   - **重要**：如果你的回答引用了同一个《资源标签》里的多条信息，这些信息必须使用**同一个角标**。
   - 角标必须根据资源在回答中出现的先后顺序，从 [1] 开始重新连续编号。
3. **出处对照**：在回答结束后的“参考出处”栏下，按照 [1], [2]... 的顺序，列出对应的“资源标签”。

### 内容列表：
{context_text}

### 用户问题：
{question}

---
回答范例：
员工每月享有 50 澳元餐补，并提供年度体检福利 [1]。此外，晋升成功后年假会增加 [2]。

参考出处：
[1] 【八、福利补贴】第1,2条
[2] 【七、晋升与考核机制】第3条
"""

# =========================
# 8. 调用 LLM
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
    print("\nRAG 系统已启动！输入 exit 退出。\n")

    while True:
        question = input("你的问题：")
        if question.lower() == "exit": break

        # 1. 搜索子块
        search_results = search(index, chunks, question, top_k=5)
        
        # 2. 获取并排序父块
        contexts = get_ordered_contexts(search_results, paraDict, sentenceDict)
        
        if not contexts:
            print("\n回答: 不知道。\n")
            continue

        # 3. 生成 Prompt 并提问
        prompt = build_prompt(contexts, question)
        answer = ask_llm(prompt)

        print(f"\n回答:\n{answer}")
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()