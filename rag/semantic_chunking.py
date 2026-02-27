import re
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
client = OpenAI()

def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
    
def semantic_chunking(text, client, model="text-embedding-3-small", breakpoint_percentile=50):
    # 1. 粗暴但有效的按句切分
    sentences = re.split(r'--------------------------------', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    print(f"共切分出 {len(sentences)} 个句子。")
    for i, sentence in enumerate(sentences):
        print(f"--- Sentence {i+1} ---")
        print(sentence)
    
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

text = load_text("./rag/doc.txt")
chunks = semantic_chunking(text, client=client)
print(f"切分成 {len(chunks)} 个语义块：")

for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ---")
    print(chunk)