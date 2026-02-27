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
    
def sub_sentence_chunking(text, client, model="text-embedding-3-small", breakpoint_percentile=50):

    paragraphs = re.split(r'--------------------------------', text)
    paraDict = {i: para for i, para in enumerate(paragraphs)}
    paraToSentences = {}
    allSentences = []
    for i, para in enumerate(paragraphs):
        sentences = re.split(r'(?<=[。！？])\s*', para)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        
        sentenceArr = paraToSentences.get(i)
        if sentenceArr is None:
            sentenceArr = []

        for j, sentence in enumerate(sentences):
            sentenceArr.append(sentence)
            allSentences.append(sentence)


    print(f"共切分出 {len(paragraphs)} 个段落。")
    print(f"共切分出 {len(allSentences)} 个句子。")

    
    return allSentences

text = load_text("./rag/knowledge_base_entangled.txt")
chunks = sub_sentence_chunking(text, client=client)
# print(f"切分成 {len(chunks)} 个语义块：")

# for i, chunk in enumerate(chunks):
#     print(f"--- Chunk {i+1} ---")
#     print(chunk)