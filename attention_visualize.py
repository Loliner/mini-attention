import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertModel
import numpy as np

# ====== 1. 输入句子 ======
sentence = "I love this movie because it is amazing"

# ====== 2. 加载模型 ======
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# print('tokenizer:', tokenizer) # BertTokenizer(name_or_path='bert-base-uncased', vocab_size=30522, model_max_length=512, is_fast=True, padding_side='right', truncation_side='right')
model = BertModel.from_pretrained(
    "bert-base-uncased",
    output_attentions=True
)
# print('model:', model)

model.eval()

# ====== 3. 编码输入 ======
inputs = tokenizer(sentence, return_tensors="pt")
print('inputs:', inputs)

# ====== 4. 前向传播 ======
with torch.no_grad():
    outputs = model(**inputs)

attentions = outputs.attentions

# ====== 5. 选择 layer 和 head ======
layer = 8
head = 3

attention = attentions[layer][0][head].numpy()
print('attention:', attention.shape, attention) # attention shape: (10, 10) -> 10个token之间的注意力权重矩阵

# ====== 6. 取 token ======
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
print('tokens:', tokens) # tokens: ['[CLS]', 'i', 'love', 'this', 'movie', 'because', 'it', 'is', 'amazing', '[SEP]']

# ====== 7. 画热力图（带数值） ======
plt.figure(figsize=(10, 8))

sns.heatmap(
    attention,
    xticklabels=tokens,
    yticklabels=tokens,
    cmap="viridis",
    annot=True,          # 显示数值
    fmt=".2f",           # 保留两位小数
    annot_kws={"size":8} # 数字大小
)

plt.title(f"Layer {layer+1} Head {head+1} Attention")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()