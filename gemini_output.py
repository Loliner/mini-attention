import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== 1. 数据准备与预处理 =====
try:
    with open('article.txt', 'r', encoding='utf-8') as f:
        text = f.read()
except FileNotFoundError:
    print("没找到文件，请确保 article.txt 在正确的位置")
    text = "hello world " * 1000

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for ch,i in stoi.items()}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

# 超参数配置
block_size = 32   # 每次看多长的上文
batch_size = 16   # 每次训练多少个片段
d_model = 64
n_embd = 64

# 数据采样函数：随机抓取数据片段
def get_batch(data, block_size, batch_size):
    # 随机生成 batch_size 个起始索引
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

# ===== 2. 模型定义 =====
class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        # num_heads=2 可以让模型同时关注两个不同的特征维度
        self.attention = nn.MultiheadAttention(d_model, num_heads=2, batch_first=True)
        self.fc = nn.Linear(d_model, vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # 1. 词向量 + 位置向量
        tok_emb = self.token_embedding(idx) # (B, T, d_model)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device)) # (T, d_model)
        x = tok_emb + pos_emb # (B, T, d_model)
        
        # 2. 生成因果掩码 (Causal Mask)
        # 形状为 (T, T)，上三角全为 -inf，防止看到未来
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(idx.device)
        
        # 3. 注意力层
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        
        # 4. 输出层
        logits = self.fc(attn_output) # (B, T, vocab_size)
        
        loss = None
        if targets is not None:
            # CrossEntropy 需要 (N, C) 形状，所以要展平
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
            
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # 从当前已经生成的全部序列中，只截取最后（最新）的 block_size 个词，保证输入长度不超过模型的上下文窗口大小
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            # 关注最后一个时间步的输出，得到下一个 token 的概率分布
            probs = F.softmax(logits[:, -1, :], dim=-1)
            # 根据概率分布进行采样，得到下一个 token 的索引
            # 与 argmax 不同，非 TOP1 也有概率被选中，这样生成的文本会更丰富多样
            next_idx = torch.multinomial(probs, num_samples=1)
            # 拼接多个张量
            # dim=0: 纵向拼接（增加行数）
            # dim=1: 横向拼接（增加列数）
            idx = torch.cat((idx, next_idx), dim=1)
        # idx.shape 是 (B, T + max_new_tokens)，返回完整的生成序列
        return idx

# ===== 3. 训练与预测 =====
model = TinyTransformer(vocab_size, d_model)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 简易训练循环
for step in range(50000):
    xb, yb = get_batch(data, block_size, batch_size)
    logits, loss = model(xb, yb)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if step % 500 == 0:
        print(f"步数 {step}, 损失值 {loss.item():.4f}")

# ===== 4. 预测后续 20 个 Token =====
model.eval()
context = "The transformer" # 给定半句话
# 转换为 tensor 并增加 batch 维度 (1, T)
x_input = torch.tensor([stoi[c] for c in context], dtype=torch.long).unsqueeze(0)
generated_ids = model.generate(x_input, max_new_tokens=20)
res = "".join([itos[int(i)] for i in generated_ids[0]])

print("\n模型生成的完整句子:")
print(res)