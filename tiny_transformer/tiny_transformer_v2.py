# 这是基于 v1 的优化版本，新增了：
# 1. 支持大文本读取，支持 block_size(每次看多长的上文) 和 batch_size(每次训练多少个片段)
# 2. 支持多头注意力机制，增加了掩码
# 3. 加上了残差网络和层归一化，保持数值稳定性
# 4. 加上了 temperature ，控制生成文本的多样性
# 5. 支持给定前半句，生成后续文本(100 个 token)
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# ===== 数据准备 =====
try:
    with open('article.txt', 'r', encoding='utf-8') as f:
        text = f.read()
except FileNotFoundError:
    print("没找到文件，请确保 article.txt 在正确的位置")
    text = "hello world " * 1000 # 备用方案

chars = sorted(list(set(text))) # 去重排序
vocab_size = len(chars) # 总共有多少个字符
# print("chars", chars, "vocab_size", vocab_size)

# 生成[数字-字符]的字典，这里的数字表示的是 token id
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for ch,i in stoi.items()}
# print("stoi", stoi) # {' ': 0, 'd': 1, 'e': 2, 'h': 3, 'l': 4, 'o': 5, 'r': 6, 'w': 7}
# print("itos", itos) # {0: ' ', 1: 'd', 2: 'e', 3: 'h', 4: 'l', 5: 'o', 6: 'r', 7: 'w'}

tokens = [stoi[c] for c in text]
# print("tokens", tokens)
data = torch.tensor(tokens, dtype=torch.long)
print("data", data) # tensor([3, 2, 4, 4, 5, 0, 7, 5, 6, 4, 1]) # "hello world"

# 超参数配置
block_size = 32   # 每次看多长的上文
batch_size = 16   # 每次训练多少个片段
d_model = 64
n_embd = 64
temperature = 0.8

# 数据采样函数：随机抓取数据片段
def get_batch(data, block_size, batch_size):
    # 随机生成 batch_size 个起始索引
    # print("ix:", ix) # tensor([12345, 67890, ...]) 每次运行都会不同
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # print("x:", x) # tensor([[3, 2, 4, ..., 5, 0, 7], [2, 4, 4, ..., 6, 4, 1], ...])
    # shape: (batch_size, block_size)
    x = torch.stack([data[i:i+block_size] for i in ix])
    # print("y:", y) # tensor([[2, 4, 4, ..., 0, 7, 1], [4, 4, 5, ..., 4, 1, 3], ...])
    # shape: (batch_size, block_size)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

# ===== 模型定义 =====
class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=16):
        super().__init__()
        
        # token embedding table (vocab_size, d_model), 用于将 token id 转换为向量
        # 得到的是(vocab_size, d_model)的矩阵，每一行是一个 token 的向量表示，数值是随机初始化的
        # 该矩阵会在后续训练时被更新
        self.embedding = nn.Embedding(vocab_size, d_model) 
        # print("embedding:", self.embedding.weight.mean(), self.embedding.weight.std())
        # self.pos_embedding = nn.Parameter(torch.zeros(100, d_model))
        self.pos_embedding = nn.Embedding(block_size, d_model) # 位置嵌入表，最多支持 100 个位置
        # print("pos_embedding:", self.pos_embedding)
        self.attention = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        
        # 定义层归一化，保持数值稳定性
        self.ln1 = nn.LayerNorm(d_model)
        
        # y = xW.T + b
        self.fc = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        # x 输入的是总共有 batch_size 段文本，每段文本长 block_size，数值是 token id
        B, T = x.shape # (batch_size, block_size) = (16, 32)
        # print(f"T: {T}")
        
        # token id -> 向量
        # [hello, world]
        # -> [[h,e,l,l,o], [w,o,r,l,d]]
        # -> 
        # [ batch_size, 有多少段文本
        #   [ block_size, 每段文本有多少个 token
        #       [0.1, 0.2, ...], 每个 token 对应的向量表示，维度是 d_model
        #       [0.3, 0.4, ...],
        #       ...
        #   ],
        #   [[0.5, 0.6, ...], [0.7, 0.8, ...], ...]
        # ]
        x = self.embedding(x)  # (B, T) -> (B, T, d)
        # print("embedding:", x)
        
        # 初始有 100 个位置，我们只用了 T 个位置，所以取前 T 个位置的 pos embedding
        pos = self.pos_embedding(torch.arange(T)) # (T, d_model)
        # print("pos embedding:", pos)
        x = x + pos
        # print("After pos embedding:", x)
        
        # 2. 生成因果掩码 (Causal Mask)
        # 形状为 (T, T)，上三角全为 True，防止看到未来
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(x.device)
        # print("mask:", mask)
         
        # 内部会执行（也就是 attention.py 中流程）：
        # Q = x @ Wq
        # K = x @ Wk
        # V = x @ Wv
        # scores = Q @ K.T
        # probs = softmax(scores)
        # output = probs @ V
        attn_output, _ = self.attention(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask) # (B, T, d)
        attn_output = x + attn_output # 残差连接，保持数值稳定性
        # print('After attention:', attn_output.shape)
        
        # 全连接层，将 d_model 维度的向量映射到 vocab_size 维度，得到每个 token 的 logits
        logits = self.fc(attn_output) # (B, T, vocab_size)
        # print('After fc:', logits.shape)
        
        return logits
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # 从当前已经生成的全部序列中，只截取最后（最新）的 block_size 个词，保证输入长度不超过模型的上下文窗口大小
            idx_cond = idx[:, -block_size:]
            logits = self(idx_cond) # (B, T, vocab_size)
            # print("logits.shape:", logits.shape) # (1, T, vocab_size)
            # 关注最后一个时间步的输出，得到下一个 token 的概率分布
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            # 根据概率分布进行采样，得到下一个 token 的索引
            # 与 argmax 不同，非 TOP1 也有概率被选中，这样生成的文本会更丰富多样
            next_idx = torch.multinomial(probs, num_samples=1)
            # 拼接多个张量
            # dim=0: 纵向拼接（增加行数）
            # dim=1: 横向拼接（增加列数）
            idx = torch.cat((idx, next_idx), dim=1)
        # idx.shape 是 (B, T + max_new_tokens)，返回完整的生成序列
        return idx

# ===== 训练 =====
model = TinyTransformer(vocab_size, d_model)
# print("Initial model parameters:")
# for name, param in model.named_parameters():
#     print(f"  {name}: {param.shape}")
# print("model parameters:", model.parameters())
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for step in range(20000):
    x, y = get_batch(data, block_size, batch_size)
    logits = model(x) # (B, T, vocab_size)
    # 计算 loss
    # 这里需要将 logits 展平为 (B*T, vocab_size)，y 展平为 (B*T)
    logits = logits.view(-1, vocab_size)
    y = y.view(-1)
    loss = F.cross_entropy(logits, y) # loss = -log(p(y|x)) = -log(softmax(logits)[y])
    # print("logits:", logits, "y:", y, "loss:", loss.item())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if step % 1000 == 0:
        print(f"step {step}, loss {loss.item():.4f}")

# ===== 测试预测 =====
model.eval()
context = "I know" # 给定半句话
x_input = torch.tensor([stoi[c] for c in context], dtype=torch.long).unsqueeze(0)
print("x_input:", x_input.shape, x_input) # tensor([[3, 2, 4, 4, 5, 0]]) # "hello "
generated_ids = model.generate(x_input, max_new_tokens=100)
res = "".join([itos[int(i)] for i in generated_ids[0]])
print("\n模型生成的完整句子:")
print(res)