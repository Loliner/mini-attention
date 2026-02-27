# 这是基于 v3 的优化版本，新增了：
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

torch.manual_seed(42)

# filename = 'article.txt' # 你可以替换成自己的文本文件路径
filename = './tiny_transformer/AndersenFairyTales.txt' # 你可以替换成自己的文本文件路径
# ===== 数据准备 =====

with open(filename, 'r', encoding='utf-8') as f:
    text = f.read()
    
chars = sorted(list(set(text))) # 去重排序
vocab_size = len(chars) # 总共有多少个字符
print("vocab_size", vocab_size)

# 生成[数字-字符]的字典，这里的数字表示的是 token id
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for ch,i in stoi.items()}
# print("stoi", stoi) # {' ': 0, 'd': 1, 'e': 2, 'h': 3, 'l': 4, 'o': 5, 'r': 6, 'w': 7}
# print("itos", itos) # {0: ' ', 1: 'd', 2: 'e', 3: 'h', 4: 'l', 5: 'o', 6: 'r', 7: 'w'}

tokens = [stoi[c] for c in text]
# print("tokens", tokens)
data = torch.tensor(tokens, dtype=torch.long)
# print("data", data) # tensor([3, 2, 4, 4, 5, 0, 7, 5, 6, 4, 1]) # "hello world"

# 超参数配置
block_size = 1024   # 每次看多长的上文
batch_size = 32   # 每次训练多少个片段
d_model = 512 # 每个 token 的向量维度
n_head = 8 # 注意力头数，越多模型越强大，但计算成本也越高
temperature = 0.8 # 控制生成文本的多样性，值越小生成的文本越确定（重复），值越大生成的文本越随机
layer = 8 # Block 层数
lr = 0.001 # 学习率
step_num = 4000 # 训练步数

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

class Block(nn.Module):
    """ 一个完整的 Transformer 块：通信 + 计算 """
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        # 1. 自注意力部分 (通信)
        self.attention = nn.MultiheadAttention(d_model, num_heads=n_head, batch_first=True)
        # 2. 前馈网络部分 (计算/消化信息)
        self.ffn = nn.Sequential(
            # 第一层：升维。扩大特征空间，让模型能学到更细致的模式。
            # W转换矩阵会被后续的训练更新
            nn.Linear(d_model, 4 * d_model),
            # 非线性激活：ReLU(x) = max(0, x)。
            # 就像大脑神经元的阈值，只有信号够强才传递，这让网络能拟合复杂的函数。
            nn.ReLU(),
            # 第二层：降维。把处理后的信息压缩回 d_model，准备进入下一个 Block。
            # W转换矩阵会被后续的训练更新
            nn.Linear(4 * d_model, d_model),
            # 在计算后加入随机失活，防止过拟合，增强模型的泛化能力
            nn.Dropout(dropout)
        )
        # 归一化层
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 这里的写法是 Pre-LN 结构，比 Post-LN 更容易训练深层网络
        # 残差连接 1: Attention
        x_norm = self.ln1(x)
        # 内部会执行（也就是 attention.py 中流程）：
        # Q = x @ Wq
        # K = x @ Wk
        # V = x @ Wv
        # scores = Q @ K.T
        # probs = softmax(scores)
        # output = probs @ V
        attn_out, _ = self.attention(x_norm, x_norm, x_norm, attn_mask=mask)
        x = x + self.attn_dropout(attn_out) # 残差连接，保持数值稳定性
        
        # 残差连接 2: FeedForward
        x = x + self.ffn(self.ln2(x))
        return x

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
        # self.attention = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.blocks = nn.ModuleList([Block(d_model, n_head=n_head) for _ in range(layer)]) # 堆叠 layer 个 Block
        
        # 定义层归一化，保持数值稳定性
        self.ln_f = nn.LayerNorm(d_model) # 最终归一化
        
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
         
        # 3. 通过 Block 层处理信息
        for block in self.blocks:
            x = block(x, mask=mask)
        
        x = self.ln_f(x)
        
        # 全连接层，将 d_model 维度的向量映射到 vocab_size 维度，得到每个 token 的 logits
        logits = self.fc(x) # (B, T, vocab_size)
        # print('After fc:', logits.shape)
        
        return logits
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, top_k=10):
        for _ in range(max_new_tokens):
            # 1. 从当前已经生成的全部序列中，只截取最后（最新）的 block_size 个词，保证输入长度不超过模型的上下文窗口大小
            idx_cond = idx[:, -block_size:]

            # 2. 前向传播得到 logits，并应用温度缩放
            logits = self(idx_cond) # (B, T, vocab_size)
            # print("logits.shape:", logits.shape) # (1, T, vocab_size)
            # 关注最后一个时间步的输出
            logits = logits[:, -1, :] / temperature
            
            # 3. 根据概率分布进行采样 TOPK，得到下一个 token 的索引
            if top_k is not None:
                # 找到 logits 中前 k 大的数值
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # 将小于第 k 名数值的所有位置设为负无穷，使其在 softmax 后概率为 0
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # 4. 计算概率分布并采样
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            
            # 5. 拼接多个张量
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
optimizer = torch.optim.Adam(model.parameters(), lr)
# 定义余弦退火调度器
# T_max: 学习率下降到最小值的总步数（通常设为总训练步数）
# eta_min: 学习率的最小值（通常设为一个很小的数，如 1e-5 或 0）
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5000, eta_min=1e-5)

model.train()
step_times = []
for step in range(step_num):
    step_start = time.time()

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
    scheduler.step() # 更新学习率

    step_elapsed = time.time() - step_start
    step_times.append(step_elapsed)
    avg_step_time = sum(step_times[-100:]) / len(step_times[-100:])
    remaining_steps = step_num - step - 1
    remaining_secs = avg_step_time * remaining_steps
    remaining_str = time.strftime("%H:%M:%S", time.gmtime(remaining_secs))

    # if step % 100 == 0:
    current_lr = optimizer.param_groups[0]['lr']
    print(f"step {step}/{step_num}, loss {loss.item():.4f}, lr {current_lr:.6f}, step_time {step_elapsed*1000:.1f}ms, remaining {remaining_str}")

# ===== 测试预测 =====
model.eval()
context = "你为什么睡在那儿？" # 给定半句话
x_input = torch.tensor([stoi[c] for c in context], dtype=torch.long).unsqueeze(0)
print("x_input:", x_input.shape, x_input) # tensor([[3, 2, 4, 4, 5, 0]]) # "hello "
generated_ids = model.generate(x_input, max_new_tokens=1000)
res = "".join([itos[int(i)] for i in generated_ids[0]])
print("\n模型生成的完整句子:")
print(res)