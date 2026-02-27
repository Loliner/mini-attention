# 这是一个非常简化的 Transformer 模型实现
# 主要流程包含
# 1. 数据准备：把文本转换成数字（token id）
# 2. 模型定义：包含 embedding、positional embedding、attention 和fc全连接层
# 3. 训练：使用交叉熵损失函数，优化模型参数
# 4. 测试预测：给定一个输入（遍历 hello worl），看看模型预测下一个是否准确（输出 ello world）
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# ===== 数据准备 =====
text = "hello world"
chars = sorted(list(set(text))) # 去重排序
vocab_size = len(chars) # 总共有多少个词
# print("chars", chars, "vocab_size", vocab_size)

# 生成[数字-字符]的字典，这里的数字表示的是 token id
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for ch,i in stoi.items()}
print("stoi", stoi) # {' ': 0, 'd': 1, 'e': 2, 'h': 3, 'l': 4, 'o': 5, 'r': 6, 'w': 7}
print("itos", itos) # {0: ' ', 1: 'd', 2: 'e', 3: 'h', 4: 'l', 5: 'o', 6: 'r', 7: 'w'}

tokens = [stoi[c] for c in text]
# print("tokens", tokens)
data = torch.tensor(tokens, dtype=torch.long)
print("data", data) # tensor([3, 2, 4, 4, 5, 0, 7, 5, 6, 4, 1]) # "hello world"

# 输入是前面的字符，目标是下一个字符
x = data[:-1]
y = data[1:]
print("x", x) # tensor([3, 2, 4, 4, 5, 0, 7, 5, 6, 4]) # "hello worl"
print("y", y) # tensor([2, 4, 4, 5, 0, 7, 5, 6, 4, 1]) # "ello world"

# ===== 模型定义 =====
class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=16):
        super().__init__()
        
        # token embedding table (vocab_size, d_model), 用于将 token id 转换为向量
        # 得到的是(vocab_size, d_model)的矩阵，每一行是一个 token 的向量表示，数值是随机初始化的
        # 该矩阵会在后续训练时被更新
        self.embedding = nn.Embedding(vocab_size, d_model) 
        # print("embedding:", self.embedding.weight.mean(), self.embedding.weight.std())
        self.pos_embedding = nn.Parameter(torch.zeros(100, d_model))
        # print("pos_embedding:", self.pos_embedding)
        self.attention = nn.MultiheadAttention(d_model, num_heads=1, batch_first=True)
        
        # y = xW.T + b
        self.fc = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        B = 1
        T = x.size(0) # 这里的 T 是输入序列的长度，也就是我们输入了多少个 token，比如 "hello worl" 就是 10 个 token，所以 T=10
        # print(f"T: {T}")
        
        # token id -> 向量
        # 如 hello -> [h, e, l, l, o] -> [[0.1, 0.2, ...], [0.3, 0.4, ...], ...]
        x = self.embedding(x)  # (T) -> (T, d)
        # print("embedding:", x)
        
        # 初始有 100 个位置，我们只用了 T 个位置，所以取前 T 个位置的 pos embedding
        pos = self.pos_embedding[:T] # (T, d)
        # print("pos embedding:", pos)
        x = x + pos
        # print("After pos embedding:", x)
         
        x = x.unsqueeze(0)  # (1, T, d)
        # print('After unsqueeze:', x.shape)
        
        # 内部会执行（也就是 attention.py 中流程）：
        # Q = x @ Wq
        # K = x @ Wk
        # V = x @ Wv
        # scores = Q @ K.T
        # probs = softmax(scores)
        # output = probs @ V
        attn_output, _ = self.attention(x, x, x) # (1, T, d)
        # print('After attention:', attn_output.shape)
        
        logits = self.fc(attn_output) # (1, T, vocab_size)
        # print('After fc:', logits.shape)
        
        return logits.squeeze(0) # (T, vocab_size)

# ===== 训练 =====
model = TinyTransformer(vocab_size)
# print("Initial model parameters:")
# for name, param in model.named_parameters():
#     print(f"  {name}: {param.shape}")
# print("model parameters:", model.parameters())
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for step in range(500):
    logits = model(x)
    loss = F.cross_entropy(logits, y) # loss = -log(p(y|x)) = -log(softmax(logits)[y])
    # print("logits:", logits, "y:", y, "loss:", loss.item())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if step % 50 == 0:
        print(f"step {step}, loss {loss.item():.4f}")

# ===== 测试预测 =====
model.eval()
with torch.no_grad():
    logits = model(x)
    probs = F.softmax(logits, dim=-1)
    
    print("\nPredictions:")
    for i in range(len(x)):
        # print("probs:", probs[i])
        # print("max prob:", torch.max(probs[i]))
        predicted = torch.argmax(probs[i]).item()
        print(text[i], "->", itos[predicted])