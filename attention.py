import numpy as np

np.random.seed(42)

def softmax(x):
    # 由于 f(x)=e^x 当 x 很大时，e^x 会非常大，可能导致数值溢出，所以减去最大值，防止数值过大
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def layer_norm(x):
    mean = np.mean(x, axis=-1, keepdims=True)  # 计算最后一个维度的均值，这里也就是每一行的均值
    std = np.std(x, axis=-1, keepdims=True)    # 计算最后一个维度的标准差，这里也就是每一行的标准差
    # x - mean: 使得均值为0。[1001, 1002, 1003] -> [-1, 0, 1]
    # std + 1e-6: 防止除零，归一化
    return (x - mean) / (std + 1e-6)           # 归一化，添加小常数防止除零

def attention(X, d_model, divide=True):
    print("=" * 50)
    print(f"d_model = {d_model}, divide = {divide}")
    
    print(f"Input X: {X}")
    X = layer_norm(X)
    print(f"After layer norm: {X}")
    
    # 随机初始化权重
    Wq = np.random.randn(d_model, d_model)
    Wk = np.random.randn(d_model, d_model)
    Wv = np.random.randn(d_model, d_model)
    
    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv
    print("Q shape:", Q.shape)
    scores = Q @ K.T
    
    print("\nQK^T raw scores:")
    print(scores)
    print("max score:", np.max(scores))
    
    if divide:
        scores = scores / np.sqrt(d_model)
        print("\nAfter dividing by sqrt(d):")
        print(scores)
        print("max score:", np.max(scores))
    
    # 将 scores 转换为概率分布
    probs = softmax(scores)
    
    print("\nSoftmax probabilities:")
    print(probs)
    print("max prob:", np.max(probs))
    
    output = probs @ V
    return output


# 创建一个简单输入（3个token）
def run_experiment(d_model, divide=True, small=True):
    if small:
        X = np.random.randn(3, d_model) * 10 # 让输入更大，观察数值稳定性
    else:
        X = np.random.randn(3, d_model)
    attention(X, d_model, divide)


# 实验1：小维度
# run_experiment(d_model=4, divide=True)

# 实验2：大维度
run_experiment(d_model=512, divide=True, small=True)
run_experiment(d_model=512, divide=True, small=False)
# run_experiment(d_model=2048, divide=True, small=False)

# 实验3：大维度 + 不除 sqrt(d)
# run_experiment(d_model=512, divide=False)