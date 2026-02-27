# Mini Attention

A hands-on project for learning Transformer attention mechanisms, built progressively from NumPy to a GPT-style model.

## Files

| File | Description |
|---|---|
| `attention.py` | Scaled dot-product attention from scratch using NumPy |
| `attention_visualize.py` | BERT attention heatmap visualization |
| `predict_transformer.py` | TinyGPT with causal masking and text generation |
| `tiny_transformer/tiny_transformer.py` | v1: Minimal transformer trained on "hello world" |
| `tiny_transformer/tiny_transformer_v2.py` | v2: Adds multi-head attention, causal mask, residual connections |
| `tiny_transformer/tiny_transformer_v3.py` | v3: Full GPT-style with stacked blocks, FFN, dropout, top-k sampling |

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install torch numpy matplotlib seaborn transformers
```

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT)
