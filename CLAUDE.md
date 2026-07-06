# Mini-Attention

An educational project for learning Transformer architecture, RAG systems, and AI agents from first principles. Each component has multiple versions showing incremental improvements.

## Project Structure

```
mini-attention/
├── attention.py                 # NumPy-based scaled dot-product attention
├── attention_visualize.py       # BERT attention weight visualization
├── agent/
│   ├── cal_agent_v1.py          # ReAct agent with regex parsing
│   └── cal_agent_v2.py          # Async agent with OpenAI function calling
├── rag/
│   ├── knowledge_base.txt
│   ├── tiny_rag_v1.py → v6.py   # RAG evolution: vector search → reranking
│   └── tiny_rag_v7/
│       └── tiny_rag_v7.py       # Latest: ChromaDB persistence + reranking
└── tiny_transformer/
    ├── tiny_transformer.py       # v1: minimal single-head transformer
    ├── tiny_transformer_v2.py    # v2: multi-head + causal masking
    ├── tiny_transformer_v3.py    # v3: stacked blocks + FFN + dropout + top-k
    └── tiny_transformer_v4.py    # v4: d_model=512, 8 heads, 8 layers, ctx=1024
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install torch numpy matplotlib seaborn transformers faiss-cpu chromadb sentence-transformers openai python-dotenv
```

Set `OPENAI_API_KEY` in a `.env` file at the project root (used by RAG and agent modules).

## Running Components

```bash
# Attention demo (no API key needed)
python attention.py

# BERT attention visualization
python attention_visualize.py

# Train transformer on fairy tales and generate text
python tiny_transformer/tiny_transformer_v4.py

# RAG with ChromaDB + cross-encoder reranking
python rag/tiny_rag_v7/tiny_rag_v7.py

# AI agent with parallel async tool calls
python agent/cal_agent_v2.py
```

## Architecture Notes

### Transformer Progression
- v1: Single-head attention, "hello world" training
- v2: Multi-head attention, causal masking, residual connections, layer norm
- v3: Stacked blocks (4 layers), FFN (4x expansion), dropout, top-k sampling, cosine LR
- v4: Larger capacity — d_model=512, 8 heads, 8 layers, 1024 context window

### RAG Progression
- v1–v5: OpenAI embeddings + FAISS similarity search
- v6: Adds cross-encoder reranking (BAAI/bge-reranker-base)
- v7: ChromaDB for persistent vector storage, lazy embedding on first run

### Agent Progression
- v1: ReAct pattern, regex parses `Action:` / `Observation:` from LLM output
- v2: OpenAI function calling API, async/await, parallel tool execution via `asyncio.gather()`
- v3: Short-term memory (session cache) + multi-turn conversation
- v4: Long-term memory via ChromaDB (cross-session retrieval)
- v5: Plan-Execute architecture — LLM plans all steps upfront, reduces LLM calls
- v6: Re-Plan support — triggers replanning when intermediate results require it

## Curriculum

Learning priority: **Agent → Engineering & Production → RAG**

### Agent（当前重点）
- [x] v1: ReAct + regex parsing
- [x] v2: Function calling + async parallel tools
- [x] v3: Short-term memory + multi-turn
- [x] v4: Long-term memory (ChromaDB)
- [x] v5: Plan-Execute
- [x] v6: Re-Plan for conditional tasks
- [ ] v7: Multi-Agent — Orchestrator routes subtasks to specialized Sub-Agents, parallel execution, result aggregation
- [ ] v8: Reliability — automated eval (input → expected output → actual comparison), self-reflection & retry
- [ ] v9: Real tools — Web search (Tavily/SerpAPI), sandboxed code execution

### Engineering & Production
- [ ] Structured output — Pydantic schema + `instructor` library for typed LLM responses
- [ ] Streaming — SSE streaming output, real-time token display
- [ ] Prompt caching — OpenAI/Anthropic cache mechanism to reduce token costs
- [ ] Observability — LangFuse tracing, per-call token stats, latency tracking
- [ ] Error handling — production-grade timeout, fallback, rate limit handling

### RAG（后续）
- [ ] Hybrid search — BM25 sparse + dense vector, RRF fusion ranking
- [ ] Query rewriting — HyDE (Hypothetical Document Embeddings), Multi-Query expansion
- [ ] RAG evaluation — RAGAS framework (faithfulness, answer relevance, recall)

## Tech Stack

- **PyTorch** — transformer models
- **NumPy** — low-level attention demos
- **OpenAI API** — GPT-4o, text-embedding-3-small
- **Hugging Face Transformers** — BERT, cross-encoder reranker
- **FAISS** — vector similarity search
- **ChromaDB** — persistent vector database
- **asyncio** — parallel agent tool execution
- Apple Silicon (MPS) detected automatically, falls back to CPU
