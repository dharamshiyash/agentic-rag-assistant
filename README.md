# 🧠 Agentic RAG Assistant

A production-quality **Retrieval-Augmented Generation (RAG)** AI assistant that answers questions strictly from a curated knowledge base. Built with LangChain, Groq, FAISS, and Streamlit — designed to demonstrate modern RAG architecture with anti-hallucination guardrails.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3+-1C3C3C?style=flat&logo=chainlink&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-LLaMA_3.1-F55036?style=flat)
![FAISS](https://img.shields.io/badge/FAISS-Vector_DB-0467DF?style=flat)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔍 **MMR Retrieval** | Maximal Marginal Relevance search for diverse, high-quality document retrieval |
| 🧠 **Conversation Memory** | Sliding window memory enables follow-up questions and multi-turn conversations |
| ⚡ **Streaming Responses** | Token-by-token streaming for a modern chatbot experience |
| 📎 **Source Attribution** | Every answer includes document name, section, and chunk references |
| 💡 **Suggested Questions** | Three relevant follow-up questions after every response |
| 🛡️ **Anti-Hallucination** | Strict context-only answering with explicit uncertainty acknowledgment |
| 🚀 **Smart Caching** | Hash-based cache detects document changes and skips unnecessary embedding regeneration |
| 📊 **Rich Metadata** | Every chunk stores document name, topic, section, and chunk ID |
| 🎨 **Professional UI** | Dark-themed Streamlit interface with source cards, chat bubbles, and sidebar controls |
| 📄 **Document Upload** | Upload new .txt documents directly through the sidebar |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface                          │
│                  (Streamlit — ui/app.py)                        │
│   Chat Input → Streaming Display → Source Cards → Suggestions  │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                        RAG Agent                               │
│                   (src/agent.py)                                │
│   Question + Memory → Retrieve → Format → LLM → Parse Output  │
└──────┬──────────────┬──────────────┬────────────────────────────┘
       │              │              │
┌──────▼──────┐ ┌─────▼──────┐ ┌────▼────────────────┐
│  Retriever  │ │  Prompts   │ │     LLM (Groq)      │
│ (MMR/FAISS) │ │ (System +  │ │  LLaMA 3.1 8B       │
│ src/        │ │  Memory)   │ │  Temperature: 0      │
│ retriever.py│ │ src/       │ │                      │
│             │ │ prompts.py │ │                      │
└──────┬──────┘ └────────────┘ └──────────────────────┘
       │
┌──────▼──────────────────────────────────────────────┐
│              Vector Store (FAISS)                    │
│             src/embeddings.py                        │
│  Embeddings: all-MiniLM-L6-v2  │  Cache: Hash-based │
└──────┬──────────────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────────────┐
│            Document Processing                       │
│              src/loader.py                           │
│  Load .txt → Chunk (1500/300) → Enrich Metadata     │
└──────┬──────────────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────────────┐
│            Knowledge Base (data/)                    │
│  6 Documents × ~4000 words each                     │
│  AI │ Biotech │ Climate │ Quantum │ Space │ Energy  │
└─────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Framework** | LangChain 0.3+ | RAG pipeline orchestration |
| **LLM** | Groq (LLaMA 3.1 8B Instant) | Fast, free inference |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 | Semantic similarity |
| **Vector Store** | FAISS | Efficient similarity search |
| **UI** | Streamlit | Interactive web interface |
| **Language** | Python 3.9+ | Core implementation |

---

## 📁 Folder Structure

```
rag-agentic-assistant/
├── data/                          # Knowledge base documents
│   ├── artificial_intelligence.txt
│   ├── biotechnology.txt
│   ├── climate_science.txt
│   ├── quantum_computing.txt
│   ├── space_exploration.txt
│   └── sustainable_energy.txt
├── src/                           # Core source code
│   ├── agent.py                   # RAG Agent (memory, streaming, citations)
│   ├── embeddings.py              # Vector store management + caching
│   ├── loader.py                  # Document loading + chunking + metadata
│   ├── prompts.py                 # System prompt templates
│   ├── retriever.py               # MMR retrieval + source extraction
│   └── utils.py                   # Logging, config, helpers
├── ui/                            # User interface
│   └── app.py                     # Streamlit application
├── faiss_index/                   # Generated vector store (gitignored)
├── main.py                        # Application entry point
├── requirements.txt               # Python dependencies
├── CHANGELOG.md                   # Detailed change history
├── .env                           # API keys (gitignored)
├── .gitignore
└── README.md
```

---

## 🚀 Installation

### Prerequisites

- Python 3.9+
- A free [Groq API Key](https://console.groq.com/keys)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/dharamshiyash/agentic-rag-assistant.git
   cd agentic-rag-assistant
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure your API key**
   Create a `.env` file in the root directory:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

5. **Run the application**
   ```bash
   python main.py
   ```
   Or directly:
   ```bash
   streamlit run ui/app.py
   ```

The first launch will generate embeddings from the knowledge base documents (~30 seconds). Subsequent launches reuse cached embeddings unless documents change.

---

## 💬 Usage

### In-Scope Questions (will be answered from knowledge base)
- "What is deep learning and how does it work?"
- "Explain CRISPR gene editing technology"
- "What are the main causes of climate change?"
- "How do quantum computers differ from classical computers?"
- "Tell me about the James Webb Space Telescope"
- "Compare solar and wind energy"

### Follow-Up Questions (uses conversation memory)
- "What about healthcare applications?"
- "Explain that in more detail"
- "Compare it with the previous topic"

### Out-of-Scope Questions (will be politely declined)
- "Who is the president of the USA?"
- "Write me a Python script"
- "What's the weather today?"

---

## ⚙️ Configuration

Key parameters can be adjusted in the source files:

| Parameter | File | Default | Description |
|-----------|------|---------|-------------|
| `CHUNK_SIZE` | `src/loader.py` | 1500 | Maximum characters per chunk |
| `CHUNK_OVERLAP` | `src/loader.py` | 300 | Overlap between consecutive chunks |
| `SEARCH_K` | `src/retriever.py` | 5 | Number of documents to retrieve |
| `FETCH_K` | `src/retriever.py` | 20 | MMR candidate pool size |
| `LAMBDA_MULT` | `src/retriever.py` | 0.7 | MMR diversity parameter (0=diverse, 1=relevant) |
| `MEMORY_WINDOW` | `src/agent.py` | 5 | Conversation turns to remember |
| `LLM_MODEL` | `src/agent.py` | llama-3.1-8b-instant | Groq model to use |

---

## 🛡️ Anti-Hallucination Measures

1. **Retrieval-Only Answering**: The system prompt strictly forbids using the LLM's internal knowledge.
2. **MMR Retrieval**: Diverse document selection provides comprehensive, non-redundant context.
3. **Source Attribution**: Every answer references specific documents and sections, making claims verifiable.
4. **Explicit Uncertainty**: When information isn't available, the assistant clearly states this instead of guessing.
5. **Temperature Zero**: The LLM is configured with temperature=0 for deterministic, focused responses.
6. **Metadata-Rich Context**: Each chunk includes source metadata so the LLM knows exactly where information comes from.

---

## 🔮 Future Improvements

- [ ] PDF and DOCX document support
- [ ] Hybrid retrieval (BM25 + vector search)
- [ ] Context compression for large result sets
- [ ] User authentication and session persistence
- [ ] Analytics dashboard (query patterns, response quality metrics)
- [ ] Multi-language support
- [ ] API endpoint for programmatic access
- [ ] Docker containerization for easy deployment
- [ ] Evaluation framework with RAGAS metrics

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

<p align="center">
  Built with ❤️ using LangChain, Groq, FAISS, and Streamlit
</p>
