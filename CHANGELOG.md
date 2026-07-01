# Changelog

All notable changes to the Agentic RAG Assistant are documented in this file.

## [2.0.0] — 2025-07-01

### Production-Quality Upgrade

Comprehensive upgrade from working prototype to portfolio-ready RAG application.
Every enhancement targets improved retrieval quality, response accuracy, and user experience.

---

### 📚 Knowledge Base (Part 1)

**Changed**: Expanded all 6 knowledge documents from ~500-1100 bytes to ~3000-5000 words each.

**Why it matters for RAG**: Larger, well-structured documents provide more diverse chunks for retrieval. Dense information organized under clear headings improves embedding quality because each chunk represents a complete semantic idea rather than a fragment of a long paragraph.

| Document | Before | After |
|----------|--------|-------|
| artificial_intelligence.txt | 1,120 bytes | ~12,000 bytes |
| biotechnology.txt | 858 bytes | ~12,000 bytes |
| climate_science.txt | 830 bytes | ~11,000 bytes |
| quantum_computing.txt | 864 bytes | ~12,000 bytes |
| space_exploration.txt | 792 bytes | ~12,000 bytes |
| sustainable_energy.txt | 724 bytes | ~12,000 bytes |

Each document now includes: Introduction, History, Timeline, Core Concepts, Technical Explanation, Terminology, Types, Architecture/Working, Advantages, Limitations, Applications, Industry Use Cases, Case Studies, Future Trends, Interesting Facts, Common Misconceptions, Comparison Tables, Glossary, FAQ, Interview Questions, and Summary.

**Removed**: `sample_documents.txt` — contained trivial, unrelated facts that polluted retrieval results.

---

### ✂️ Improved Chunking (Part 2)

**Changed** in `src/loader.py`:
- `chunk_size`: 1000 → 1500 (keeps sections intact)
- `chunk_overlap`: 200 → 300 (more context at boundaries)
- Added heading-aware separators: `["\n\n## ", "\n\n### ", "\n\n", "\n", " "]`

**Why it matters for RAG**: Heading-aware splitting prevents chunks from starting mid-section. Larger chunks with more overlap preserve the semantic coherence of each chunk, improving both embedding quality and retrieval relevance.

---

### 🏷️ Metadata Enrichment (Part 3)

**Added** in `src/loader.py`:
- Every chunk now stores: `document_name`, `topic`, `section`, `chunk_id`
- Section is extracted from the nearest Markdown heading
- Topic is derived from the filename

**Why it matters for RAG**: Metadata enables source attribution in responses, allowing users to verify claims. It also enables future filtering and faceted search.

---

### 🔍 MMR Retrieval (Part 4)

**Changed** in `src/retriever.py`:
- Replaced custom `similarity_search_with_score` with LangChain's built-in MMR retriever
- Configuration: `k=5`, `fetch_k=20`, `lambda_mult=0.7`
- Removed hardcoded L2 distance threshold (1.1)

**Why it matters for RAG**: MMR (Maximal Marginal Relevance) selects documents that are both relevant AND diverse. This prevents returning near-duplicate chunks and provides broader context to the LLM, leading to more comprehensive answers.

---

### 📝 Professional Prompt (Part 5)

**Changed** in `src/prompts.py`:
- Complete rewrite with structured instructions for response formatting
- Anti-hallucination guardrails (context-only, explicit uncertainty)
- Source attribution instructions
- Suggested follow-up question generation
- Conversation awareness instructions
- Added `MessagesPlaceholder` for chat history

**Why it matters for RAG**: A well-crafted prompt is the most impactful single improvement in a RAG system. It determines how the LLM uses the retrieved context, whether it hallucinates, and how readable the output is.

---

### 📎 Source Attribution (Part 6)

**Added** in `src/retriever.py` and `src/agent.py`:
- `extract_sources()` extracts deduplicated metadata from retrieved documents
- `format_docs_for_prompt()` includes source tags in the context
- Every response includes document name, section, and chunk ID

**Why it matters for RAG**: Source attribution makes answers verifiable and builds user trust. It also helps users navigate the knowledge base for deeper exploration.

---

### 🧠 Conversation Memory (Part 7)

**Added** in `src/agent.py`:
- Sliding window memory (last 5 conversation turns)
- Uses LangChain `HumanMessage` and `AIMessage` objects
- Passed to the prompt via `MessagesPlaceholder`
- `clear_memory()` method for resetting conversations

**Why it matters for RAG**: Memory enables natural follow-up questions ("What about healthcare?", "Explain that further") without requiring the user to re-state the full context. This is essential for a professional chatbot experience.

---

### 💡 Suggested Questions (Part 8)

**Added** in `src/agent.py` and `src/prompts.py`:
- LLM generates 3 suggested follow-up questions after each answer
- Questions are parsed from the response and displayed as clickable buttons
- Clicking a suggestion automatically sends it as the next query

**Why it matters for RAG**: Suggested questions guide users toward productive queries, improving engagement and demonstrating the breadth of the knowledge base.

---

### ⚡ Streaming Responses (Part 9)

**Added** in `src/agent.py` and `ui/app.py`:
- `stream()` method yields tokens as they are generated
- UI uses `st.write_stream()` for real-time display
- Memory is updated after streaming completes

**Why it matters for RAG**: Streaming provides immediate visual feedback, making the application feel responsive. Without streaming, users wait for the entire response before seeing anything.

---

### 🎨 UI Enhancement (Part 10)

**Changed** in `ui/app.py`:
- Custom CSS dark theme with gradients and hover effects
- Source attribution cards with topic, section, and chunk info
- Suggested question buttons (clickable)
- Sidebar with knowledge base topics, system info, and controls
- File upload section for adding documents
- Clear chat button with memory reset
- `st.status` for initialization progress
- Removed Streamlit branding (main menu and footer hidden)

**Why it matters**: A polished UI is essential for portfolio projects. The dark theme, source cards, and interactive suggestions create a professional, modern appearance.

---

### 🚀 Performance Optimization (Part 11)

**Added** in `src/embeddings.py` and `src/utils.py`:
- SHA-256 hash-based cache invalidation for the vector store
- Data hash is stored alongside the FAISS index
- On startup, current document hash is compared with stored hash
- Embeddings are only regenerated when documents actually change
- `force_rebuild` parameter for manual cache bypass

**Why it matters for RAG**: Embedding generation is the most time-consuming step. Hash-based caching reduces startup time from ~30 seconds to ~2 seconds on subsequent launches with unchanged documents.

---

### 📊 Enhanced Logging (Part 12)

**Added** in `src/utils.py`:
- `log_step()` function with colored, emoji-prefixed console output
- Styles: info (ℹ), success (✓), warning (⚠), error (✗), step (→)
- Key pipeline stages are logged for demo-friendly visibility

**Why it matters**: Clear, visible logging makes demonstrations more impressive and debugging easier. Colored output distinguishes between different types of messages at a glance.

---

### 🧹 Project Cleanup (Part 13)

**Changed**:
- All source files now have module-level docstrings
- Functions have comprehensive docstrings with Args/Returns sections
- Configuration constants are defined at module level with comments
- `main.py` uses `sys.executable` for reliable Python invocation
- `.gitignore` expanded with additional entries
- `requirements.txt` pinned with minimum versions

---

### 📖 Documentation (Part 14)

**Changed**: `README.md` — Complete rewrite with:
- Feature table
- ASCII architecture diagram
- Tech stack with badges
- Folder structure
- Step-by-step installation guide
- Configuration reference table
- Anti-hallucination measures section
- Future improvements roadmap

**Added**: `CHANGELOG.md` — This file, documenting every modification with rationale.
