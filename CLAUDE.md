# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MemoryChat is an AI memorial companion system that uses RAG (Retrieval-Augmented Generation) to create conversational experiences based on a deceased person's chat history and notes. This is a sensitive application requiring strict ethical and legal compliance.

**Critical Ethics & Safety Requirements:**
- All responses MUST be labeled as AI simulations, never claiming to be the actual person
- Requires explicit legal authorization (will/executor consent) before use
- Must filter dangerous content (suicide, self-harm, medical/legal/financial advice)
- Family members must be able to delete all data at any time
- This tool assists with grief counseling but CANNOT replace professional psychological support

## Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment (optional for local SBERT, required for LLM APIs)
cp .env.example .env
# Edit .env to add OPENAI_API_KEY or ANTHROPIC_API_KEY

# Run interactive demo
python demo.py

# Test individual modules
python src/utils/safety_filter.py
```

## Core Architecture

The system follows a 4-stage RAG pipeline:

### 1. Data Parsing (`src/parsers/`)
- **WhatsAppParser**: Parses WhatsApp export files (.txt format)
  - Supports multiple timestamp formats (DD/MM/YYYY, MM/DD/YYYY, YYYY-MM-DD)
  - Filters system messages and media placeholders
  - Extracts persona features (common words, message patterns, avg length)
  - Multi-line message handling for conversation continuity

### 2. Vectorization (`src/rag/vectorizer.py`)
- **Pluggable embedding providers**:
  - `sbert` (local, no API key, recommended for testing): Uses sentence-transformers
  - `openai` (cloud, requires API key): Uses OpenAI embeddings API
- **Pluggable vector stores**:
  - `simple` (in-memory, for testing): Basic cosine similarity search
  - `chroma` (persistent, for production): ChromaDB with HNSW indexing
- **Document chunking**: Messages grouped into overlapping chunks (default: 5 messages with 1 overlap)
- **Factory pattern**: Use `create_vectorizer(provider, store, **kwargs)` for instantiation

### 3. RAG Pipeline (`src/rag/rag_pipeline.py`)
- **Retrieval**: Query → embedding → vector search → top-k relevant documents
- **Prompt Construction**:
  - System prompt includes persona features (common words, style)
  - User prompt includes retrieved context with clear attribution
  - Always instructs to label responses with AI identity
- **LLM Support**:
  - OpenAI (GPT-3.5/GPT-4)
  - Anthropic (Claude 3 family)
- **Two query modes**:
  - `query()`: Single-turn Q&A
  - `chat()`: Multi-turn conversations with history

### 4. Safety Filtering (`src/utils/safety_filter.py`)
- **Multi-layer filtering**:
  1. User query screening (detects sensitive topics before generation)
  2. Forbidden words (prevents identity confusion)
  3. Professional advice blocking (medical/legal/financial)
  4. Privacy info detection (credit cards, SSN, IDs, emails)
- **Emergency response**: Triggers on suicide/self-harm keywords, provides crisis hotlines
- **AI labeling enforcement**: Ensures all responses include `[基于 {name} 历史的 AI 模拟]` prefix

## Key Configuration (`.env`)

```bash
# Embedding: "sbert" (local) or "openai" (API)
EMBEDDING_PROVIDER=sbert

# Vector store: "simple" (memory) or "chroma" (persistent)
VECTOR_STORE=chroma

# LLM: "openai" or "anthropic" (requires corresponding API key)
LLM_PROVIDER=openai

# RAG parameters
RAG_TOP_K=5          # Number of documents to retrieve
CHUNK_SIZE=5         # Messages per chunk
CHUNK_OVERLAP=1      # Overlapping messages between chunks
MAX_TOKENS=500       # LLM response limit

# Safety
ENABLE_EMERGENCY_ALERT=true
```

## Data Flow

```
WhatsApp export (.txt)
    ↓
WhatsAppParser.parse_file() → List[Message]
    ↓
WhatsAppParser.clean_messages() → Filtered messages
    ↓
WhatsAppParser.extract_persona_features() → Persona dict
    ↓
Vectorizer.process_messages() → List[Document] with embeddings
    ↓
Vectorizer.index_documents() → Store in vector DB
    ↓
RAGPipeline.query(question) → Retrieve docs → Build prompt → LLM generate
    ↓
SafetyFilter.filter() → Check for sensitive/harmful content
    ↓
RAGResponse (with retrieved_docs, metadata)
```

## Module Interconnections

**Critical integration points:**
1. **Parser → Vectorizer**: `Message.to_dict()` converts parsed messages to dict format expected by `process_messages()`
2. **Vectorizer → RAG**: The `Vectorizer` instance is passed to `RAGPipeline` and used for `.search()` during queries
3. **RAG ↔ Safety**: `SafetyFilter` is called twice:
   - Before generation: checks user query for sensitive topics
   - After generation: validates LLM response for forbidden content
4. **Persona Features**: Extracted by parser, passed to `PromptBuilder` in RAG pipeline to customize system prompt

## Development Guidelines

**When adding new features:**

1. **New embedding providers**: Extend `EmbeddingProvider` base class in `vectorizer.py`, add to `create_vectorizer()` factory
2. **New vector stores**: Extend `VectorStore` base class, implement `add_documents()` and `search()` methods
3. **New LLM providers**: Extend `LLMProvider` base class in `rag_pipeline.py`, handle system message differences (e.g., Claude requires separate `system` parameter)
4. **Safety filters**: Add keywords to `SafetyFilter.SENSITIVE_KEYWORDS` dict, or patterns to `PROFESSIONAL_ADVICE_PATTERNS`

**Testing changes:**
- Run `demo.py` for end-to-end testing with sample data
- Individual module tests available by running files directly (e.g., `python src/utils/safety_filter.py`)
- Sample data auto-generated in `data/sample/chat.txt` if missing

**Dependencies are optional by design:**
- All imports wrapped in try/except with `*_AVAILABLE` flags
- System gracefully degrades and provides helpful error messages
- Allows running with minimal dependencies (e.g., SBERT only, no LLM API)

## Important Implementation Notes

**WhatsApp timestamp parsing**: The parser tries multiple date formats sequentially. If adding support for new locales, add formats to `_parse_timestamp()` formats list.

**Persona prompt construction**: The `PromptBuilder` class generates prompts that:
- Explicitly forbid claiming to be the real person
- Require AI identity labeling at response start
- Block medical/legal/financial advice
- Include retrieved context with clear separation

**Vector search scoring**: ChromaDB uses cosine distance (lower = more similar), SimpleVectorStore uses cosine similarity (higher = more similar). Handle accordingly when displaying scores.

**Multi-turn chat context**: The `chat()` method injects retrieved context only into the **last** user message to avoid context bloat, while preserving full conversation history for coherence.

## Security & Privacy Reminders

- Never commit `.env` files (already in `.gitignore`)
- Persona data and chat histories contain PII - treat as highly sensitive
- `chroma_db/` directory contains vector embeddings of personal data - must be encrypted in production
- Safety filter is a starting point, not a complete solution - professional moderation recommended for production use
