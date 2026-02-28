# NexusAI - Phase 1 MVP

High-availability Enterprise Knowledge Base with RAG support.

## 🎯 Harness Philosophy
This project follows the **Harness** development methodology:
- **`init.sh`**: Automated environment setup.
- **`feature_list.json`**: Behavior-driven feature tracking.
- **`progress.md`**: Real-time project status.

## 🚀 Getting Started

### 1. Prerequisites
- Docker (for Qdrant)
- Conda (Python 3.9 — `daily_3_9` environment)
- Node.js (v18+)

### 2. Backend Setup
```bash
cd backend
# Activate environment
conda activate daily_3_9
# Install dependencies
pip install -r requirements.txt
# Verify dependencies and API connectivity
PYTHONPATH=. python app/harness_check.py
# Start server
uvicorn app.main:app --reload --port 8001
```

### 3. Frontend Setup
```bash
cd frontend
# Install dependencies
npm install
# Start development server
npm run dev
```

## 🛠 Tech Stack
- **Frontend**: Next.js (App Router), Tailwind CSS
- **Backend**: FastAPI, Qdrant (Vector DB)
- **LLM**: DeepSeek Reasoner API
- **Embeddings**: Local `Qwen3-VL-Embedding-2B` (Multimodal, 32k context)
