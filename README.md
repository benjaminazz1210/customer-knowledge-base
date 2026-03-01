# NexusAI - Customer Knowledge Base

企业知识库问答系统（RAG），支持文档上传、向量检索、流式回答和来源追溯。

## 1. 项目架构

```mermaid
graph TD
    U[User]
    F[Frontend<br/>Next.js App Router]
    B[Backend API<br/>FastAPI]
    P[DocumentParser]
    C[TextChunker]
    E[EmbeddingService<br/>Qwen3-VL-Embedding-2B]
    V[VectorStore]
    Q[(Qdrant)]
    R[RAGService]
    H[(Redis History)]
    L[LLM Provider<br/>Ollama or DeepSeek]

    U --> F
    F -->|REST/SSE| B
    B -->|upload| P --> C --> E --> V --> Q
    B -->|chat| R
    R --> E
    R --> V --> Q
    B --> H
    R --> L
```

### 组件职责
- `frontend/`: Chat 页面与文件管理页面，消费 REST + SSE。
- `backend/app/routers/`: API 路由层（`upload/files/chat/history/health`）。
- `backend/app/services/`: 文档解析、分块、嵌入、向量库和 RAG 编排。
- `Qdrant`: 存储 chunk 向量与 payload。
- `Redis`: 保存会话历史（`/api/history`）。
- `LLM`: 通过 OpenAI 兼容接口调用（默认可用 `ollama`，也支持 `deepseek`）。

## 2. Embedding 具体实现

### 2.1 文档到向量的上传链路
1. `POST /api/upload` 读取文件字节流。  
2. `DocumentParser.parse` 按扩展名解析文本：`.txt/.md/.pdf/.docx/.pptx`。  
3. `TextChunker.chunk` 进行固定窗口切片（默认 `chunk_size=1000`，`overlap=200`）。  
4. `EmbeddingService.get_embeddings` 调用 `Qwen3VLEmbedder.process` 生成向量。  
5. 若模型输出维度 > `config.VECTOR_DIMENSION`（默认 `1024`），执行截断（MRL 友好）。  
6. `VectorStore.upsert_chunks` 写入 Qdrant，payload 包含：
   - `source_file`
   - `chunk_text`
   - `chunk_index`

### 2.2 EmbeddingService 关键设计
- 单例初始化：`EmbeddingService.__new__` 避免重复加载大模型。
- 自动设备选择：
  - `mps`（Apple Silicon 优先）
  - `cuda`
  - `cpu`
- dtype 选择：
  - MPS 使用 `float16`
  - 其他默认 `float32`
- 多模态能力已预留：`get_multimodal_embeddings` 支持 `{"text": ...}` / `{"image": ...}` 输入。

### 2.3 向量存储实现要点
- `VectorStore._ensure_collection` 启动时检查 collection 维度。
- 如果实际维度与配置维度不一致，会重建 collection，避免脏数据导致检索失败。
- 距离函数使用 `COSINE`，适合语义检索场景。

## 3. RAG 具体实现

### 3.1 检索增强流程（`RAGService.generate_response`）
1. 将用户 query 向量化。  
2. 在 Qdrant 检索 Top-K（当前 `limit=5`）。  
3. 提取命中 chunk，拼接为 `context_text`。  
4. 构造 `system_prompt`，要求“基于检索文档回答，不可编造”。  
5. 追加 Redis 历史对话（`/api/history`）并发送给 LLM。  
6. 使用流式生成返回给上层路由。

### 3.2 Chat SSE 协议（`POST /api/chat`）
后端按如下事件顺序输出：
1. `data: {"sources":[...]}`（一次）  
2. `data: {"token":"..."}`（多次）  
3. `data: [DONE]`（结束标记）  

前端逐行解析 `data:`，实时拼接 token，并渲染来源引用。

### 3.3 失败与兜底策略
- 请求体校验：Pydantic 自动返回 `422`（例如 message 缺失）。
- 上传异常分层：
  - 空文档/不支持格式 -> `400`
  - 其他异常 -> `500`
- 流式阶段异常：输出错误 token，并最终输出 `[DONE]`，避免前端挂起。

## 4. 对应知识点（学习导图）

### 检索与向量化
- Chunking（窗口切片 + overlap）
- Dense Embedding（语义向量）
- MRL 截断（以更低维度保留主要语义）
- Cosine Similarity
- Top-K Retrieval

### 生成与编排
- RAG（Retrieval-Augmented Generation）
- Prompt Grounding（让回答绑定检索上下文）
- Hallucination 控制（无命中时明确告知）
- Multi-turn Context（历史会话拼接）

### 工程实现
- FastAPI 路由分层
- SSE 流式协议（`text/event-stream`）
- Qdrant payload filter delete（按 `source_file` 级联删除）
- Redis 会话持久化
- 严格回归测试矩阵（`feature_list.json` + `backend/run_tests.py`）

## 5. 本地启动

### 5.1 依赖
- Docker（Qdrant/Redis）
- Conda 环境（建议 Python 3.9）
- Node.js 18+

### 5.2 Backend
```bash
cd backend
conda run -n daily_3_9 pip install -r requirements.txt
conda run -n daily_3_9 uvicorn app.main:app --reload --port 8001
```

### 5.3 Frontend
```bash
cd frontend
npm install
npm run dev
```

## 6. 严格测试与 CI

### 6.1 本地严格回归
```bash
cd backend
conda run -n daily_3_9 python run_tests.py
```

说明：
- 该脚本会执行 `feature_list.json` 中全部 `auto` 项并回写 `passes`。
- 覆盖基础设施、后端成功/失败路径、前端 lint/build 与关键静态契约检查。

### 6.2 GitHub Actions
- 工作流文件：`.github/workflows/ci.yml`
- 触发：`push`、`pull_request`、`workflow_dispatch`
- 关键步骤：
  - 启动/检查 Qdrant 与 Redis
  - 启动后端
  - 执行 `backend/run_tests.py`

## 7. 参考文档
- 技术深挖：`TECHNICAL_DEEP_DIVE.md`
- 功能验收清单：`feature_list.json`
- 项目进度：`progress.md`
