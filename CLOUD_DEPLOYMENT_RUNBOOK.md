# NexusAI 阿里云 ECS 部署方案（更新版）

更新时间：2026-03-03  
适用代码分支：`main`（包含 Phase2/3/4 能力）

## 1. 目标与边界

本文档给出当前项目在阿里云 ECS 的可执行部署方案，覆盖：
- 文档上传 + RAG 对话（SSE）
- Phase2：结构化解析、图片描述入库、混合检索
- Phase3/4：`/api/workflows/generate` 与 `/api/workflows/revise` 文件生成与反馈改写

## 2. 推荐上线架构（当前代码最稳）

当前仓库没有现成 `Dockerfile`（前后端），因此推荐：
- 前端：Next.js（系统进程 + `systemd`）
- 后端：FastAPI（Conda 环境 + `systemd`）
- Qdrant：Docker 单容器（仅本机监听）
- Redis：系统服务（仅本机监听）
- Nginx：统一反代 + TLS（Let's Encrypt）

当前代码约束（请按本文方案部署）：
- `HistoryService` 默认连接 `localhost:6379`，因此 Redis 需与 backend 在同机本地监听
- backend 无 Redis host 环境变量切换能力（如需容器化拆分，需先改代码）

架构拓扑：
- `:443` -> Nginx
- Nginx `/` -> `127.0.0.1:3000`（frontend）
- Nginx `/api/` -> `127.0.0.1:8001`（backend）
- backend -> `127.0.0.1:6333`（Qdrant）
- backend -> `127.0.0.1:6379`（Redis）

## 3. 资源规格与网络策略

### 3.1 ECS 规格建议
- 测试环境：`4 vCPU / 16GB RAM / 80GB SSD`
- 生产建议：`8 vCPU / 32GB RAM / 200GB SSD`
- 若保留本地大 embedding（`Qwen/Qwen3-VL-Embedding-2B`），建议优先高内存实例

### 3.2 安全组
只开放：
- `22`（限制为运维固定 IP）
- `80`
- `443`

禁止公网开放：
- `3000`、`8001`、`6333`、`6379`

## 4. 服务器初始化

以下以 Ubuntu 22.04 为例，部署目录使用 `/opt/nexusai`。

```bash
sudo apt-get update
sudo apt-get install -y git curl nginx redis-server ca-certificates gnupg lsb-release
```

安装 Docker（仅用于 Qdrant）：

```bash
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo \"$VERSION_CODENAME\") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER
newgrp docker
```

创建部署目录并拉代码：

```bash
sudo mkdir -p /opt/nexusai
sudo chown -R $USER:$USER /opt/nexusai
cd /opt/nexusai
git clone git@github.com:benjaminazz1210/customer-knowledge-base.git .
```

## 5. 中间件部署（Qdrant + Redis）

### 5.1 Qdrant（Docker）
```bash
mkdir -p /opt/nexusai/qdrant_data
docker run -d \
  --name qdrant \
  --restart unless-stopped \
  -p 127.0.0.1:6333:6333 \
  -v /opt/nexusai/qdrant_data:/qdrant/storage \
  qdrant/qdrant:latest
```

### 5.2 Redis（系统服务）
Redis 默认本机监听，按需加固：

```bash
sudo sed -i 's/^#\\?bind .*/bind 127.0.0.1 ::1/' /etc/redis/redis.conf
sudo sed -i 's/^protected-mode .*/protected-mode yes/' /etc/redis/redis.conf
sudo systemctl enable redis-server
sudo systemctl restart redis-server
```

## 6. 应用依赖与构建

### 6.1 Backend（Conda）
建议使用 Miniforge/Conda，环境名保持与项目一致：`daily_3_9`。

```bash
cd /opt/nexusai
conda create -y -n daily_3_9 python=3.9
conda run -n daily_3_9 pip install -r backend/requirements.txt
```

补充系统依赖（给 `unstructured`/OCR/PDF 解析用）：

```bash
sudo apt-get install -y libmagic1 poppler-utils tesseract-ocr libgl1
```

### 6.2 Frontend（Node）
建议 Node 20 LTS：

```bash
cd /opt/nexusai/frontend
npm ci
npm run build -- --webpack
```

## 7. 生产配置文件

### 7.1 Backend `.env`
创建 `/opt/nexusai/backend/.env`（示例）：

```env
# LLM provider (openai | heiyucode | deepseek | ollama)
LLM_PROVIDER=heiyucode
LLM_MODEL=gpt-5.3-codex
OPENAI_API_KEY=YOUR_HEIYUCODE_OR_OPENAI_KEY
OPENAI_BASE_URL=https://www.heiyucode.com/v1

# Vector store
QDRANT_HOST=127.0.0.1
QDRANT_PORT=6333
COLLECTION_NAME=nexusai_knowledge_base
VECTOR_DIMENSION=1024

# Embedding/chunk
EMBEDDING_BACKEND=dashscope
EMBEDDING_MODEL=qwen3-vl-embedding
DASHSCOPE_API_KEY=YOUR_DASHSCOPE_API_KEY
DASHSCOPE_EMBEDDING_MODEL=qwen3-vl-embedding
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Phase2: parser + vision
DOCUMENT_PARSER_BACKEND=auto
LLAMA_CLOUD_API_KEY=YOUR_LLAMA_CLOUD_API_KEY
VISION_ENABLED=true
VISION_MODEL=gpt-4o-mini
VISION_MAX_IMAGES=20

# Phase3/4: workflow
WORKFLOW_OUTPUT_DIR=generated
WORKFLOW_TEMPLATE_DIR=assets/templates
WORKFLOW_HYBRID_ALPHA=0.7
WORKFLOW_MAX_CONTEXT_CHUNKS=8
```

说明：
- `EMBEDDING_BACKEND=dashscope`：使用阿里云线上 embedding，降低 ECS 内存压力
- `DOCUMENT_PARSER_BACKEND=auto`：优先尝试 `unstructured` -> `llamaparse`，失败自动回退 `builtin`
- `generated/` 下会产出 docx/pptx，需持久化与备份

### 7.2 Frontend `.env.production`
创建 `/opt/nexusai/frontend/.env.production`：

```env
NEXT_PUBLIC_API_BASE_URL=https://kb.your-domain.com
```

## 8. systemd 托管（前后端）

> 将 `<DEPLOY_USER>` 替换为你的 ECS 登录用户（如 `ubuntu` 或 `ecs-user`）。

### 8.1 Backend service
`/etc/systemd/system/nexusai-backend.service`

```ini
[Unit]
Description=NexusAI Backend (FastAPI)
After=network.target redis-server.service

[Service]
Type=simple
User=<DEPLOY_USER>
WorkingDirectory=/opt/nexusai/backend
Environment=PYTHONUNBUFFERED=1
ExecStart=/bin/bash -lc 'source ~/.bashrc && conda run -n daily_3_9 uvicorn app.main:app --host 127.0.0.1 --port 8001'
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### 8.2 Frontend service
`/etc/systemd/system/nexusai-frontend.service`

```ini
[Unit]
Description=NexusAI Frontend (Next.js)
After=network.target

[Service]
Type=simple
User=<DEPLOY_USER>
WorkingDirectory=/opt/nexusai/frontend
Environment=NODE_ENV=production
ExecStart=/bin/bash -lc 'npm run start -- -p 3000'
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

启用与启动：

```bash
sudo systemctl daemon-reload
sudo systemctl enable nexusai-backend nexusai-frontend
sudo systemctl restart nexusai-backend nexusai-frontend
sudo systemctl status nexusai-backend --no-pager
sudo systemctl status nexusai-frontend --no-pager
```

## 9. Nginx 与 HTTPS

创建 `/etc/nginx/sites-available/nexusai`：

```nginx
server {
    listen 80;
    server_name kb.your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /api/chat {
        proxy_pass http://127.0.0.1:8001/api/chat;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_buffering off;
        proxy_read_timeout 3600;
    }

    location /api/ {
        proxy_pass http://127.0.0.1:8001/api/;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 3600;
    }
}
```

启用：

```bash
sudo ln -sf /etc/nginx/sites-available/nexusai /etc/nginx/sites-enabled/nexusai
sudo nginx -t
sudo systemctl reload nginx
```

签发证书：

```bash
sudo apt-get install -y certbot python3-certbot-nginx
sudo certbot --nginx -d kb.your-domain.com
```

## 10. 验收清单（上线必测）

```bash
curl -sS https://kb.your-domain.com/api/health
curl -sS "https://kb.your-domain.com/api/workflows/jobs?limit=3"
```

前端手工验收：
1. 聊天页可打开，`/api/chat` 有流式输出（SSE）
2. 上传 `.pdf/.docx/.pptx` 正常入库并可溯源
3. 新对话支持“保存并新建/清空并新建”，聊天记录可恢复
4. 工作流页可生成 docx/pptx，并能按反馈修订

后端自动验收（建议）：

```bash
cd /opt/nexusai
BASE_URL=https://kb.your-domain.com conda run -n daily_3_9 python backend/run_tests.py

# 如需把结果回写到 feature_list.json
BASE_URL=https://kb.your-domain.com conda run -n daily_3_9 python backend/run_tests.py --write-feature-status
```

## 11. 发布与回滚

### 11.1 发布
```bash
cd /opt/nexusai
git pull origin main
conda run -n daily_3_9 pip install -r backend/requirements.txt
cd frontend && npm ci && npm run build -- --webpack && cd ..
sudo systemctl restart nexusai-backend nexusai-frontend
```

### 11.2 回滚
```bash
cd /opt/nexusai
git log --oneline -n 5
git checkout <last_good_commit>
cd frontend && npm run build -- --webpack && cd ..
sudo systemctl restart nexusai-backend nexusai-frontend
```

## 12. 运维与备份建议

### 12.1 日志
```bash
sudo journalctl -u nexusai-backend -f
sudo journalctl -u nexusai-frontend -f
docker logs -f qdrant
```

### 12.2 备份对象
- `/opt/nexusai/qdrant_data`（向量数据）
- `/opt/nexusai/backend/generated/files`（工作流产物）
- `/opt/nexusai/backend/assets/templates`（企业模板）
- `/var/lib/redis`（会话历史）

建议每日快照到阿里云 OSS，并保留最近 7~30 天版本。

## 13. 常见问题

1. `api/health` 返回 `qdrant=false`
- 检查 `docker ps` 中 `qdrant` 是否存活
- 检查 `QDRANT_HOST/QDRANT_PORT` 是否为 `127.0.0.1:6333`

2. 聊天不流式或中途断开
- 检查 Nginx `location /api/chat` 是否设置 `proxy_buffering off`
- 检查 `proxy_read_timeout`

3. 上传后解析效果不稳定
- 检查 `DOCUMENT_PARSER_BACKEND` 与 `LLAMA_CLOUD_API_KEY`
- 缺少 LlamaParse key 时会自动回退 builtin（可用但精度较低）

4. 工作流文件生成失败
- 确认 `backend/generated/` 可写权限
- 查看 `journalctl -u nexusai-backend -f` 中 workflow 异常
