# NexusAI 云端部署操作方案（Runbook）

本文档提供一套可落地的生产部署方案，目标是把当前项目稳定部署到云端并可持续运维。

## 1. 方案总览

### 1.1 推荐首发架构
- 云主机: 1 台 Ubuntu 22.04（先单机部署，后续再拆分）
- 运行方式: Docker Compose
- 网关: Nginx + HTTPS（Let's Encrypt）
- 核心服务:
  - Frontend: Next.js
  - Backend: FastAPI
  - Qdrant: 向量库
  - Redis: 历史会话
- 域名: `kb.your-domain.com`（示例）

### 1.2 资源规格建议
- 小流量验证环境:
  - 4 vCPU / 16GB RAM / 80GB SSD
- 生产建议（本地 Embedding 模型）:
  - 8 vCPU / 32GB RAM / 200GB SSD
  - 若要稳定低延迟，建议 GPU 实例（本地 embedding/LLM 场景）

## 2. 部署前检查（必须完成）

### 2.1 代码侧前置改造
当前前端将 API 地址写死为 `http://localhost:8001`，部署到云端后浏览器会请求“用户本机 localhost”，会失败。  
上线前必须改为“环境变量驱动”或“相对路径 + 反向代理”。

建议改造为：
1. 新增前端环境变量: `NEXT_PUBLIC_API_BASE_URL`
2. 所有 `fetch("http://localhost:8001/...")` 改为 `fetch(\`${NEXT_PUBLIC_API_BASE_URL}/...\`)`
3. 生产值设置为 `https://kb.your-domain.com`

### 2.2 密钥与配置
准备以下配置：
- `DEEPSEEK_API_KEY`（若使用 deepseek）
- `LLM_PROVIDER`（`deepseek` 或 `ollama`）
- `LLM_MODEL`
- `QDRANT_HOST=qdrant`
- `QDRANT_PORT=6333`
- `COLLECTION_NAME=nexusai_knowledge_base`
- `VECTOR_DIMENSION=1024`
- `REDIS_HOST=redis`（若后续代码显式接入 env）

## 3. 云主机初始化

### 3.1 服务器与安全组
开放端口：
- `22`（仅运维 IP）
- `80`
- `443`

不要对公网开放：
- `6333`（Qdrant）
- `6379`（Redis）
- `8001`（Backend）
- `3000`（Frontend）

### 3.2 安装 Docker / Compose / Nginx
```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg lsb-release nginx

sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

sudo usermod -aG docker $USER
newgrp docker
```

## 4. 拉取代码与部署目录

```bash
sudo mkdir -p /opt/nexusai
sudo chown -R $USER:$USER /opt/nexusai
cd /opt/nexusai
git clone git@github.com:benjaminazz1210/customer-knowledge-base.git .
```

## 5. 生产配置文件

### 5.1 Backend 环境文件
创建 `backend/.env`（示例）：
```env
LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-reasoner
DEEPSEEK_API_KEY=YOUR_DEEPSEEK_KEY
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1

QDRANT_HOST=qdrant
QDRANT_PORT=6333
COLLECTION_NAME=nexusai_knowledge_base

EMBEDDING_MODEL=Qwen/Qwen3-VL-Embedding-2B
VECTOR_DIMENSION=1024
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

### 5.2 Frontend 环境文件
创建 `frontend/.env.production`：
```env
NEXT_PUBLIC_API_BASE_URL=https://kb.your-domain.com
```

## 6. Docker Compose 生产编排

在项目根目录创建 `docker-compose.cloud.yml`（建议）：
```yaml
version: "3.9"

services:
  qdrant:
    image: qdrant/qdrant:latest
    restart: always
    volumes:
      - qdrant_data:/qdrant/storage
    networks:
      - nexus_net

  redis:
    image: redis:7-alpine
    restart: always
    command: ["redis-server", "--appendonly", "yes"]
    volumes:
      - redis_data:/data
    networks:
      - nexus_net

  backend:
    build:
      context: ./backend
    restart: always
    env_file:
      - ./backend/.env
    depends_on:
      - qdrant
      - redis
    command: uvicorn app.main:app --host 0.0.0.0 --port 8001
    volumes:
      - hf_cache:/root/.cache/huggingface
    networks:
      - nexus_net

  frontend:
    build:
      context: ./frontend
    restart: always
    depends_on:
      - backend
    environment:
      - NODE_ENV=production
    command: npm run start -- -p 3000
    networks:
      - nexus_net

volumes:
  qdrant_data:
  redis_data:
  hf_cache:

networks:
  nexus_net:
    driver: bridge
```

启动：
```bash
docker compose -f docker-compose.cloud.yml up -d --build
docker compose -f docker-compose.cloud.yml ps
```

## 7. Nginx 反向代理与 HTTPS

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

    location /api/ {
        proxy_pass http://127.0.0.1:8001/api/;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_buffering off;
        proxy_read_timeout 3600;
    }
}
```

启用并重载：
```bash
sudo ln -s /etc/nginx/sites-available/nexusai /etc/nginx/sites-enabled/nexusai
sudo nginx -t
sudo systemctl reload nginx
```

申请证书：
```bash
sudo apt-get install -y certbot python3-certbot-nginx
sudo certbot --nginx -d kb.your-domain.com
```

## 8. 验收步骤

```bash
curl -I https://kb.your-domain.com
curl -s https://kb.your-domain.com/api/health
```

前端手工验收：
1. 打开首页，确认页面可加载。
2. 上传一个 `.txt` 文件，确认出现在 `/files` 列表。
3. 发起聊天，确认有流式输出和 sources。
4. 删除文件，确认文件和向量均被删除。

后端自动验收（可选）：
```bash
cd /opt/nexusai
BASE_URL=https://kb.your-domain.com conda run -n daily_3_9 python backend/run_tests.py
```

## 9. CI/CD 发布流程建议

建议在 GitHub Actions 新增 `deploy.yml`：
1. 触发条件：`push main` 或 `workflow_dispatch`
2. 步骤：
   - SSH 到云主机
   - `git pull`
   - `docker compose -f docker-compose.cloud.yml up -d --build`
   - 健康检查 `curl /api/health`
3. 失败自动回滚：
   - 回退到上一个 git tag
   - 重启 compose

## 10. 运维与备份

### 10.1 日志
```bash
docker compose -f docker-compose.cloud.yml logs -f backend
docker compose -f docker-compose.cloud.yml logs -f frontend
```

### 10.2 备份
- Qdrant 数据卷：定时快照（每日）
- Redis AOF/RDB：每日备份
- 备份目标：对象存储（S3/OSS/COS）

### 10.3 监控告警（建议）
- 主机 CPU/内存/磁盘
- Backend `5xx` 比例
- `/api/health` 可用性
- Qdrant 请求延迟

## 11. 分阶段上线策略

### Phase A（当前推荐）
- 单机 Docker Compose
- 快速上线，低复杂度

### Phase B（流量上来后）
- Frontend 与 Backend 分离部署
- Qdrant 使用托管或独立节点
- Redis 托管化
- 加入 WAF/CDN 与灰度发布

## 12. 常见问题与排查

1. 前端请求命中 localhost
- 原因：API 地址写死 `localhost`
- 处理：改为环境变量或相对路径

2. 首次 embedding 很慢
- 原因：模型首次下载和加载
- 处理：预热请求 + 持久化 `hf_cache` 卷

3. SSE 中断
- 原因：反代超时或 buffering
- 处理：Nginx `proxy_buffering off` + 延长 `proxy_read_timeout`

4. 内存不足
- 原因：模型加载 + 并发请求
- 处理：升级实例或切到外部 embedding/LLM 服务
