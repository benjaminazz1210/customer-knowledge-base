# 单机 Docker 部署（前后端 + Qdrant + Redis + Nginx）

> 目标：在同一台云主机上运行完整系统，LLM/Embedding 走线上 API。

## 1. 本地准备环境文件

```bash
cp deploy/backend.env.example deploy/backend.env
```

可选：使用一键脚本切换整套模型配置（默认会更新 `backend/.env` 和 `deploy/backend.env`）：

```bash
python3 scripts/switch_profile.py local --dry-run
python3 scripts/switch_profile.py local
# 或
python3 scripts/switch_profile.py cloud
```

仅查看当前配置（不写入）：

```bash
python3 scripts/switch_profile.py cloud --check --file deploy/backend.env
```

编辑 `deploy/backend.env`，至少填写：

- `OPENAI_API_KEY`（你的 heiyucode key）
- `OPENAI_BASE_URL`（例如 `https://www.heiyucode.com/v1`）
- `LLM_PROVIDER=heiyucode`
- `LLM_MODEL=gpt-5.3-codex`
- `DASHSCOPE_API_KEY`
- `EMBEDDING_BACKEND=dashscope`
- `DASHSCOPE_EMBEDDING_MODEL=qwen3-vl-embedding`

## 2. 本地构建镜像（使用 compose build 覆盖文件）

```bash
docker compose -f docker-compose.yml -f docker-compose.build.yml build
```

构建完成后会得到：

- `nexusai-backend:latest`
- `nexusai-frontend:latest`

## 3. 导出镜像并上传到云主机

```bash
docker save -o nexusai-images.tar nexusai-backend:latest nexusai-frontend:latest
# 上传文件到云主机（示例）
scp nexusai-images.tar docker-compose.yml deploy/backend.env deploy/nginx/default.conf root@YOUR_SERVER_IP:/opt/nexusai/
```

## 4. 云主机导入镜像并启动

```bash
cd /opt/nexusai
docker load -i nexusai-images.tar
docker compose up -d
```

### 可选：先不使用 Nginx（快速跑通）

当你暂时不想上 Nginx 时，使用 no-nginx 覆盖文件启动，并确保前端构建时 API 基址指向后端地址。

1. 本地构建前端镜像时显式指定 API 地址（示例：云主机公网 IP）：

```bash
NEXT_PUBLIC_API_BASE_URL=http://YOUR_SERVER_IP:8001 \
docker compose -f docker-compose.yml -f docker-compose.build.yml build frontend
```

2. 云主机启动（禁用 Nginx，直接暴露 `3000/8001`）：

```bash
docker compose -f docker-compose.yml -f docker-compose.no-nginx.yml up -d
```

3. 访问：

- 前端：`http://YOUR_SERVER_IP:3000`
- 后端健康检查：`http://YOUR_SERVER_IP:8001/api/health`

## 5. 验证

```bash
curl -sS http://127.0.0.1/api/health
# 或从本机访问
curl -sS http://YOUR_SERVER_IP/api/health
```

预期返回：

```json
{"status":"ok","qdrant":true}
```

## 6. 常用运维命令

```bash
docker compose ps
docker compose logs -f nginx
docker compose logs -f backend
docker compose restart backend
```

## 7. 说明

- 这个 compose 方案默认对外暴露 `80` 端口（Nginx）。
- 浏览器访问路径：`http://YOUR_SERVER_IP/`。
- 前端通过同源 `/api` 调后端，SSE (`/api/chat`) 已在 Nginx 配置中关闭 buffering。
- Qdrant 与 Redis 使用 Docker volume 持久化：`qdrant_data`、`redis_data`、`backend_generated`。
