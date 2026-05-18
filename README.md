# vibeclip-ai-proxy（Railway AI Proxy）

本项目是 **vibeclip_ali（阿里云版维播）** 在 Railway 上部署的 **最小 OpenAI 视觉中转服务**，用于国内链路经由海外节点调用 OpenAI Vision，并返回模型原始文本供阿里云后端继续做 S1 JSON 解析与 schema 校验。

**说明：**

- **不是** vibeclip（纯海外版维播主项目）。
- **不是** vibeclip_ali 主项目仓库（前后端在阿里云）。
- 本仓库为 **独立项目**，仅承担鉴权、转发 Vision Chat Completions、返回 `raw_text`。

## 功能范围（第一阶段）

- `GET /health`：健康检查。
- `POST /s1/vision`：S1 产品图片理解；校验内部 Token；将图片 URL、`system_prompt`、`user_payload` 发往 OpenAI-compatible `chat/completions`；返回 `choices[0].message.content` 作为 `raw_text`。

不包含数据库、前端、通用 AI 网关、S2/S3/S4/S5、业务 JSON 解析或业务规则。

## 环境变量

| 变量 | 说明 |
|------|------|
| `PROXY_AUTH_TOKEN` | 内部鉴权 Token（**必填**，缺失则请求返回 500） |
| `OPENAI_API_KEY` | OpenAI API Key（**必填**，缺失则请求返回 500） |
| `OPENAI_BASE_URL` | OpenAI 兼容 API 根路径，默认 `https://api.openai.com/v1` |
| `S1_VISION_MODEL` | 模型名，默认 `gpt-4o-mini` |
| `REQUEST_TIMEOUT_SECONDS` | 上游请求超时（秒），默认 `120` |

参考 `.env.example` 填写本地或 Railway 变量。

## 本地启动

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

（需已安装依赖：`pip install -r requirements.txt`）

## 接口测试

健康检查：

```bash
curl http://localhost:8000/health
```

S1 Vision（将 `$PROXY_AUTH_TOKEN` 设为与服务器一致的值）：

```bash
curl -X POST "http://localhost:8000/s1/vision" \
  -H "Authorization: Bearer $PROXY_AUTH_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "image_urls": ["https://example.com/product.png"],
    "system_prompt": "You are a product image understanding assistant. Return JSON only.",
    "user_payload": {
      "task": "understand product image"
    }
  }'
```

## Railway 部署

1. 在 Railway 新建项目，连接本 Git 仓库（或通过 CLI 部署）。
2. **Root Directory**：指向本仓库根目录（包含 `main.py`、`requirements.txt`、`Procfile`）。
3. Railway 会自动识别 `Procfile`：`web: uvicorn main:app --host 0.0.0.0 --port $PORT`。
4. 在 Railway **Variables** 中配置下列环境变量（至少 `PROXY_AUTH_TOKEN`、`OPENAI_API_KEY`）：
   - `PROXY_AUTH_TOKEN`
   - `OPENAI_API_KEY`
   - （可选）`OPENAI_BASE_URL`、`S1_VISION_MODEL`、`REQUEST_TIMEOUT_SECONDS`
5. 部署完成后，将分配的公开 URL 提供给 vibeclip_ali 后端作为代理地址（例如 `https://<your-service>.up.railway.app/s1/vision`）。

## Railway 环境变量清单

- **必填**：`PROXY_AUTH_TOKEN`、`OPENAI_API_KEY`
- **可选**：`OPENAI_BASE_URL`、`S1_VISION_MODEL`、`REQUEST_TIMEOUT_SECONDS`
