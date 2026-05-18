import json
import os
import time
import uuid
from typing import Annotated, Any

import httpx
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

app = FastAPI()


class S1VisionRequest(BaseModel):
    image_urls: list[str] = Field(..., description="Product image URLs")
    system_prompt: str
    user_payload: dict[str, Any]


class S1VisionResponse(BaseModel):
    raw_text: str


def get_env(key: str, default: str | None = None) -> str | None:
    val = os.getenv(key)
    if val is None or val.strip() == "":
        return default
    return val


def truncate_for_log(text: str, max_len: int = 500) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "...(truncated)"


def parse_bearer_token(authorization: str | None) -> str | None:
    """Return token string if Bearer format is valid; None means 401 (malformed/missing)."""
    if authorization is None:
        return None
    stripped = authorization.strip()
    parts = stripped.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip()


def require_proxy_auth(authorization: str | None) -> None:
    proxy_token = os.getenv("PROXY_AUTH_TOKEN")
    if proxy_token is None or proxy_token.strip() == "":
        raise HTTPException(status_code=500, detail="proxy_auth_token_not_configured")

    parsed = parse_bearer_token(authorization)
    if parsed is None:
        raise HTTPException(status_code=401, detail="missing_or_invalid_authorization")

    if parsed != proxy_token:
        raise HTTPException(status_code=403, detail="forbidden")


def extract_message_content(data: dict[str, Any]) -> str | None:
    choices = data.get("choices")
    if not choices:
        return None
    message = choices[0].get("message") or {}
    content = message.get("content")

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                t = item.get("text")
                if isinstance(t, str):
                    parts.append(t)
        joined = "".join(parts)
        return joined if joined else None
    return None


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.post("/s1/vision", response_model=S1VisionResponse)
async def s1_vision(
    body: S1VisionRequest,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> S1VisionResponse:
    request_id = str(uuid.uuid4())
    image_count = len(body.image_urls)

    require_proxy_auth(authorization)

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key is None or openai_key.strip() == "":
        raise HTTPException(status_code=500, detail="openai_api_key_not_configured")

    base_url = get_env("OPENAI_BASE_URL", "https://api.openai.com/v1") or "https://api.openai.com/v1"
    model = get_env("S1_VISION_MODEL", "gpt-4o-mini") or "gpt-4o-mini"
    timeout_raw = get_env("REQUEST_TIMEOUT_SECONDS", "120") or "120"
    try:
        timeout_seconds = float(timeout_raw)
    except ValueError:
        timeout_seconds = 120.0

    timeout = httpx.Timeout(timeout_seconds)

    print(
        f"[AI_PROXY_S1_VISION_REQUEST] request_id={request_id} "
        f"image_count={image_count} model={model}"
    )

    user_content: list[dict[str, Any]] = []
    for url in body.image_urls:
        user_content.append({"type": "image_url", "image_url": {"url": url}})
    user_content.append({"type": "text", "text": json.dumps(body.user_payload, ensure_ascii=False)})

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": body.system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.2,
    }

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {openai_key}",
        "Content-Type": "application/json",
    }

    start = time.perf_counter()

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, headers=headers, json=payload)

        elapsed_ms = int((time.perf_counter() - start) * 1000)

        if resp.status_code >= 400:
            body_preview = truncate_for_log(resp.text[:800], max_len=800)
            print(
                f"[AI_PROXY_S1_VISION_ERROR] request_id={request_id} "
                f"error_type=upstream_http status_code={resp.status_code} "
                f"message={truncate_for_log(body_preview)} elapsed_ms={elapsed_ms}"
            )
            raise HTTPException(
                status_code=502,
                detail={
                    "error": "upstream_error",
                    "status_code": resp.status_code,
                    "body": resp.text[:800],
                },
            )

        try:
            data = resp.json()
        except json.JSONDecodeError as exc:
            print(
                f"[AI_PROXY_S1_VISION_ERROR] request_id={request_id} "
                f"error_type=json_decode_error message={truncate_for_log(str(exc))} "
                f"elapsed_ms={elapsed_ms}"
            )
            raise HTTPException(status_code=500, detail="unexpected_error")

        raw_text = extract_message_content(data)
        if raw_text is None or raw_text.strip() == "":
            print(
                f"[AI_PROXY_S1_VISION_ERROR] request_id={request_id} "
                f"error_type=empty_model_output message={truncate_for_log('empty content')} "
                f"elapsed_ms={elapsed_ms}"
            )
            raise HTTPException(status_code=502, detail="empty_model_output")

        print(
            f"[AI_PROXY_S1_VISION_RESPONSE] request_id={request_id} success=true "
            f"raw_length={len(raw_text)} elapsed_ms={elapsed_ms}"
        )
        return S1VisionResponse(raw_text=raw_text)

    except HTTPException:
        raise

    except httpx.TimeoutException:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        print(
            f"[AI_PROXY_S1_VISION_ERROR] request_id={request_id} "
            f"error_type=upstream_timeout message={truncate_for_log('request timed out')} "
            f"elapsed_ms={elapsed_ms}"
        )
        raise HTTPException(status_code=504, detail="upstream_timeout")

    except httpx.ConnectError as exc:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        print(
            f"[AI_PROXY_S1_VISION_ERROR] request_id={request_id} "
            f"error_type=network_connect_error message={truncate_for_log(str(exc))} "
            f"elapsed_ms={elapsed_ms}"
        )
        raise HTTPException(status_code=502, detail="network_connect_error")

    except httpx.RequestError as exc:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        print(
            f"[AI_PROXY_S1_VISION_ERROR] request_id={request_id} "
            f"error_type=network_connect_error message={truncate_for_log(str(exc))} "
            f"elapsed_ms={elapsed_ms}"
        )
        raise HTTPException(status_code=502, detail="network_connect_error")

    except Exception as exc:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        print(
            f"[AI_PROXY_S1_VISION_ERROR] request_id={request_id} "
            f"error_type=unexpected_error message={truncate_for_log(str(exc))} "
            f"elapsed_ms={elapsed_ms}"
        )
        raise HTTPException(status_code=500, detail="unexpected_error")
