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


class TextCompletionRequest(BaseModel):
    system_prompt: str
    user_text: str = Field(..., description="User message body, often JSON string from backend")
    image_urls: list[str] = Field(default_factory=list, description="Optional vision parts (OpenAI-style image_url)")
    max_tokens: int = Field(default=8192, ge=1, le=200000)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    service_name: str = Field(default="", description="Caller hint for logs, e.g. creative_brief")


class S1VisionResponse(BaseModel):
    raw_text: str


class ImageGenerationRequest(BaseModel):
    prompt: str
    response_format: str = Field(default="url", description="url | b64_json")
    aspect_ratio: str | None = None
    resolution: str | None = None
    project_id: int | None = None
    target_type: str | None = None
    target_id: int | None = None


class ImageGenerationResponse(BaseModel):
    url: str | None = None
    b64_json: str | None = None


def get_env(key: str, default: str | None = None) -> str | None:
    val = os.getenv(key)
    if val is None or val.strip() == "":
        return default
    return val


def infer_upstream_provider_label(base_url: str) -> str:
    u = (base_url or "").strip().lower()
    if "x.ai" in u:
        return "xai"
    if "openai.com" in u:
        return "openai"
    return "openai_compatible"


def resolve_s1_vision_model() -> tuple[str, str]:
    """Return (model_id, env_key_used). No implicit gpt-4o-mini — must be explicit config."""
    for key in ("S1_VISION_MODEL", "XAI_MODEL", "OPENAI_MODEL"):
        v = get_env(key)
        if v:
            return v.strip(), key
    return "", ""


def resolve_text_chat_model() -> tuple[str, str]:
    """Structured text / creative_brief / chat: prefer dedicated text model env, then shared XAI_MODEL."""
    for key in ("XAI_TEXT_MODEL", "XAI_MODEL", "OPENAI_MODEL"):
        v = get_env(key)
        if v:
            return v.strip(), key
    return "", ""


def resolve_image_generation_model() -> tuple[str, str]:
    """S3 asset images: prefer dedicated image model env, then shared XAI_MODEL."""
    for key in ("XAI_IMAGE_MODEL", "SHORT_DRAMA_XAI_IMAGE_MODEL", "XAI_MODEL", "OPENAI_MODEL"):
        v = get_env(key)
        if v:
            return v.strip(), key
    return "grok-imagine-image", "default_grok_imagine_image"


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


@app.post("/text/completions", response_model=S1VisionResponse)
async def text_completions(
    body: TextCompletionRequest,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> S1VisionResponse:
    """OpenAI-compatible chat/completions via Railway; returns assistant message as raw_text."""
    request_id = str(uuid.uuid4())
    image_count = len(body.image_urls)
    user_chars = len(body.user_text or "")

    require_proxy_auth(authorization)

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key is None or openai_key.strip() == "":
        raise HTTPException(status_code=500, detail="openai_api_key_not_configured")

    base_url = get_env("OPENAI_BASE_URL", "https://api.openai.com/v1") or "https://api.openai.com/v1"
    model, model_source_env = resolve_text_chat_model()
    if not model:
        print(
            f"[AI_PROXY_TEXT_COMPLETION_ERROR] request_id={request_id} "
            f"error_type=model_not_configured message="
            f"{truncate_for_log('set XAI_TEXT_MODEL or XAI_MODEL or OPENAI_MODEL')}"
        )
        raise HTTPException(
            status_code=500,
            detail=(
                "text_chat_model_not_configured: set one of "
                "XAI_TEXT_MODEL, XAI_MODEL, OPENAI_MODEL"
            ),
        )

    provider_label = infer_upstream_provider_label(base_url)
    timeout_raw = get_env("REQUEST_TIMEOUT_SECONDS", "120") or "120"
    try:
        timeout_seconds = float(timeout_raw)
    except ValueError:
        timeout_seconds = 120.0

    timeout = httpx.Timeout(timeout_seconds)

    if body.image_urls:
        user_parts: list[dict[str, Any]] = []
        for url in body.image_urls:
            user_parts.append({"type": "image_url", "image_url": {"url": url}})
        user_parts.append({"type": "text", "text": body.user_text})
        user_message: dict[str, Any] = {"role": "user", "content": user_parts}
    else:
        user_message = {"role": "user", "content": body.user_text}

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": body.system_prompt},
            user_message,
        ],
        "temperature": body.temperature,
        "max_tokens": body.max_tokens,
    }

    upstream_url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {openai_key}",
        "Content-Type": "application/json",
    }

    svc = truncate_for_log(body.service_name or "(unset)", max_len=80)
    print(
        f"[AI_PROXY_TEXT_COMPLETION_REQUEST] request_id={request_id} "
        f"provider={provider_label} base_url={base_url} model={model} model_env={model_source_env} "
        f"service_name={svc} image_count={image_count} user_text_chars={user_chars} "
        f"max_tokens={body.max_tokens}"
    )

    start = time.perf_counter()

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(upstream_url, headers=headers, json=payload)

        elapsed_ms = int((time.perf_counter() - start) * 1000)

        if resp.status_code >= 400:
            body_preview = truncate_for_log(resp.text[:800], max_len=800)
            print(
                f"[AI_PROXY_TEXT_COMPLETION_ERROR] request_id={request_id} "
                f"error_type=upstream_http status_code={resp.status_code} "
                f"message={body_preview} elapsed_ms={elapsed_ms}"
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
                f"[AI_PROXY_TEXT_COMPLETION_ERROR] request_id={request_id} "
                f"error_type=json_decode_error message={truncate_for_log(str(exc))} "
                f"elapsed_ms={elapsed_ms}"
            )
            raise HTTPException(status_code=500, detail="unexpected_error")

        raw_text = extract_message_content(data)
        if raw_text is None or raw_text.strip() == "":
            print(
                f"[AI_PROXY_TEXT_COMPLETION_ERROR] request_id={request_id} "
                f"error_type=empty_model_output message={truncate_for_log('empty content')} "
                f"elapsed_ms={elapsed_ms}"
            )
            raise HTTPException(status_code=502, detail="empty_model_output")

        print(
            f"[AI_PROXY_TEXT_COMPLETION_RESPONSE] request_id={request_id} success=true "
            f"raw_length={len(raw_text)} elapsed_ms={elapsed_ms}"
        )
        return S1VisionResponse(raw_text=raw_text)

    except HTTPException:
        raise

    except httpx.TimeoutException:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        print(
            f"[AI_PROXY_TEXT_COMPLETION_ERROR] request_id={request_id} "
            f"error_type=upstream_timeout message={truncate_for_log('request timed out')} "
            f"elapsed_ms={elapsed_ms}"
        )
        raise HTTPException(status_code=504, detail="upstream_timeout")

    except httpx.ConnectError as exc:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        print(
            f"[AI_PROXY_TEXT_COMPLETION_ERROR] request_id={request_id} "
            f"error_type=network_connect_error message={truncate_for_log(str(exc))} "
            f"elapsed_ms={elapsed_ms}"
        )
        raise HTTPException(status_code=502, detail="network_connect_error")

    except httpx.RequestError as exc:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        print(
            f"[AI_PROXY_TEXT_COMPLETION_ERROR] request_id={request_id} "
            f"error_type=network_connect_error message={truncate_for_log(str(exc))} "
            f"elapsed_ms={elapsed_ms}"
        )
        raise HTTPException(status_code=502, detail="network_connect_error")

    except Exception as exc:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        print(
            f"[AI_PROXY_TEXT_COMPLETION_ERROR] request_id={request_id} "
            f"error_type=unexpected_error message={truncate_for_log(str(exc))} "
            f"elapsed_ms={elapsed_ms}"
        )
        raise HTTPException(status_code=500, detail="unexpected_error")


@app.post("/images/generations", response_model=ImageGenerationResponse)
async def images_generations(
    body: ImageGenerationRequest,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> ImageGenerationResponse:
    """OpenAI-compatible POST /images/generations via Railway (xAI grok-imagine-image, etc.)."""
    request_id = str(uuid.uuid4())
    require_proxy_auth(authorization)

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key is None or openai_key.strip() == "":
        raise HTTPException(status_code=500, detail="openai_api_key_not_configured")

    base_url = get_env("OPENAI_BASE_URL", "https://api.openai.com/v1") or "https://api.openai.com/v1"
    model, model_source_env = resolve_image_generation_model()
    provider_label = infer_upstream_provider_label(base_url)
    timeout_raw = get_env("REQUEST_TIMEOUT_SECONDS", "120") or "120"
    try:
        timeout_seconds = float(timeout_raw)
    except ValueError:
        timeout_seconds = 120.0

    timeout = httpx.Timeout(timeout_seconds)
    fmt = (body.response_format or "url").strip().lower()
    payload: dict[str, Any] = {
        "model": model,
        "prompt": body.prompt,
        "n": 1,
        "response_format": fmt,
    }
    if body.aspect_ratio:
        payload["aspect_ratio"] = body.aspect_ratio
    if body.resolution:
        payload["resolution"] = body.resolution

    upstream_url = f"{base_url.rstrip('/')}/images/generations"
    headers = {
        "Authorization": f"Bearer {openai_key}",
        "Content-Type": "application/json",
    }

    print(
        f"[AI_PROXY_IMAGE_REQUEST] request_id={request_id} "
        f"provider={provider_label} base_url={base_url} model={model} model_env={model_source_env} "
        f"project_id={body.project_id} target_type={body.target_type or ''} target_id={body.target_id} "
        f"response_format={fmt} timeout_seconds={timeout_seconds}"
    )

    start = time.perf_counter()

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(upstream_url, headers=headers, json=payload)

        elapsed_ms = int((time.perf_counter() - start) * 1000)

        if resp.status_code >= 400:
            body_preview = truncate_for_log(resp.text[:800], max_len=800)
            print(
                f"[AI_PROXY_IMAGE_ERROR] request_id={request_id} "
                f"error_type=upstream_http status_code={resp.status_code} "
                f"message={body_preview} elapsed_ms={elapsed_ms}"
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
                f"[AI_PROXY_IMAGE_ERROR] request_id={request_id} "
                f"error_type=json_decode_error message={truncate_for_log(str(exc))} "
                f"elapsed_ms={elapsed_ms}"
            )
            raise HTTPException(status_code=500, detail="unexpected_error")

        items = data.get("data")
        if not isinstance(items, list) or not items:
            print(
                f"[AI_PROXY_IMAGE_ERROR] request_id={request_id} "
                f"error_type=invalid_upstream_response message=missing data[] elapsed_ms={elapsed_ms}"
            )
            raise HTTPException(status_code=502, detail="invalid_upstream_response")

        first = items[0]
        if not isinstance(first, dict):
            raise HTTPException(status_code=502, detail="invalid_upstream_response")

        if fmt == "b64_json":
            b64 = first.get("b64_json")
            if not isinstance(b64, str) or not b64.strip():
                raise HTTPException(status_code=502, detail="missing_b64_json")
            print(
                f"[AI_PROXY_IMAGE_RESPONSE] request_id={request_id} success=true format=b64_json "
                f"elapsed_ms={elapsed_ms}"
            )
            return ImageGenerationResponse(b64_json=b64.strip())

        url = first.get("url")
        if not isinstance(url, str) or not url.strip():
            raise HTTPException(status_code=502, detail="missing_image_url")

        print(
            f"[AI_PROXY_IMAGE_RESPONSE] request_id={request_id} success=true format=url "
            f"elapsed_ms={elapsed_ms}"
        )
        return ImageGenerationResponse(url=url.strip())

    except HTTPException:
        raise

    except httpx.TimeoutException:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        print(
            f"[AI_PROXY_IMAGE_ERROR] request_id={request_id} "
            f"error_type=upstream_timeout elapsed_ms={elapsed_ms}"
        )
        raise HTTPException(status_code=504, detail="upstream_timeout")

    except httpx.ConnectError as exc:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        print(
            f"[AI_PROXY_IMAGE_ERROR] request_id={request_id} "
            f"error_type=network_connect_error message={truncate_for_log(str(exc))} elapsed_ms={elapsed_ms}"
        )
        raise HTTPException(status_code=502, detail="network_connect_error")

    except httpx.RequestError as exc:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        print(
            f"[AI_PROXY_IMAGE_ERROR] request_id={request_id} "
            f"error_type=network_connect_error message={truncate_for_log(str(exc))} elapsed_ms={elapsed_ms}"
        )
        raise HTTPException(status_code=502, detail="network_connect_error")

    except Exception as exc:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        print(
            f"[AI_PROXY_IMAGE_ERROR] request_id={request_id} "
            f"error_type=unexpected_error message={truncate_for_log(str(exc))} elapsed_ms={elapsed_ms}"
        )
        raise HTTPException(status_code=500, detail="unexpected_error")


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
    model, model_source_env = resolve_s1_vision_model()
    if not model:
        print(
            f"[AI_PROXY_S1_VISION_ERROR] request_id={request_id} "
            f"error_type=model_not_configured message="
            f"{truncate_for_log('set S1_VISION_MODEL or XAI_MODEL or OPENAI_MODEL')}"
        )
        raise HTTPException(
            status_code=500,
            detail=(
                "s1_vision_model_not_configured: set one of "
                "S1_VISION_MODEL, XAI_MODEL, OPENAI_MODEL (no default model)"
            ),
        )
    provider_label = infer_upstream_provider_label(base_url)
    timeout_raw = get_env("REQUEST_TIMEOUT_SECONDS", "120") or "120"
    try:
        timeout_seconds = float(timeout_raw)
    except ValueError:
        timeout_seconds = 120.0

    timeout = httpx.Timeout(timeout_seconds)

    print(
        f"[AI_PROXY_S1_VISION_REQUEST] request_id={request_id} "
        f"provider={provider_label} base_url={base_url} model={model} "
        f"model_env={model_source_env} image_count={image_count}"
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
