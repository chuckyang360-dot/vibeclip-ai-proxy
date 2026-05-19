import base64
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
    model: str | None = Field(default=None, description="Image model id; when set, forwarded upstream as-is")
    response_format: str = Field(default="url", description="url | b64_json")
    aspect_ratio: str | None = None
    resolution: str | None = None
    project_id: int | None = None
    target_type: str | None = None
    target_id: int | None = None


class ImageGenerationDataItem(BaseModel):
    b64_json: str


class ImageGenerationB64Response(BaseModel):
    """Caller always receives inline base64 so backends behind restrictive egress need not fetch image URLs."""

    data: list[ImageGenerationDataItem]
    mime_type: str


_MAX_IMAGE_DOWNLOAD_BYTES = 25 * 1024 * 1024


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


def resolve_image_generation_model(body_model: str | None) -> tuple[str, str]:
    """Pick upstream image model for /images/generations.

    Priority:
      1. Request body ``model`` (non-empty)
      2. XAI_IMAGE_MODEL
      3. SHORT_DRAMA_XAI_IMAGE_MODEL (legacy alias)
      4. IMAGE_MODEL
      5. grok-imagine-image

    Never uses XAI_MODEL / OPENAI_MODEL (those are text/chat defaults).
    """
    if body_model is not None:
        m = str(body_model).strip()
        if m:
            return m, "request_body"
    for key in ("XAI_IMAGE_MODEL", "SHORT_DRAMA_XAI_IMAGE_MODEL", "IMAGE_MODEL"):
        v = get_env(key)
        if v:
            return v.strip(), key
    return "grok-imagine-image", "default_grok_imagine_image"


def truncate_for_log(text: str, max_len: int = 500) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "...(truncated)"


def _mime_from_content_type(content_type: str | None) -> str | None:
    if not content_type or not str(content_type).strip():
        return None
    main = str(content_type).split(";")[0].strip().lower()
    return main if main else None


def sniff_image_mime(data: bytes) -> str:
    if len(data) >= 8 and data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if len(data) >= 3 and data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    if len(data) >= 6 and (data.startswith(b"GIF87a") or data.startswith(b"GIF89a")):
        return "image/gif"
    return "application/octet-stream"


def _resolve_mime_from_bytes(raw: bytes, content_type: str | None) -> str:
    ct = _mime_from_content_type(content_type)
    if ct and ct.startswith("image/"):
        return ct
    sniffed = sniff_image_mime(raw)
    if sniffed != "application/octet-stream":
        return sniffed
    return ct or "image/png"


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


@app.post("/images/generations", response_model=ImageGenerationB64Response)
async def images_generations(
    body: ImageGenerationRequest,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> ImageGenerationB64Response:
    """Forward POST /images/generations upstream; always return inline ``data[0].b64_json`` + ``mime_type``.

    If the upstream returns a temporary ``url`` (common when ``response_format=url``), the proxy fetches
    the bytes on Railway and base64-encodes them so callers behind restrictive egress never hit xAI URLs.
    """
    request_id = str(uuid.uuid4())
    require_proxy_auth(authorization)

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key is None or openai_key.strip() == "":
        raise HTTPException(status_code=500, detail="openai_api_key_not_configured")

    base_url = get_env("OPENAI_BASE_URL", "https://api.openai.com/v1") or "https://api.openai.com/v1"
    requested_raw = (body.model or "").strip() if body.model is not None else ""
    requested_model_log = requested_raw if requested_raw else "(none)"
    resolved_model, model_source_env = resolve_image_generation_model(body.model)
    provider_label = infer_upstream_provider_label(base_url)
    timeout_raw = get_env("REQUEST_TIMEOUT_SECONDS", "120") or "120"
    try:
        timeout_seconds = float(timeout_raw)
    except ValueError:
        timeout_seconds = 120.0

    timeout = httpx.Timeout(timeout_seconds)
    fmt = (body.response_format or "url").strip().lower()
    if fmt not in ("url", "b64_json"):
        raise HTTPException(
            status_code=400,
            detail="response_format must be 'url' or 'b64_json'",
        )

    payload: dict[str, Any] = {
        "model": resolved_model,
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
        f"requested_model={requested_model_log} resolved_model={resolved_model} "
        f"model_source={model_source_env} provider={provider_label} "
        f"upstream_url={upstream_url} response_format={fmt} "
        f"project_id={body.project_id} target_type={body.target_type or ''} target_id={body.target_id} "
        f"timeout_seconds={timeout_seconds}"
    )

    start = time.perf_counter()

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(upstream_url, headers=headers, json=payload)

            if resp.status_code >= 400:
                elapsed_ms = int((time.perf_counter() - start) * 1000)
                body_preview = truncate_for_log(resp.text[:800], max_len=800)
                print(
                    f"[AI_PROXY_IMAGE_ERROR] request_id={request_id} "
                    f"error_type=upstream_http upstream_status_code={resp.status_code} "
                    f"requested_model={requested_model_log} resolved_model={resolved_model} "
                    f"upstream_body_preview={body_preview} elapsed_ms={elapsed_ms}"
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
                elapsed_ms = int((time.perf_counter() - start) * 1000)
                print(
                    f"[AI_PROXY_IMAGE_ERROR] request_id={request_id} "
                    f"error_type=json_decode_error requested_model={requested_model_log} "
                    f"resolved_model={resolved_model} message={truncate_for_log(str(exc))} "
                    f"elapsed_ms={elapsed_ms}"
                )
                raise HTTPException(status_code=500, detail="unexpected_error")

            items = data.get("data")
            if not isinstance(items, list) or not items:
                elapsed_ms = int((time.perf_counter() - start) * 1000)
                print(
                    f"[AI_PROXY_IMAGE_ERROR] request_id={request_id} "
                    f"error_type=invalid_upstream_response requested_model={requested_model_log} "
                    f"resolved_model={resolved_model} message=missing_data_array elapsed_ms={elapsed_ms}"
                )
                raise HTTPException(status_code=502, detail="invalid_upstream_response")

            first = items[0]
            if not isinstance(first, dict):
                raise HTTPException(status_code=502, detail="invalid_upstream_response")

            b64_in = first.get("b64_json")
            url_in = first.get("url")

            final_b64: str
            mime_type: str
            b64_source: str

            if isinstance(b64_in, str) and b64_in.strip():
                final_b64 = b64_in.strip()
                try:
                    raw_bytes = base64.standard_b64decode(final_b64)
                except Exception:
                    elapsed_ms = int((time.perf_counter() - start) * 1000)
                    print(
                        f"[AI_PROXY_IMAGE_ERROR] request_id={request_id} "
                        f"error_type=invalid_b64_json requested_model={requested_model_log} "
                        f"resolved_model={resolved_model} elapsed_ms={elapsed_ms}"
                    )
                    raise HTTPException(status_code=502, detail="invalid_b64_json")
                mime_type = _resolve_mime_from_bytes(raw_bytes, None)
                b64_source = "upstream_b64"
            elif isinstance(url_in, str) and url_in.strip():
                img_url = url_in.strip()
                try:
                    dl = await client.get(img_url, follow_redirects=True)
                except httpx.TimeoutException:
                    elapsed_ms = int((time.perf_counter() - start) * 1000)
                    print(
                        f"[AI_PROXY_IMAGE_ERROR] request_id={request_id} "
                        f"error_type=image_download_timeout requested_model={requested_model_log} "
                        f"resolved_model={resolved_model} elapsed_ms={elapsed_ms}"
                    )
                    raise HTTPException(status_code=504, detail="image_download_timeout")
                except httpx.RequestError as exc:
                    elapsed_ms = int((time.perf_counter() - start) * 1000)
                    print(
                        f"[AI_PROXY_IMAGE_ERROR] request_id={request_id} "
                        f"error_type=image_download_network requested_model={requested_model_log} "
                        f"resolved_model={resolved_model} message={truncate_for_log(str(exc))} "
                        f"elapsed_ms={elapsed_ms}"
                    )
                    raise HTTPException(status_code=502, detail="image_download_network_error")

                if dl.status_code >= 400:
                    elapsed_ms = int((time.perf_counter() - start) * 1000)
                    prev = truncate_for_log(dl.text[:800], max_len=800)
                    print(
                        f"[AI_PROXY_IMAGE_ERROR] request_id={request_id} "
                        f"error_type=image_download_http upstream_status_code={dl.status_code} "
                        f"requested_model={requested_model_log} resolved_model={resolved_model} "
                        f"upstream_body_preview={prev} elapsed_ms={elapsed_ms}"
                    )
                    raise HTTPException(
                        status_code=502,
                        detail={
                            "error": "image_download_upstream_error",
                            "status_code": dl.status_code,
                            "body": dl.text[:800],
                        },
                    )

                raw_bytes = dl.content
                if len(raw_bytes) > _MAX_IMAGE_DOWNLOAD_BYTES:
                    elapsed_ms = int((time.perf_counter() - start) * 1000)
                    print(
                        f"[AI_PROXY_IMAGE_ERROR] request_id={request_id} "
                        f"error_type=image_too_large requested_model={requested_model_log} "
                        f"resolved_model={resolved_model} elapsed_ms={elapsed_ms}"
                    )
                    raise HTTPException(status_code=502, detail="image_too_large")

                mime_type = _resolve_mime_from_bytes(raw_bytes, dl.headers.get("content-type"))
                final_b64 = base64.standard_b64encode(raw_bytes).decode("ascii")
                b64_source = "url_downloaded"
            else:
                elapsed_ms = int((time.perf_counter() - start) * 1000)
                print(
                    f"[AI_PROXY_IMAGE_ERROR] request_id={request_id} "
                    f"error_type=missing_image_payload requested_model={requested_model_log} "
                    f"resolved_model={resolved_model} elapsed_ms={elapsed_ms}"
                )
                raise HTTPException(status_code=502, detail="missing_image_payload")

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        print(
            f"[AI_PROXY_IMAGE_RESPONSE] request_id={request_id} success=true format=b64_json "
            f"source={b64_source} mime_type={mime_type} resolved_model={resolved_model} "
            f"elapsed_ms={elapsed_ms}"
        )
        return ImageGenerationB64Response(
            data=[ImageGenerationDataItem(b64_json=final_b64)],
            mime_type=mime_type,
        )

    except HTTPException:
        raise

    except httpx.TimeoutException:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        print(
            f"[AI_PROXY_IMAGE_ERROR] request_id={request_id} "
            f"error_type=upstream_timeout requested_model={requested_model_log} "
            f"resolved_model={resolved_model} elapsed_ms={elapsed_ms}"
        )
        raise HTTPException(status_code=504, detail="upstream_timeout")

    except httpx.ConnectError as exc:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        print(
            f"[AI_PROXY_IMAGE_ERROR] request_id={request_id} "
            f"error_type=network_connect_error requested_model={requested_model_log} "
            f"resolved_model={resolved_model} message={truncate_for_log(str(exc))} elapsed_ms={elapsed_ms}"
        )
        raise HTTPException(status_code=502, detail="network_connect_error")

    except httpx.RequestError as exc:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        print(
            f"[AI_PROXY_IMAGE_ERROR] request_id={request_id} "
            f"error_type=network_connect_error requested_model={requested_model_log} "
            f"resolved_model={resolved_model} message={truncate_for_log(str(exc))} elapsed_ms={elapsed_ms}"
        )
        raise HTTPException(status_code=502, detail="network_connect_error")

    except Exception as exc:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        print(
            f"[AI_PROXY_IMAGE_ERROR] request_id={request_id} "
            f"error_type=unexpected_error requested_model={requested_model_log} "
            f"resolved_model={resolved_model} message={truncate_for_log(str(exc))} elapsed_ms={elapsed_ms}"
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
