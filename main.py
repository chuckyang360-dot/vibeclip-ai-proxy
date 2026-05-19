import asyncio
import base64
import json
import os
import time
import uuid
from typing import Annotated, Any, Union
from urllib.parse import quote

import boto3
import httpx
from botocore.config import Config
from botocore.exceptions import ClientError
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
    response_format: str = Field(
        default="url",
        description="url | b64_json | r2_url (r2_url uploads to R2 and returns public URL only)",
    )
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


class ImageGenerationDataUrlItem(BaseModel):
    url: str


class ImageGenerationR2UrlResponse(BaseModel):
    data: list[ImageGenerationDataUrlItem]
    mime_type: str
    storage: str = Field(default="r2")


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


def extension_for_mime(mime_type: str) -> str:
    m = (mime_type or "").strip().lower().split(";")[0].strip()
    return {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/gif": ".gif",
    }.get(m, ".bin")


def sanitize_path_segment(seg: str | None, fallback: str = "unknown") -> str:
    s = (seg or "").strip() or fallback
    return s.replace("/", "_").replace("\\", "_")[:200]


def build_r2_image_object_key(
    *,
    project_id: int | None,
    target_type: str | None,
    target_id: int | None,
    request_id: str,
    mime_type: str,
) -> str:
    ext = extension_for_mime(mime_type)
    pid = project_id if project_id is not None else 0
    tid = target_id if target_id is not None else 0
    tt = sanitize_path_segment(target_type, "unknown")
    return f"short-drama/assets/{pid}/{tt}/{tid}/{request_id}{ext}"


def load_r2_settings() -> tuple[str, str, str, str, str]:
    endpoint = (get_env("R2_ENDPOINT") or "").strip()
    access = (get_env("R2_ACCESS_KEY") or "").strip()
    secret = (get_env("R2_SECRET_KEY") or "").strip()
    bucket = (get_env("R2_BUCKET_NAME") or "").strip()
    public_base = (get_env("R2_PUBLIC_BASE_URL") or "").strip().rstrip("/")
    return endpoint, access, secret, bucket, public_base


def upload_bytes_to_r2_sync(*, object_key: str, data: bytes, content_type: str) -> None:
    endpoint, access, secret, bucket, _public = load_r2_settings()
    if not all((endpoint, access, secret, bucket)):
        raise RuntimeError("r2_env_incomplete")

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access,
        aws_secret_access_key=secret,
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )
    client.put_object(Bucket=bucket, Key=object_key, Body=data, ContentType=content_type)


async def extract_image_bytes_from_upstream_item(
    client: httpx.AsyncClient,
    first: dict[str, Any],
    *,
    request_id: str,
    requested_model_log: str,
    resolved_model: str,
    start: float,
) -> tuple[bytes, str, str]:
    """Return (raw_bytes, mime_type, source_tag) where source_tag is upstream_b64 | url_downloaded."""
    b64_in = first.get("b64_json")
    url_in = first.get("url")

    if isinstance(b64_in, str) and b64_in.strip():
        fb64 = b64_in.strip()
        try:
            raw_bytes = base64.standard_b64decode(fb64)
        except Exception:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            print(
                f"[AI_PROXY_IMAGE_ERROR] request_id={request_id} "
                f"error_type=invalid_b64_json requested_model={requested_model_log} "
                f"resolved_model={resolved_model} elapsed_ms={elapsed_ms}"
            )
            raise HTTPException(status_code=502, detail="invalid_b64_json")
        mime_type = _resolve_mime_from_bytes(raw_bytes, None)
        return raw_bytes, mime_type, "upstream_b64"

    if isinstance(url_in, str) and url_in.strip():
        img_url = url_in.strip()
        try:
            dl = await client.get(img_url, follow_redirects=True)
        except httpx.TimeoutException:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            print(
                f"[AI_PROXY_IMAGE_ERROR] request_id={request_id} "
                f"error_type=upstream_image_download_failed reason=timeout "
                f"requested_model={requested_model_log} resolved_model={resolved_model} elapsed_ms={elapsed_ms}"
            )
            raise HTTPException(status_code=504, detail="upstream_image_download_failed")
        except httpx.RequestError as exc:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            print(
                f"[AI_PROXY_IMAGE_ERROR] request_id={request_id} "
                f"error_type=upstream_image_download_failed reason=network "
                f"requested_model={requested_model_log} resolved_model={resolved_model} "
                f"message={truncate_for_log(str(exc))} elapsed_ms={elapsed_ms}"
            )
            raise HTTPException(status_code=502, detail="upstream_image_download_failed")

        if dl.status_code >= 400:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            prev = truncate_for_log(dl.text[:800], max_len=800)
            print(
                f"[AI_PROXY_IMAGE_ERROR] request_id={request_id} "
                f"error_type=upstream_image_download_failed reason=http "
                f"upstream_status_code={dl.status_code} requested_model={requested_model_log} "
                f"resolved_model={resolved_model} upstream_body_preview={prev} elapsed_ms={elapsed_ms}"
            )
            raise HTTPException(
                status_code=502,
                detail={
                    "error": "upstream_image_download_failed",
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
        return raw_bytes, mime_type, "url_downloaded"

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    print(
        f"[AI_PROXY_IMAGE_ERROR] request_id={request_id} "
        f"error_type=missing_image_payload requested_model={requested_model_log} "
        f"resolved_model={resolved_model} elapsed_ms={elapsed_ms}"
    )
    raise HTTPException(status_code=502, detail="missing_image_payload")


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


@app.post(
    "/images/generations",
    response_model=Union[ImageGenerationB64Response, ImageGenerationR2UrlResponse],
)
async def images_generations(
    body: ImageGenerationRequest,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> Union[ImageGenerationB64Response, ImageGenerationR2UrlResponse]:
    """Forward POST /images/generations upstream.

    - ``response_format=url|b64_json``: return ``data[0].b64_json`` + ``mime_type`` (URLs fetched on Railway).
    - ``response_format=r2_url``: ask upstream for ``b64_json``, resolve bytes (download URL if needed),
      upload to R2, return only ``data[0].url`` + ``mime_type`` + ``storage=r2``.
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
    if fmt not in ("url", "b64_json", "r2_url"):
        raise HTTPException(
            status_code=400,
            detail="response_format must be 'url', 'b64_json', or 'r2_url'",
        )

    upstream_fmt: str = "b64_json" if fmt == "r2_url" else fmt

    payload: dict[str, Any] = {
        "model": resolved_model,
        "prompt": body.prompt,
        "n": 1,
        "response_format": upstream_fmt,
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
        f"upstream_url={upstream_url} response_format={fmt} upstream_response_format={upstream_fmt} "
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

            raw_bytes, mime_type, b64_source = await extract_image_bytes_from_upstream_item(
                client,
                first,
                request_id=request_id,
                requested_model_log=requested_model_log,
                resolved_model=resolved_model,
                start=start,
            )

            if fmt == "r2_url":
                r2_endpoint, r2_access, r2_secret, r2_bucket, r2_public = load_r2_settings()
                if not all((r2_endpoint, r2_access, r2_secret, r2_bucket, r2_public)):
                    elapsed_ms = int((time.perf_counter() - start) * 1000)
                    print(
                        f"[AI_PROXY_IMAGE_ERROR] request_id={request_id} "
                        f"error_type=r2_not_configured elapsed_ms={elapsed_ms}"
                    )
                    raise HTTPException(
                        status_code=500,
                        detail="r2_not_configured: set R2_ENDPOINT, R2_ACCESS_KEY, R2_SECRET_KEY, "
                        "R2_BUCKET_NAME, R2_PUBLIC_BASE_URL",
                    )

                storage_key = build_r2_image_object_key(
                    project_id=body.project_id,
                    target_type=body.target_type,
                    target_id=body.target_id,
                    request_id=request_id,
                    mime_type=mime_type,
                )

                tt_log = body.target_type or ""
                pid_log = body.project_id if body.project_id is not None else 0
                tid_log = body.target_id if body.target_id is not None else 0

                print(
                    f"[AI_PROXY_IMAGE_R2_UPLOAD_STARTED] request_id={request_id} "
                    f"project_id={pid_log} target_type={tt_log} target_id={tid_log} "
                    f"bytes_len={len(raw_bytes)} mime_type={mime_type}"
                )

                try:
                    await asyncio.to_thread(
                        upload_bytes_to_r2_sync,
                        object_key=storage_key,
                        data=raw_bytes,
                        content_type=mime_type,
                    )
                except ClientError as exc:
                    elapsed_ms = int((time.perf_counter() - start) * 1000)
                    em = truncate_for_log(str(exc))
                    print(
                        f"[AI_PROXY_IMAGE_ERROR] request_id={request_id} "
                        f"error_type=r2_upload_failed requested_model={requested_model_log} "
                        f"resolved_model={resolved_model} message={em} elapsed_ms={elapsed_ms}"
                    )
                    raise HTTPException(status_code=502, detail="r2_upload_failed") from exc
                except Exception as exc:
                    elapsed_ms = int((time.perf_counter() - start) * 1000)
                    print(
                        f"[AI_PROXY_IMAGE_ERROR] request_id={request_id} "
                        f"error_type=r2_upload_failed requested_model={requested_model_log} "
                        f"resolved_model={resolved_model} message={truncate_for_log(str(exc))} "
                        f"elapsed_ms={elapsed_ms}"
                    )
                    raise HTTPException(status_code=502, detail="r2_upload_failed") from exc

                public_url = f"{r2_public}/{quote(storage_key, safe='/')}"

                print(
                    f"[AI_PROXY_IMAGE_R2_UPLOAD_SUCCESS] request_id={request_id} "
                    f"url={public_url} storage_key={storage_key} bytes_len={len(raw_bytes)}"
                )

                elapsed_ms = int((time.perf_counter() - start) * 1000)
                print(
                    f"[AI_PROXY_IMAGE_RESPONSE] request_id={request_id} success=true format=r2_url "
                    f"storage=r2 resolved_model={resolved_model} elapsed_ms={elapsed_ms}"
                )
                return ImageGenerationR2UrlResponse(
                    data=[ImageGenerationDataUrlItem(url=public_url)],
                    mime_type=mime_type,
                    storage="r2",
                )

            final_b64 = base64.standard_b64encode(raw_bytes).decode("ascii")
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
