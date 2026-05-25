import asyncio
import base64
import json
import os
import time
import uuid
from typing import Annotated, Any
from urllib.parse import quote

import httpx
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from r2_upload import load_r2_settings, upload_bytes_to_r2
from xai_video_upstream import generate_xai_video_sync
from gemini_veo_upstream import generate_gemini_veo_video_sync

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
    provider: str | None = Field(default=None, description="Optional upstream provider hint: xai | gemini")
    response_format: str = Field(
        default="url",
        description="Client: url | b64_json | r2_url (r2_url rehosts to R2; upstream only receives url / b64_json)",
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


class ImageGenerationResponse(BaseModel):
    url: str | None = None
    image_url: str | None = None
    b64_json: str | None = None
    storage: str | None = None
    response_format: str | None = None


class XaiVideoGenerationRequest(BaseModel):
    project_id: int
    segment_id: str
    prompt: str
    reference_image_urls: list[str] = Field(default_factory=list)
    duration_seconds: int = Field(ge=1, le=60)
    aspect_ratio: str = "9:16"
    resolution: str | None = None
    model: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class XaiVideoGenerationResponse(BaseModel):
    ok: bool
    provider: str = "xai"
    model: str = ""
    request_id: str = ""
    video_url: str | None = None
    xai_video_url: str | None = None
    gemini_video_uri: str | None = None
    storage: str | None = None
    r2_key: str | None = None
    duration_seconds: int | None = None
    error_code: str | None = None
    error_message: str | None = None


class VideoUnderstandingRequest(BaseModel):
    video_url: str
    mime_type: str = "video/mp4"
    system_prompt: str
    user_payload: dict[str, Any] = Field(default_factory=dict)
    provider: str | None = None
    model: str | None = None
    service_name: str = "reference_video_understanding"


class VideoUnderstandingResponse(BaseModel):
    ok: bool
    provider: str = "gemini"
    model: str = ""
    request_id: str = ""
    raw_text: str | None = None
    analysis_json: dict[str, Any] | None = None
    error_code: str | None = None
    error_message: str | None = None


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


def resolve_gemini_understanding_model(body_model: str | None) -> tuple[str, str]:
    if body_model is not None:
        m = str(body_model).strip()
        if m:
            return m, "request_body"
    for key in ("GEMINI_UNDERSTANDING_MODEL", "GEMINI_VISION_MODEL", "GEMINI_TEXT_MODEL"):
        v = get_env(key)
        if v:
            return v.strip(), key
    return "gemini-2.5-flash", "default_gemini_2_5_flash"


def effective_gemini_understanding_base_url() -> str:
    return (
        get_env("GEMINI_UNDERSTANDING_BASE_URL")
        or get_env("GEMINI_BASE_URL")
        or get_env("GEMINI_API_URL")
        or "https://generativelanguage.googleapis.com/v1beta"
    ).rstrip("/")


def effective_gemini_understanding_timeout_seconds() -> float:
    raw = (
        get_env("GEMINI_UNDERSTANDING_TIMEOUT_SECONDS")
        or get_env("GEMINI_TIMEOUT_SECONDS")
        or get_env("REQUEST_TIMEOUT_SECONDS")
        or "300"
    )
    try:
        return max(30.0, float(raw))
    except ValueError:
        return 300.0


def max_video_understanding_download_bytes() -> int:
    raw = get_env("GEMINI_UNDERSTANDING_MAX_DOWNLOAD_BYTES") or get_env("MAX_VIDEO_DOWNLOAD_BYTES") or str(512 * 1024 * 1024)
    try:
        return max(1_000_000, int(raw))
    except ValueError:
        return 512 * 1024 * 1024


async def download_video_for_understanding(
    *,
    video_url: str,
    request_id: str,
    timeout_seconds: float,
) -> tuple[bytes, str | None]:
    max_bytes = max_video_understanding_download_bytes()
    timeout = httpx.Timeout(timeout_seconds, connect=min(30.0, timeout_seconds))
    started = time.perf_counter()
    print(
        f"[GEMINI_VIDEO_UNDERSTANDING_DOWNLOAD_START] request_id={request_id} "
        f"max_bytes={max_bytes}",
        flush=True,
    )
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            async with client.stream("GET", video_url) as resp:
                if resp.status_code >= 400:
                    body = await resp.aread()
                    print(
                        f"[GEMINI_VIDEO_UNDERSTANDING_DOWNLOAD_ERROR] request_id={request_id} "
                        f"status_code={resp.status_code} body={truncate_for_log(body.decode('utf-8', errors='ignore'), 500)}",
                        flush=True,
                    )
                    raise HTTPException(status_code=502, detail="video_download_failed")
                content_length = resp.headers.get("content-length")
                if content_length:
                    try:
                        if int(content_length) > max_bytes:
                            raise HTTPException(status_code=413, detail="video_too_large_for_understanding")
                    except ValueError:
                        pass
                chunks: list[bytes] = []
                total = 0
                async for chunk in resp.aiter_bytes():
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > max_bytes:
                        raise HTTPException(status_code=413, detail="video_too_large_for_understanding")
                    chunks.append(chunk)
                raw = b"".join(chunks)
                if not raw:
                    raise HTTPException(status_code=502, detail="video_download_empty")
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                print(
                    f"[GEMINI_VIDEO_UNDERSTANDING_DOWNLOAD_SUCCESS] request_id={request_id} "
                    f"bytes={len(raw)} content_type={resp.headers.get('content-type') or ''} elapsed_ms={elapsed_ms}",
                    flush=True,
                )
                return raw, resp.headers.get("content-type")
    except HTTPException:
        raise
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="video_download_timeout") from None
    except httpx.RequestError as exc:
        print(
            f"[GEMINI_VIDEO_UNDERSTANDING_DOWNLOAD_ERROR] request_id={request_id} "
            f"error_type=network_error message={truncate_for_log(str(exc))}",
            flush=True,
        )
        raise HTTPException(status_code=502, detail="video_download_network_error") from exc


def normalize_video_mime(request_mime: str, downloaded_content_type: str | None) -> str:
    requested = _mime_from_content_type(request_mime)
    if requested and requested.startswith("video/"):
        return requested
    downloaded = _mime_from_content_type(downloaded_content_type)
    if downloaded and downloaded.startswith("video/"):
        return downloaded
    return "video/mp4"


async def upload_video_to_gemini_file_api(
    *,
    video_bytes: bytes,
    mime_type: str,
    request_id: str,
    api_key: str,
    display_name: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    upload_start_url = "https://generativelanguage.googleapis.com/upload/v1beta/files"
    timeout = httpx.Timeout(timeout_seconds, connect=min(30.0, timeout_seconds))
    headers = {
        "x-goog-api-key": api_key,
        "X-Goog-Upload-Protocol": "resumable",
        "X-Goog-Upload-Command": "start",
        "X-Goog-Upload-Header-Content-Length": str(len(video_bytes)),
        "X-Goog-Upload-Header-Content-Type": mime_type,
        "Content-Type": "application/json",
    }
    started = time.perf_counter()
    print(
        f"[GEMINI_FILE_UPLOAD_START] request_id={request_id} bytes={len(video_bytes)} mime_type={mime_type}",
        flush=True,
    )
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            start_resp = await client.post(
                upload_start_url,
                headers=headers,
                json={"file": {"display_name": display_name}},
            )
            if start_resp.status_code >= 400:
                print(
                    f"[GEMINI_FILE_UPLOAD_ERROR] request_id={request_id} step=start "
                    f"status_code={start_resp.status_code} body={truncate_for_log(start_resp.text, 1000)}",
                    flush=True,
                )
                raise HTTPException(
                    status_code=502,
                    detail={"error": "gemini_file_upload_start_failed", "status_code": start_resp.status_code, "body": start_resp.text[:1000]},
                )
            upload_url = start_resp.headers.get("x-goog-upload-url") or start_resp.headers.get("X-Goog-Upload-URL")
            if not upload_url:
                raise HTTPException(status_code=502, detail="gemini_file_upload_url_missing")

            upload_resp = await client.post(
                upload_url,
                headers={
                    "Content-Length": str(len(video_bytes)),
                    "X-Goog-Upload-Offset": "0",
                    "X-Goog-Upload-Command": "upload, finalize",
                    "Content-Type": mime_type,
                },
                content=video_bytes,
            )
            if upload_resp.status_code >= 400:
                print(
                    f"[GEMINI_FILE_UPLOAD_ERROR] request_id={request_id} step=finalize "
                    f"status_code={upload_resp.status_code} body={truncate_for_log(upload_resp.text, 1000)}",
                    flush=True,
                )
                raise HTTPException(
                    status_code=502,
                    detail={"error": "gemini_file_upload_finalize_failed", "status_code": upload_resp.status_code, "body": upload_resp.text[:1000]},
                )
            try:
                data = upload_resp.json()
            except json.JSONDecodeError:
                raise HTTPException(status_code=502, detail="gemini_file_upload_invalid_json") from None
    except HTTPException:
        raise
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="gemini_file_upload_timeout") from None
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail="gemini_file_upload_network_error") from exc

    file_obj = data.get("file") if isinstance(data, dict) else None
    if not isinstance(file_obj, dict):
        raise HTTPException(status_code=502, detail="gemini_file_upload_missing_file")
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    print(
        f"[GEMINI_FILE_UPLOAD_SUCCESS] request_id={request_id} "
        f"name={file_obj.get('name') or ''} uri_present={bool(file_obj.get('uri'))} "
        f"state={file_obj.get('state') or ''} elapsed_ms={elapsed_ms}",
        flush=True,
    )
    return file_obj


async def wait_for_gemini_file_active(
    *,
    file_obj: dict[str, Any],
    api_key: str,
    request_id: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    name = str(file_obj.get("name") or "").strip()
    state = str(file_obj.get("state") or "").strip().upper()
    if state == "ACTIVE" or not name:
        return file_obj
    deadline = time.monotonic() + min(max(30.0, timeout_seconds), 300.0)
    get_url = f"https://generativelanguage.googleapis.com/v1beta/{name}"
    timeout = httpx.Timeout(30.0)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        while time.monotonic() < deadline:
            await asyncio.sleep(2.0)
            resp = await client.get(get_url, params={"key": api_key})
            if resp.status_code >= 400:
                print(
                    f"[GEMINI_FILE_POLL_ERROR] request_id={request_id} status_code={resp.status_code} "
                    f"body={truncate_for_log(resp.text, 500)}",
                    flush=True,
                )
                raise HTTPException(status_code=502, detail="gemini_file_poll_failed")
            try:
                current = resp.json()
            except json.JSONDecodeError:
                raise HTTPException(status_code=502, detail="gemini_file_poll_invalid_json") from None
            if not isinstance(current, dict):
                raise HTTPException(status_code=502, detail="gemini_file_poll_invalid_response")
            state = str(current.get("state") or "").strip().upper()
            print(
                f"[GEMINI_FILE_POLL] request_id={request_id} name={name} state={state}",
                flush=True,
            )
            if state == "ACTIVE":
                return current
            if state == "FAILED":
                raise HTTPException(status_code=502, detail="gemini_file_processing_failed")
    raise HTTPException(status_code=504, detail="gemini_file_processing_timeout")


def truncate_for_log(text: str, max_len: int = 500) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "...(truncated)"


def try_parse_json_object(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(raw[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


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


def resolve_upstream_image_response_format(client_response_format: str) -> str:
    """Map client response_format to what OpenAI-compatible image APIs accept upstream (url | b64_json only)."""
    if client_response_format == "r2_url":
        raw = (get_env("AI_PROXY_IMAGE_UPSTREAM_FORMAT_FOR_R2") or "url").strip().lower()
        if raw in ("url", "b64_json"):
            return raw
        return "url"
    return client_response_format


def image_request_is_gemini(body: ImageGenerationRequest, resolved_model: str) -> bool:
    provider = (body.provider or "").strip().lower()
    model = (resolved_model or "").strip().lower()
    return provider == "gemini" or model.startswith("gemini-")


def effective_gemini_image_base_url() -> str:
    return (
        get_env("GEMINI_IMAGE_BASE_URL")
        or get_env("GEMINI_BASE_URL")
        or get_env("GEMINI_API_URL")
        or "https://generativelanguage.googleapis.com/v1beta"
    ).rstrip("/")


def effective_gemini_image_timeout_seconds() -> float:
    raw = (
        get_env("GEMINI_IMAGE_TIMEOUT_SECONDS")
        or get_env("GEMINI_TIMEOUT_SECONDS")
        or get_env("REQUEST_TIMEOUT_SECONDS")
        or "180"
    )
    try:
        return max(5.0, float(raw))
    except ValueError:
        return 180.0


def extract_first_gemini_image(data: dict[str, Any]) -> tuple[bytes, str]:
    for cand in data.get("candidates") or []:
        if not isinstance(cand, dict):
            continue
        content = cand.get("content") or {}
        if not isinstance(content, dict):
            continue
        for part in content.get("parts") or []:
            if not isinstance(part, dict):
                continue
            inline = part.get("inlineData") or part.get("inline_data")
            if not isinstance(inline, dict):
                continue
            b64 = inline.get("data")
            if not isinstance(b64, str) or not b64.strip():
                continue
            try:
                raw = base64.b64decode(b64.strip(), validate=False)
            except Exception:
                raise HTTPException(status_code=502, detail="invalid_gemini_b64_image") from None
            if not raw:
                raise HTTPException(status_code=502, detail="empty_gemini_image")
            mime = inline.get("mimeType") or inline.get("mime_type") or _resolve_mime_from_bytes(raw, None)
            return raw, str(mime)
    raise HTTPException(status_code=502, detail="missing_gemini_image")


def extension_for_mime_image(mime_type: str) -> str:
    m = (mime_type or "").strip().lower().split(";")[0].strip()
    return {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/gif": ".gif",
    }.get(m, ".bin")


def sanitize_image_target_path_segment(value: str | None, max_len: int = 200) -> str:
    raw = (value or "").strip().replace("/", "_").replace("\\", "_")
    return raw[:max_len] if raw else "unknown"


def build_image_r2_object_key(
    *,
    project_id: int | None,
    target_type: str | None,
    target_id: int | None,
    proxy_request_id: str,
    mime_type: str,
) -> str:
    ext = extension_for_mime_image(mime_type)
    pid = project_id if project_id is not None else 0
    tid = target_id if target_id is not None else 0
    seg = sanitize_image_target_path_segment(target_type)
    return f"short-drama/assets/{pid}/{seg}/{tid}/{proxy_request_id}{ext}"


async def extract_image_bytes_from_upstream(
    first: dict[str, Any],
    upstream_fmt: str,
    client: httpx.AsyncClient,
) -> tuple[bytes, str]:
    """Normalize upstream image item to bytes + MIME (download URL or decode base64)."""

    b64_val = first.get("b64_json")
    url_val = first.get("url")

    if upstream_fmt == "b64_json" and isinstance(b64_val, str) and b64_val.strip():
        try:
            raw = base64.b64decode(b64_val.strip(), validate=False)
        except Exception:
            raise HTTPException(status_code=502, detail="invalid_b64_json") from None
        mime = _resolve_mime_from_bytes(raw, None)
        return raw, mime

    if isinstance(url_val, str) and url_val.strip():
        u = url_val.strip()
        dl = await client.get(u, follow_redirects=True)
        if dl.status_code >= 400:
            raise HTTPException(status_code=502, detail="image_download_failed")
        raw = dl.content
        if len(raw) > _MAX_IMAGE_DOWNLOAD_BYTES:
            raise HTTPException(status_code=502, detail="image_too_large")
        mime = _resolve_mime_from_bytes(raw, dl.headers.get("content-type"))
        return raw, mime

    if isinstance(b64_val, str) and b64_val.strip():
        try:
            raw = base64.b64decode(b64_val.strip(), validate=False)
        except Exception:
            raise HTTPException(status_code=502, detail="invalid_b64_json") from None
        mime = _resolve_mime_from_bytes(raw, None)
        return raw, mime

    raise HTTPException(status_code=502, detail="missing_image_payload")


async def generate_gemini_image_response(
    *,
    body: ImageGenerationRequest,
    request_id: str,
    resolved_model: str,
    requested_model_log: str,
    client_fmt: str,
) -> ImageGenerationResponse:
    api_key = get_env("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="gemini_api_key_not_configured")

    base_url = effective_gemini_image_base_url()
    timeout_seconds = effective_gemini_image_timeout_seconds()
    endpoint = f"{base_url}/models/{resolved_model}:generateContent"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": body.prompt}]}],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"],
            "temperature": 0.9,
        },
    }
    storage_target = "r2" if client_fmt == "r2_url" else "direct"
    print(
        f"[GEMINI_IMAGE_REQUEST] request_id={request_id} requested_model={requested_model_log} "
        f"resolved_model={resolved_model} endpoint={endpoint} client_response_format={client_fmt} "
        f"storage_target={storage_target} project_id={body.project_id} "
        f"target_type={body.target_type or ''} target_id={body.target_id} timeout_seconds={timeout_seconds}"
    )
    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds)) as client:
            resp = await client.post(endpoint, params={"key": api_key}, json=payload)
    except httpx.TimeoutException:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        print(
            f"[GEMINI_IMAGE_ERROR] request_id={request_id} error_type=upstream_timeout "
            f"resolved_model={resolved_model} elapsed_ms={elapsed_ms}"
        )
        raise HTTPException(status_code=504, detail="gemini_upstream_timeout")
    except httpx.RequestError as exc:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        print(
            f"[GEMINI_IMAGE_ERROR] request_id={request_id} error_type=network_error "
            f"resolved_model={resolved_model} message={truncate_for_log(str(exc))} elapsed_ms={elapsed_ms}"
        )
        raise HTTPException(status_code=502, detail="gemini_network_error")

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    if resp.status_code >= 400:
        body_preview = truncate_for_log(resp.text[:800], max_len=800)
        print(
            f"[GEMINI_IMAGE_ERROR] request_id={request_id} error_type=upstream_http "
            f"upstream_status_code={resp.status_code} resolved_model={resolved_model} "
            f"upstream_body_preview={body_preview} elapsed_ms={elapsed_ms}"
        )
        raise HTTPException(
            status_code=502,
            detail={"error": "gemini_upstream_error", "status_code": resp.status_code, "body": resp.text[:800]},
        )

    try:
        data = resp.json()
    except json.JSONDecodeError as exc:
        print(
            f"[GEMINI_IMAGE_ERROR] request_id={request_id} error_type=json_decode_error "
            f"resolved_model={resolved_model} message={truncate_for_log(str(exc))} elapsed_ms={elapsed_ms}"
        )
        raise HTTPException(status_code=502, detail="gemini_invalid_json")

    raw_bytes, mime_type = extract_first_gemini_image(data)
    b64 = base64.b64encode(raw_bytes).decode("ascii")

    if client_fmt == "r2_url":
        r2s = load_r2_settings()
        if r2s is None:
            print(
                f"[GEMINI_IMAGE_ERROR] request_id={request_id} error_type=r2_not_configured "
                f"resolved_model={resolved_model} elapsed_ms={elapsed_ms}"
            )
            raise HTTPException(
                status_code=500,
                detail=(
                    "r2_not_configured: set R2_ENDPOINT or R2_ACCOUNT_ID, "
                    "R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME, R2_PUBLIC_BASE_URL"
                ),
            )
        object_key = build_image_r2_object_key(
            project_id=body.project_id,
            target_type=body.target_type,
            target_id=body.target_id,
            proxy_request_id=request_id,
            mime_type=mime_type,
        )
        await asyncio.to_thread(
            upload_bytes_to_r2,
            object_key=object_key,
            data=raw_bytes,
            content_type=mime_type,
        )
        public_url = f"{r2s.public_base_url}/{quote(object_key, safe='/')}"
        print(
            f"[GEMINI_IMAGE_RESPONSE] request_id={request_id} success=true storage=r2 "
            f"r2_url_present=true resolved_model={resolved_model} image_bytes={len(raw_bytes)} "
            f"mime={mime_type} elapsed_ms={elapsed_ms}"
        )
        return ImageGenerationResponse(
            url=public_url,
            image_url=public_url,
            storage="r2",
            response_format="r2_url",
        )

    print(
        f"[GEMINI_IMAGE_RESPONSE] request_id={request_id} success=true storage=direct "
        f"resolved_model={resolved_model} image_bytes={len(raw_bytes)} mime={mime_type} elapsed_ms={elapsed_ms}"
    )
    return ImageGenerationResponse(b64_json=b64, response_format="b64_json")


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
    proxy_token = get_env("PROXY_AUTH_TOKEN") or get_env("RAILWAY_XAI_VIDEO_PROXY_TOKEN")
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


@app.post("/api/xai/videos/generations", response_model=XaiVideoGenerationResponse)
async def xai_videos_generations(
    body: XaiVideoGenerationRequest,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> XaiVideoGenerationResponse:
    """Aliyun backend → Railway → xAI video (submit + poll); returns public video_url."""
    require_proxy_auth(authorization)
    result = generate_xai_video_sync(
        project_id=body.project_id,
        segment_id=body.segment_id,
        prompt=body.prompt,
        reference_image_urls=body.reference_image_urls,
        duration_seconds=body.duration_seconds,
        aspect_ratio=body.aspect_ratio,
        resolution=body.resolution,
        model=body.model,
    )
    return XaiVideoGenerationResponse(**result)


@app.post("/api/gemini/videos/generations", response_model=XaiVideoGenerationResponse)
async def gemini_videos_generations(
    body: XaiVideoGenerationRequest,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> XaiVideoGenerationResponse:
    """Aliyun backend -> Railway -> Google Gemini Veo video; returns public R2 video_url."""
    require_proxy_auth(authorization)
    result = generate_gemini_veo_video_sync(
        project_id=body.project_id,
        segment_id=body.segment_id,
        prompt=body.prompt,
        reference_image_urls=body.reference_image_urls,
        duration_seconds=body.duration_seconds,
        aspect_ratio=body.aspect_ratio,
        resolution=body.resolution,
        model=body.model,
    )
    return XaiVideoGenerationResponse(**result)


async def run_gemini_video_understanding(
    body: VideoUnderstandingRequest,
    request_id: str,
) -> VideoUnderstandingResponse:
    api_key = get_env("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="gemini_api_key_not_configured")

    model, model_source = resolve_gemini_understanding_model(body.model)
    base_url = effective_gemini_understanding_base_url()
    timeout_seconds = effective_gemini_understanding_timeout_seconds()
    endpoint = f"{base_url}/models/{model}:generateContent"
    video_url = (body.video_url or "").strip()
    if not video_url:
        raise HTTPException(status_code=400, detail="video_url_required")

    video_bytes, downloaded_content_type = await download_video_for_understanding(
        video_url=video_url,
        request_id=request_id,
        timeout_seconds=timeout_seconds,
    )
    effective_mime_type = normalize_video_mime(body.mime_type or "", downloaded_content_type)
    file_obj = await upload_video_to_gemini_file_api(
        video_bytes=video_bytes,
        mime_type=effective_mime_type,
        request_id=request_id,
        api_key=api_key,
        display_name=f"reference-video-{request_id}",
        timeout_seconds=timeout_seconds,
    )
    file_obj = await wait_for_gemini_file_active(
        file_obj=file_obj,
        api_key=api_key,
        request_id=request_id,
        timeout_seconds=timeout_seconds,
    )
    file_uri = str(file_obj.get("uri") or "").strip()
    if not file_uri:
        raise HTTPException(status_code=502, detail="gemini_file_uri_missing")
    effective_mime_type = str(file_obj.get("mimeType") or effective_mime_type).strip() or effective_mime_type

    user_payload_text = json.dumps(body.user_payload or {}, ensure_ascii=False, default=str)
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"fileData": {"mimeType": effective_mime_type, "fileUri": file_uri}},
                    {"text": user_payload_text},
                ],
            }
        ],
        "systemInstruction": {
            "parts": [{"text": body.system_prompt}],
        },
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json",
        },
    }

    print(
        f"[GEMINI_VIDEO_UNDERSTANDING_REQUEST] request_id={request_id} model={model} "
        f"model_source={model_source} endpoint={endpoint} mime_type={effective_mime_type} "
        f"file_name={file_obj.get('name') or ''} "
        f"service_name={body.service_name} payload_chars={len(user_payload_text)} "
        f"timeout_seconds={timeout_seconds}",
        flush=True,
    )
    start = time.perf_counter()
    timeout = httpx.Timeout(timeout_seconds, connect=min(30.0, timeout_seconds))
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.post(endpoint, params={"key": api_key}, json=payload)
    except httpx.TimeoutException:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        print(
            f"[GEMINI_VIDEO_UNDERSTANDING_ERROR] request_id={request_id} error_type=upstream_timeout "
            f"model={model} elapsed_ms={elapsed_ms}",
            flush=True,
        )
        raise HTTPException(status_code=504, detail="gemini_video_understanding_timeout")
    except httpx.RequestError as exc:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        print(
            f"[GEMINI_VIDEO_UNDERSTANDING_ERROR] request_id={request_id} error_type=network_error "
            f"model={model} message={truncate_for_log(str(exc))} elapsed_ms={elapsed_ms}",
            flush=True,
        )
        raise HTTPException(status_code=502, detail="gemini_video_understanding_network_error")

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    if resp.status_code >= 400:
        body_preview = truncate_for_log(resp.text[:1200], max_len=1200)
        print(
            f"[GEMINI_VIDEO_UNDERSTANDING_ERROR] request_id={request_id} error_type=upstream_http "
            f"upstream_status_code={resp.status_code} model={model} body={body_preview} elapsed_ms={elapsed_ms}",
            flush=True,
        )
        raise HTTPException(
            status_code=502,
            detail={"error": "gemini_video_understanding_upstream_error", "status_code": resp.status_code, "body": resp.text[:1200]},
        )

    try:
        data = resp.json()
    except json.JSONDecodeError as exc:
        print(
            f"[GEMINI_VIDEO_UNDERSTANDING_ERROR] request_id={request_id} error_type=json_decode_error "
            f"model={model} message={truncate_for_log(str(exc))} elapsed_ms={elapsed_ms}",
            flush=True,
        )
        raise HTTPException(status_code=502, detail="gemini_video_understanding_invalid_json")

    raw_text = ""
    for cand in data.get("candidates") or []:
        if not isinstance(cand, dict):
            continue
        content = cand.get("content") or {}
        if not isinstance(content, dict):
            continue
        for part in content.get("parts") or []:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                raw_text += part["text"]
    raw_text = raw_text.strip()
    if not raw_text:
        print(
            f"[GEMINI_VIDEO_UNDERSTANDING_ERROR] request_id={request_id} error_type=empty_output "
            f"model={model} elapsed_ms={elapsed_ms}",
            flush=True,
        )
        raise HTTPException(status_code=502, detail="gemini_video_understanding_empty_output")

    analysis_json = try_parse_json_object(raw_text)
    if analysis_json is None:
        print(
            f"[GEMINI_VIDEO_UNDERSTANDING_ERROR] request_id={request_id} error_type=json_parse_failed "
            f"model={model} raw_text={truncate_for_log(raw_text, 800)} elapsed_ms={elapsed_ms}",
            flush=True,
        )
        raise HTTPException(status_code=502, detail="gemini_video_understanding_json_parse_failed")

    print(
        f"[GEMINI_VIDEO_UNDERSTANDING_RESPONSE] request_id={request_id} success=true "
        f"model={model} raw_length={len(raw_text)} elapsed_ms={elapsed_ms}",
        flush=True,
    )
    return VideoUnderstandingResponse(
        ok=True,
        provider="gemini",
        model=model,
        request_id=request_id,
        raw_text=raw_text,
        analysis_json=analysis_json,
    )


@app.post("/api/gemini/video-understanding", response_model=VideoUnderstandingResponse)
@app.post("/api/gemini/videos/understanding", response_model=VideoUnderstandingResponse)
@app.post("/api/video/understanding", response_model=VideoUnderstandingResponse)
@app.post("/video/understanding", response_model=VideoUnderstandingResponse)
async def gemini_video_understanding(
    body: VideoUnderstandingRequest,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> VideoUnderstandingResponse:
    """VibeClip backend -> Railway -> Gemini video understanding; returns strict analysis JSON."""
    require_proxy_auth(authorization)
    return await run_gemini_video_understanding(body, str(uuid.uuid4()))


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

    requested_raw = (body.model or "").strip() if body.model is not None else ""
    requested_model_log = requested_raw if requested_raw else "(none)"
    resolved_model, model_source_env = resolve_image_generation_model(body.model)
    client_fmt = (body.response_format or "url").strip().lower()
    if client_fmt not in ("url", "b64_json", "r2_url"):
        raise HTTPException(
            status_code=400,
            detail="invalid_response_format: expected url, b64_json, or r2_url",
        )

    if image_request_is_gemini(body, resolved_model):
        return await generate_gemini_image_response(
            body=body,
            request_id=request_id,
            resolved_model=resolved_model,
            requested_model_log=requested_model_log,
            client_fmt=client_fmt,
        )

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key is None or openai_key.strip() == "":
        raise HTTPException(status_code=500, detail="openai_api_key_not_configured")

    base_url = get_env("OPENAI_BASE_URL", "https://api.openai.com/v1") or "https://api.openai.com/v1"
    provider_label = infer_upstream_provider_label(base_url)
    timeout_raw = get_env("REQUEST_TIMEOUT_SECONDS", "120") or "120"
    try:
        timeout_seconds = float(timeout_raw)
    except ValueError:
        timeout_seconds = 120.0

    timeout = httpx.Timeout(timeout_seconds)
    wants_r2 = client_fmt == "r2_url"
    upstream_fmt = resolve_upstream_image_response_format(client_fmt)
    target_response_format = client_fmt

    print(
        f"[IMAGE_RESPONSE_FORMAT_RESOLVED] "
        f"client_response_format={client_fmt} "
        f"upstream_response_format={upstream_fmt} "
        f"target_response_format={target_response_format} "
        f"project_id={body.project_id} "
        f"target_type={body.target_type or ''} "
        f"target_id={body.target_id}"
    )

    if upstream_fmt == "r2_url":
        print(
            f"[AI_PROXY_IMAGE_ERROR] request_id={request_id} "
            f"error_type=upstream_format_invariant_violation "
            f"message={truncate_for_log('upstream_response_format must be url or b64_json, never r2_url')}"
        )
        raise HTTPException(
            status_code=500,
            detail="internal_error: upstream_response_format_invariant_violation",
        )

    storage_target = "r2" if wants_r2 else "direct"

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
        f"upstream_url={upstream_url} "
        f"client_response_format={client_fmt} upstream_response_format={upstream_fmt} "
        f"storage_target={storage_target} "
        f"project_id={body.project_id} target_type={body.target_type or ''} target_id={body.target_id} "
        f"timeout_seconds={timeout_seconds}"
    )

    start = time.perf_counter()

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            print(
                f"[UPSTREAM_IMAGE_REQUEST_FORMAT] "
                f"upstream_response_format={upstream_fmt} "
                f"model={resolved_model}"
            )
            resp = await client.post(upstream_url, headers=headers, json=payload)

            if resp.status_code >= 400:
                elapsed_ms = int((time.perf_counter() - start) * 1000)
                body_preview = truncate_for_log(resp.text[:800], max_len=800)
                print(
                    f"[AI_PROXY_IMAGE_ERROR] request_id={request_id} "
                    f"error_type=upstream_http upstream_status_code={resp.status_code} "
                    f"requested_model={requested_model_log} resolved_model={resolved_model} "
                    f"client_response_format={client_fmt} upstream_response_format={upstream_fmt} "
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

            if wants_r2:
                r2s = load_r2_settings()
                if r2s is None:
                    elapsed_ms = int((time.perf_counter() - start) * 1000)
                    print(
                        f"[AI_PROXY_IMAGE_ERROR] request_id={request_id} "
                        f"error_type=r2_not_configured requested_model={requested_model_log} "
                        f"resolved_model={resolved_model} elapsed_ms={elapsed_ms}"
                    )
                    raise HTTPException(
                        status_code=500,
                        detail=(
                            "r2_not_configured: set R2_ENDPOINT or R2_ACCOUNT_ID, "
                            "R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME, R2_PUBLIC_BASE_URL"
                        ),
                    )

                raw_bytes, mime_type = await extract_image_bytes_from_upstream(
                    first, upstream_fmt, client
                )
                object_key = build_image_r2_object_key(
                    project_id=body.project_id,
                    target_type=body.target_type,
                    target_id=body.target_id,
                    proxy_request_id=request_id,
                    mime_type=mime_type,
                )

                await asyncio.to_thread(
                    upload_bytes_to_r2,
                    object_key=object_key,
                    data=raw_bytes,
                    content_type=mime_type,
                )

                public_url = f"{r2s.public_base_url}/{quote(object_key, safe='/')}"
                elapsed_ms = int((time.perf_counter() - start) * 1000)
                print(
                    f"[AI_PROXY_IMAGE_RESPONSE] request_id={request_id} success=true "
                    f"client_response_format=r2_url upstream_response_format={upstream_fmt} "
                    f"storage=r2 r2_url_present=true resolved_model={resolved_model} elapsed_ms={elapsed_ms}"
                )
                return ImageGenerationResponse(
                    url=public_url,
                    image_url=public_url,
                    storage="r2",
                    response_format="r2_url",
                )

            if client_fmt == "b64_json":
                b64 = first.get("b64_json")
                if not isinstance(b64, str) or not b64.strip():
                    raise HTTPException(status_code=502, detail="missing_b64_json")
                elapsed_ms = int((time.perf_counter() - start) * 1000)
                print(
                    f"[AI_PROXY_IMAGE_RESPONSE] request_id={request_id} success=true "
                    f"client_response_format=b64_json upstream_response_format=b64_json "
                    f"storage=direct r2_url_present=false resolved_model={resolved_model} elapsed_ms={elapsed_ms}"
                )
                return ImageGenerationResponse(b64_json=b64.strip())

            url = first.get("url")
            if not isinstance(url, str) or not url.strip():
                raise HTTPException(status_code=502, detail="missing_image_url")

            elapsed_ms = int((time.perf_counter() - start) * 1000)
            print(
                f"[AI_PROXY_IMAGE_RESPONSE] request_id={request_id} success=true "
                f"client_response_format=url upstream_response_format=url "
                f"storage=direct r2_url_present=false resolved_model={resolved_model} elapsed_ms={elapsed_ms}"
            )
            return ImageGenerationResponse(url=url.strip())

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
