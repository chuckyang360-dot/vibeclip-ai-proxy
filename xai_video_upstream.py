"""xAI video generations (submit + poll) for Railway proxy — mirrors Aliyun xai_video_client."""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

_DEFAULT_XAI_VIDEO_MODEL = "grok-imagine-video"


def get_env(key: str, default: str | None = None) -> str | None:
    val = os.getenv(key)
    if val is None or str(val).strip() == "":
        return default
    return str(val).strip()


def resolve_xai_video_model(requested: str | None) -> str:
    if requested and str(requested).strip():
        return str(requested).strip()
    for key in ("XAI_VIDEO_MODEL", "XAI_MODEL"):
        v = get_env(key)
        if v:
            return v
    return _DEFAULT_XAI_VIDEO_MODEL


def resolve_xai_api_key() -> str:
    return get_env("XAI_API_KEY") or get_env("OPENAI_API_KEY") or ""


def resolve_xai_video_api_root() -> str:
    """Return base like https://api.x.ai (no trailing /v1)."""
    raw = (
        get_env("XAI_API_BASE")
        or get_env("XAI_VIDEO_BASE_URL")
        or get_env("OPENAI_BASE_URL")
        or "https://api.x.ai"
    )
    base = raw.rstrip("/")
    if base.endswith("/v1"):
        return base[: -len("/v1")]
    return base


def _timeout_seconds() -> float:
    raw = get_env("XAI_VIDEO_TIMEOUT_SECONDS") or get_env("REQUEST_TIMEOUT_SECONDS") or "600"
    try:
        return max(30.0, float(raw))
    except ValueError:
        return 600.0


def _poll_interval_seconds() -> float:
    raw = get_env("XAI_VIDEO_POLL_INTERVAL_SECONDS") or get_env("RAILWAY_XAI_VIDEO_PROXY_POLL_INTERVAL_SECONDS") or "3"
    try:
        return max(0.5, float(raw))
    except ValueError:
        return 3.0


def _poll_timeout_seconds() -> float:
    raw = get_env("XAI_VIDEO_POLL_TIMEOUT_SECONDS") or get_env("XAI_VIDEO_TIMEOUT_SECONDS") or "600"
    try:
        return max(30.0, float(raw))
    except ValueError:
        return 600.0


def _truncate(text: str, limit: int = 1000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def _headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _build_xai_payload(
    *,
    model: str,
    prompt: str,
    reference_image_urls: list[str],
    duration_seconds: int,
    aspect_ratio: str,
    resolution: str | None,
) -> dict[str, Any]:
    refs = [{"url": u} for u in reference_image_urls if (u or "").strip()]
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "reference_images": refs,
        "duration": int(duration_seconds),
        "aspect_ratio": aspect_ratio,
    }
    if resolution:
        payload["resolution"] = resolution
    return payload


def generate_xai_video_sync(
    *,
    project_id: int,
    segment_id: str,
    prompt: str,
    reference_image_urls: list[str],
    duration_seconds: int,
    aspect_ratio: str,
    resolution: str | None,
    model: str | None,
) -> dict[str, Any]:
    """
    Submit xAI video job, poll until done, return normalized dict:
    ok, provider, model, request_id, video_url, duration_seconds, error_code, error_message
    """
    api_key = resolve_xai_api_key()
    if not api_key:
        return {
            "ok": False,
            "provider": "xai",
            "model": resolve_xai_video_model(model),
            "error_code": "XAI_API_KEY_NOT_CONFIGURED",
            "error_message": "XAI_API_KEY is not configured on Railway proxy",
            "request_id": "",
        }

    resolved_model = resolve_xai_video_model(model)
    root = resolve_xai_video_api_root()
    submit_url = f"{root}/v1/videos/generations"
    timeout = httpx.Timeout(
        connect=min(30.0, _timeout_seconds()),
        read=_timeout_seconds(),
        write=_timeout_seconds(),
        pool=10.0,
    )

    payload = _build_xai_payload(
        model=resolved_model,
        prompt=prompt,
        reference_image_urls=reference_image_urls,
        duration_seconds=duration_seconds,
        aspect_ratio=aspect_ratio,
        resolution=resolution,
    )

    print(
        f"[RAILWAY_XAI_VIDEO_REQUEST] project_id={project_id} segment_id={segment_id} model={resolved_model} "
        f"prompt_chars={len(prompt or '')} reference_image_count={len(payload.get('reference_images') or [])} "
        f"duration_seconds={duration_seconds} aspect_ratio={aspect_ratio}"
    )
    print(
        f"[XAI_VIDEO_REQUEST] POST {submit_url} model={resolved_model} project_id={project_id} "
        f"segment_id={segment_id}"
    )

    try:
        with httpx.Client(timeout=timeout, http2=False, verify=True, follow_redirects=True) as client:
            resp = client.post(submit_url, headers=_headers(api_key), json=payload)
    except httpx.TimeoutException as exc:
        print(f"[XAI_VIDEO_FAILED] project_id={project_id} segment_id={segment_id} error=submit_timeout err={exc}")
        return {
            "ok": False,
            "provider": "xai",
            "model": resolved_model,
            "error_code": "XAI_VIDEO_SUBMIT_TIMEOUT",
            "error_message": str(exc),
            "request_id": "",
        }
    except httpx.RequestError as exc:
        print(f"[XAI_VIDEO_FAILED] project_id={project_id} segment_id={segment_id} error=submit_network err={exc}")
        return {
            "ok": False,
            "provider": "xai",
            "model": resolved_model,
            "error_code": "XAI_VIDEO_NETWORK_ERROR",
            "error_message": str(exc),
            "request_id": "",
        }

    if resp.status_code >= 400:
        body = _truncate(resp.text or "")
        print(
            f"[XAI_VIDEO_FAILED] project_id={project_id} segment_id={segment_id} "
            f"error=submit_http status={resp.status_code} body={body}"
        )
        return {
            "ok": False,
            "provider": "xai",
            "model": resolved_model,
            "error_code": "XAI_VIDEO_HTTP_ERROR",
            "error_message": f"HTTP {resp.status_code}: {body}",
            "request_id": "",
        }

    try:
        start_data = resp.json()
    except Exception as exc:
        return {
            "ok": False,
            "provider": "xai",
            "model": resolved_model,
            "error_code": "XAI_VIDEO_INVALID_JSON",
            "error_message": str(exc),
            "request_id": "",
        }

    request_id = str(start_data.get("request_id") or "")
    if not request_id:
        return {
            "ok": False,
            "provider": "xai",
            "model": resolved_model,
            "error_code": "XAI_VIDEO_MISSING_REQUEST_ID",
            "error_message": f"start response missing request_id: {_truncate(str(start_data))}",
            "request_id": "",
        }

    poll_url = f"{root}/v1/videos/{request_id}"
    deadline = time.monotonic() + _poll_timeout_seconds()
    interval = _poll_interval_seconds()

    while time.monotonic() < deadline:
        try:
            with httpx.Client(timeout=timeout, http2=False, verify=True, follow_redirects=True) as client:
                poll_resp = client.get(poll_url, headers=_headers(api_key))
        except httpx.TimeoutException as exc:
            return {
                "ok": False,
                "provider": "xai",
                "model": resolved_model,
                "error_code": "XAI_VIDEO_POLL_TIMEOUT",
                "error_message": str(exc),
                "request_id": request_id,
            }
        except httpx.RequestError as exc:
            return {
                "ok": False,
                "provider": "xai",
                "model": resolved_model,
                "error_code": "XAI_VIDEO_POLL_NETWORK_ERROR",
                "error_message": str(exc),
                "request_id": request_id,
            }

        if poll_resp.status_code >= 400:
            return {
                "ok": False,
                "provider": "xai",
                "model": resolved_model,
                "error_code": "XAI_VIDEO_POLL_HTTP_ERROR",
                "error_message": f"HTTP {poll_resp.status_code}: {_truncate(poll_resp.text or '')}",
                "request_id": request_id,
            }

        try:
            data = poll_resp.json()
        except Exception as exc:
            return {
                "ok": False,
                "provider": "xai",
                "model": resolved_model,
                "error_code": "XAI_VIDEO_POLL_INVALID_JSON",
                "error_message": str(exc),
                "request_id": request_id,
            }

        status = str(data.get("status") or "").lower()
        print(
            f"[XAI_VIDEO_RESPONSE] project_id={project_id} segment_id={segment_id} request_id={request_id} status={status}"
        )

        if status == "done":
            video = data.get("video") if isinstance(data.get("video"), dict) else {}
            vurl = video.get("url") if isinstance(video, dict) else None
            if not vurl:
                return {
                    "ok": False,
                    "provider": "xai",
                    "model": resolved_model,
                    "error_code": "XAI_VIDEO_MISSING_URL",
                    "error_message": f"done but no video.url: {_truncate(str(data))}",
                    "request_id": request_id,
                }
            print(
                f"[XAI_VIDEO_RESPONSE] project_id={project_id} segment_id={segment_id} request_id={request_id} "
                f"video_url={vurl}"
            )
            return {
                "ok": True,
                "provider": "xai",
                "model": resolved_model,
                "request_id": request_id,
                "video_url": str(vurl),
                "duration_seconds": int(duration_seconds),
            }

        if status in ("failed", "error", "expired"):
            return {
                "ok": False,
                "provider": "xai",
                "model": resolved_model,
                "error_code": "XAI_VIDEO_GENERATION_FAILED",
                "error_message": _truncate(str(data)),
                "request_id": request_id,
            }

        time.sleep(interval)

    return {
        "ok": False,
        "provider": "xai",
        "model": resolved_model,
        "error_code": "XAI_VIDEO_POLL_TIMEOUT",
        "error_message": f"poll exceeded {_poll_timeout_seconds()}s",
        "request_id": request_id,
    }
