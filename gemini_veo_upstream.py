"""Google Gemini Veo video generation for the Railway proxy."""

from __future__ import annotations

import os
import time
from typing import Any
from urllib.parse import quote

import httpx
from botocore.exceptions import ClientError

from r2_upload import load_r2_settings, upload_bytes_to_r2

_DEFAULT_GEMINI_VIDEO_MODEL = "veo-3.1-generate-preview"
_MAX_VIDEO_DOWNLOAD_BYTES_DEFAULT = 512 * 1024 * 1024


def get_env(key: str, default: str | None = None) -> str | None:
    val = os.getenv(key)
    if val is None or str(val).strip() == "":
        return default
    return str(val).strip()


def resolve_gemini_video_model(requested: str | None) -> str:
    if requested and str(requested).strip():
        return str(requested).strip()
    return get_env("GEMINI_VIDEO_MODEL") or _DEFAULT_GEMINI_VIDEO_MODEL


def resolve_gemini_api_key() -> str:
    return get_env("GEMINI_API_KEY") or ""


def resolve_gemini_video_base_url() -> str:
    return (get_env("GEMINI_VIDEO_BASE_URL") or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")


def _timeout_seconds() -> float:
    raw = get_env("GEMINI_VIDEO_TIMEOUT_SECONDS") or get_env("REQUEST_TIMEOUT_SECONDS") or "600"
    try:
        return max(30.0, float(raw))
    except ValueError:
        return 600.0


def _poll_interval_seconds() -> float:
    raw = get_env("GEMINI_VIDEO_POLL_INTERVAL_SECONDS") or "10"
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 10.0


def _poll_timeout_seconds() -> float:
    raw = get_env("GEMINI_VIDEO_POLL_TIMEOUT_SECONDS") or get_env("GEMINI_VIDEO_TIMEOUT_SECONDS") or "600"
    try:
        return max(30.0, float(raw))
    except ValueError:
        return 600.0


def _max_video_download_bytes() -> int:
    raw = get_env("GEMINI_VIDEO_MAX_DOWNLOAD_BYTES") or get_env("MAX_VIDEO_DOWNLOAD_BYTES")
    if raw:
        try:
            return max(1_000_000, int(raw))
        except ValueError:
            pass
    return _MAX_VIDEO_DOWNLOAD_BYTES_DEFAULT


def _truncate(text: str, limit: int = 1000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _append_key(url: str, api_key: str) -> str:
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}key={api_key}"


def sanitize_segment_id_for_r2(segment_id: str) -> str:
    s = (segment_id or "").strip() or "unknown"
    return s.replace("/", "_").replace("\\", "_")[:200]


def build_gemini_veo_payload(
    *,
    prompt: str,
    reference_image_urls: list[str],
    aspect_ratio: str,
    duration_seconds: int,
    resolution: str | None,
) -> dict[str, Any]:
    instance: dict[str, Any] = {"prompt": (prompt or "").strip()}
    refs = [{"image": {"url": u}} for u in reference_image_urls if (u or "").strip()]
    if refs:
        instance["referenceImages"] = refs[:3]
    parameters: dict[str, Any] = {
        "aspectRatio": (aspect_ratio or "9:16").strip(),
        "durationSeconds": int(duration_seconds),
    }
    if resolution:
        parameters["resolution"] = str(resolution).strip()
    return {"instances": [instance], "parameters": parameters}


def _extract_operation_name(body: Any) -> str | None:
    if not isinstance(body, dict):
        return None
    name = body.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None


def _extract_operation_error(body: Any) -> str:
    if not isinstance(body, dict):
        return ""
    err = body.get("error")
    if isinstance(err, dict):
        msg = err.get("message") or err.get("code") or err.get("status")
        if msg is not None and str(msg).strip():
            return str(msg).strip()
    if err is not None and str(err).strip():
        return str(err).strip()
    return ""


def _extract_video_uri(body: Any) -> str | None:
    if not isinstance(body, dict):
        return None
    response = body.get("response")
    if not isinstance(response, dict):
        return None
    generate_response = response.get("generateVideoResponse")
    if isinstance(generate_response, dict):
        samples = generate_response.get("generatedSamples")
        if isinstance(samples, list):
            for sample in samples:
                if not isinstance(sample, dict):
                    continue
                video = sample.get("video")
                if isinstance(video, dict):
                    uri = video.get("uri")
                    if isinstance(uri, str) and uri.strip():
                        return uri.strip()
    generated_videos = response.get("generatedVideos")
    if isinstance(generated_videos, list):
        for item in generated_videos:
            if not isinstance(item, dict):
                continue
            video = item.get("video")
            if isinstance(video, dict):
                uri = video.get("uri")
                if isinstance(uri, str) and uri.strip():
                    return uri.strip()
    return None


def generate_gemini_veo_video_sync(
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
    api_key = resolve_gemini_api_key()
    resolved_model = resolve_gemini_video_model(model)
    if not api_key:
        return {
            "ok": False,
            "provider": "gemini",
            "model": resolved_model,
            "error_code": "GEMINI_API_KEY_NOT_CONFIGURED",
            "error_message": "GEMINI_API_KEY is not configured on Railway proxy",
            "request_id": "",
        }

    base = resolve_gemini_video_base_url()
    submit_url = _append_key(f"{base}/models/{resolved_model}:predictLongRunning", api_key)
    timeout = httpx.Timeout(
        connect=min(30.0, _timeout_seconds()),
        read=_timeout_seconds(),
        write=_timeout_seconds(),
        pool=10.0,
    )
    payload = build_gemini_veo_payload(
        prompt=prompt,
        reference_image_urls=reference_image_urls,
        aspect_ratio=aspect_ratio,
        duration_seconds=duration_seconds,
        resolution=resolution,
    )

    print(
        f"[GEMINI_VEO_REQUEST] project_id={project_id} segment_id={segment_id} model={resolved_model} "
        f"duration={duration_seconds} aspect_ratio={aspect_ratio} resolution={resolution or ''} "
        f"reference_image_count={len(payload['instances'][0].get('referenceImages') or [])} "
        f"prompt_chars={len(prompt or '')}"
    )

    try:
        with httpx.Client(timeout=timeout, http2=False, verify=True, follow_redirects=True) as client:
            resp = client.post(submit_url, headers={"Content-Type": "application/json"}, json=payload)
    except httpx.TimeoutException as exc:
        return {
            "ok": False,
            "provider": "gemini",
            "model": resolved_model,
            "error_code": "GEMINI_VEO_SUBMIT_TIMEOUT",
            "error_message": str(exc),
            "request_id": "",
        }
    except httpx.RequestError as exc:
        return {
            "ok": False,
            "provider": "gemini",
            "model": resolved_model,
            "error_code": "GEMINI_VEO_NETWORK_ERROR",
            "error_message": str(exc),
            "request_id": "",
        }

    if resp.status_code >= 400:
        return {
            "ok": False,
            "provider": "gemini",
            "model": resolved_model,
            "error_code": "GEMINI_VEO_HTTP_ERROR",
            "error_message": f"HTTP {resp.status_code}: {_truncate(resp.text or '')}",
            "request_id": "",
        }

    try:
        start_data = resp.json()
    except Exception as exc:
        return {
            "ok": False,
            "provider": "gemini",
            "model": resolved_model,
            "error_code": "GEMINI_VEO_INVALID_JSON",
            "error_message": str(exc),
            "request_id": "",
        }

    operation_name = _extract_operation_name(start_data)
    if not operation_name:
        return {
            "ok": False,
            "provider": "gemini",
            "model": resolved_model,
            "error_code": "GEMINI_VEO_MISSING_OPERATION",
            "error_message": f"start response missing operation name: {_truncate(str(start_data))}",
            "request_id": "",
        }

    poll_url = _append_key(f"{base}/{operation_name}", api_key)
    deadline = time.monotonic() + _poll_timeout_seconds()
    interval = _poll_interval_seconds()

    while time.monotonic() < deadline:
        try:
            with httpx.Client(timeout=timeout, http2=False, verify=True, follow_redirects=True) as client:
                poll_resp = client.get(poll_url)
        except httpx.TimeoutException as exc:
            return {
                "ok": False,
                "provider": "gemini",
                "model": resolved_model,
                "error_code": "GEMINI_VEO_POLL_TIMEOUT",
                "error_message": str(exc),
                "request_id": operation_name,
            }
        except httpx.RequestError as exc:
            return {
                "ok": False,
                "provider": "gemini",
                "model": resolved_model,
                "error_code": "GEMINI_VEO_POLL_NETWORK_ERROR",
                "error_message": str(exc),
                "request_id": operation_name,
            }

        if poll_resp.status_code >= 400:
            return {
                "ok": False,
                "provider": "gemini",
                "model": resolved_model,
                "error_code": "GEMINI_VEO_POLL_HTTP_ERROR",
                "error_message": f"HTTP {poll_resp.status_code}: {_truncate(poll_resp.text or '')}",
                "request_id": operation_name,
            }

        try:
            data = poll_resp.json()
        except Exception as exc:
            return {
                "ok": False,
                "provider": "gemini",
                "model": resolved_model,
                "error_code": "GEMINI_VEO_POLL_INVALID_JSON",
                "error_message": str(exc),
                "request_id": operation_name,
            }

        print(
            f"[GEMINI_VEO_RESPONSE] project_id={project_id} segment_id={segment_id} "
            f"request_id={operation_name} done={bool(data.get('done'))}"
        )

        if bool(data.get("done")):
            operation_error = _extract_operation_error(data)
            if operation_error:
                return {
                    "ok": False,
                    "provider": "gemini",
                    "model": resolved_model,
                    "error_code": "GEMINI_VEO_OPERATION_FAILED",
                    "error_message": operation_error,
                    "request_id": operation_name,
                }
            video_uri = _extract_video_uri(data)
            if not video_uri:
                return {
                    "ok": False,
                    "provider": "gemini",
                    "model": resolved_model,
                    "error_code": "GEMINI_VEO_MISSING_VIDEO_URI",
                    "error_message": f"done but no video uri: {_truncate(str(data))}",
                    "request_id": operation_name,
                }

            r2_cfg = load_r2_settings()
            if r2_cfg is None:
                return {
                    "ok": False,
                    "provider": "gemini",
                    "model": resolved_model,
                    "request_id": operation_name,
                    "gemini_video_uri": video_uri,
                    "error_code": "R2_VIDEO_CONFIG_MISSING",
                    "error_message": (
                        "R2 not configured: set R2_ENDPOINT (or R2_ACCOUNT_ID), R2_BUCKET_NAME, "
                        "R2_PUBLIC_BASE_URL, R2_ACCESS_KEY_ID (or R2_ACCESS_KEY), "
                        "R2_SECRET_ACCESS_KEY (or R2_SECRET_KEY)"
                    ),
                }

            seg_safe = sanitize_segment_id_for_r2(segment_id)
            storage_key = f"short-drama/videos/{project_id}/{seg_safe}/{quote(operation_name, safe='')}.mp4"
            download_url = _append_key(video_uri, api_key)

            try:
                with httpx.Client(timeout=timeout, http2=False, verify=True, follow_redirects=True) as dl_client:
                    vid_resp = dl_client.get(download_url)
            except (httpx.TimeoutException, httpx.RequestError) as exc:
                return {
                    "ok": False,
                    "provider": "gemini",
                    "model": resolved_model,
                    "request_id": operation_name,
                    "gemini_video_uri": video_uri,
                    "error_code": "GEMINI_VEO_DOWNLOAD_FAILED",
                    "error_message": str(exc),
                }

            if vid_resp.status_code >= 400:
                return {
                    "ok": False,
                    "provider": "gemini",
                    "model": resolved_model,
                    "request_id": operation_name,
                    "gemini_video_uri": video_uri,
                    "error_code": "GEMINI_VEO_DOWNLOAD_FAILED",
                    "error_message": f"HTTP {vid_resp.status_code}: {_truncate(vid_resp.text or '')}",
                }

            raw_mp4 = vid_resp.content
            mx = _max_video_download_bytes()
            if len(raw_mp4) > mx:
                return {
                    "ok": False,
                    "provider": "gemini",
                    "model": resolved_model,
                    "request_id": operation_name,
                    "gemini_video_uri": video_uri,
                    "error_code": "GEMINI_VEO_DOWNLOAD_FAILED",
                    "error_message": f"video too large: {len(raw_mp4)} bytes (max {mx})",
                }

            try:
                upload_bytes_to_r2(object_key=storage_key, data=raw_mp4, content_type="video/mp4")
            except ClientError as exc:
                return {
                    "ok": False,
                    "provider": "gemini",
                    "model": resolved_model,
                    "request_id": operation_name,
                    "gemini_video_uri": video_uri,
                    "error_code": "R2_VIDEO_UPLOAD_FAILED",
                    "error_message": _truncate(str(exc)),
                }
            except Exception as exc:
                return {
                    "ok": False,
                    "provider": "gemini",
                    "model": resolved_model,
                    "request_id": operation_name,
                    "gemini_video_uri": video_uri,
                    "error_code": "R2_VIDEO_UPLOAD_FAILED",
                    "error_message": _truncate(str(exc)),
                }

            public_url = f"{r2_cfg.public_base_url}/{quote(storage_key, safe='/')}"
            print(
                f"[GEMINI_VEO_RESPONSE_READY] request_id={operation_name} success=true "
                f"video_url={public_url} storage=r2 r2_key={storage_key} model={resolved_model}"
            )
            return {
                "ok": True,
                "provider": "gemini",
                "model": resolved_model,
                "request_id": operation_name,
                "gemini_video_uri": video_uri,
                "video_url": public_url,
                "storage": "r2",
                "r2_key": storage_key,
                "duration_seconds": int(duration_seconds),
            }

        time.sleep(interval)

    return {
        "ok": False,
        "provider": "gemini",
        "model": resolved_model,
        "request_id": operation_name,
        "error_code": "GEMINI_VEO_POLL_TIMEOUT",
        "error_message": f"timed out after {_poll_timeout_seconds()} seconds",
    }
