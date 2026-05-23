"""Google Gemini Veo video generation for the Railway proxy."""

from __future__ import annotations

import os
import base64
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


def _image_mime_from_response(raw: bytes, content_type: str | None) -> str:
    ct = (content_type or "").split(";", 1)[0].strip().lower()
    if ct.startswith("image/"):
        return ct
    if raw.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def _log_gemini_failure(
    *,
    project_id: int,
    segment_id: str,
    error_code: str,
    error_message: str,
    request_id: str = "",
    elapsed_seconds: float | None = None,
) -> None:
    elapsed_part = "" if elapsed_seconds is None else f" elapsed_seconds={elapsed_seconds:.3f}"
    print(
        f"[GEMINI_VEO_FAILED] project_id={project_id} segment_id={segment_id} "
        f"request_id={request_id} error_code={error_code} "
        f"error_message={_truncate(str(error_message), 500)}{elapsed_part}",
        flush=True,
    )


def _append_key(url: str, api_key: str) -> str:
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}key={api_key}"


def sanitize_segment_id_for_r2(segment_id: str) -> str:
    s = (segment_id or "").strip() or "unknown"
    return s.replace("/", "_").replace("\\", "_")[:200]


def build_gemini_veo_payload(
    *,
    prompt: str,
    inline_reference_images: list[dict[str, str]] | None = None,
    aspect_ratio: str,
    duration_seconds: int,
    resolution: str | None,
) -> dict[str, Any]:
    instance: dict[str, Any] = {"prompt": (prompt or "").strip()}
    refs = list(inline_reference_images or [])
    if refs:
        # Veo REST rejects referenceImages[].image.url for this model. Send the
        # first reference as inline image bytes, which is the supported
        # image-to-video shape for predictLongRunning.
        instance["image"] = refs[0]
    parameters: dict[str, Any] = {
        "aspectRatio": (aspect_ratio or "9:16").strip(),
        "durationSeconds": int(duration_seconds),
    }
    if resolution:
        parameters["resolution"] = str(resolution).strip()
    return {"instances": [instance], "parameters": parameters}


def download_gemini_reference_images(
    *,
    project_id: int,
    segment_id: str,
    reference_image_urls: list[str],
    timeout: httpx.Timeout,
) -> tuple[list[dict[str, str]], str | None]:
    urls = [u.strip() for u in reference_image_urls if (u or "").strip()]
    if not urls:
        return [], None

    first_url = urls[0]
    started = time.monotonic()
    print(
        f"[GEMINI_VEO_REFERENCE_DOWNLOAD_START] project_id={project_id} segment_id={segment_id} "
        f"reference_image_url={first_url} requested_reference_count={len(urls)} used_reference_count=1",
        flush=True,
    )
    try:
        with httpx.Client(timeout=timeout, http2=False, verify=True, follow_redirects=True) as client:
            resp = client.get(first_url)
    except (httpx.TimeoutException, httpx.RequestError) as exc:
        message = f"reference image download failed: {exc}"
        _log_gemini_failure(
            project_id=project_id,
            segment_id=segment_id,
            error_code="GEMINI_VEO_REFERENCE_IMAGE_DOWNLOAD_FAILED",
            error_message=message,
            elapsed_seconds=time.monotonic() - started,
        )
        return [], message

    elapsed = time.monotonic() - started
    if resp.status_code >= 400:
        message = f"reference image download HTTP {resp.status_code}: {_truncate(resp.text or '')}"
        _log_gemini_failure(
            project_id=project_id,
            segment_id=segment_id,
            error_code="GEMINI_VEO_REFERENCE_IMAGE_DOWNLOAD_FAILED",
            error_message=message,
            elapsed_seconds=elapsed,
        )
        return [], message

    raw = resp.content or b""
    if not raw:
        message = "reference image download returned empty body"
        _log_gemini_failure(
            project_id=project_id,
            segment_id=segment_id,
            error_code="GEMINI_VEO_REFERENCE_IMAGE_DOWNLOAD_FAILED",
            error_message=message,
            elapsed_seconds=elapsed,
        )
        return [], message

    mime_type = _image_mime_from_response(raw, resp.headers.get("content-type"))
    print(
        f"[GEMINI_VEO_REFERENCE_DOWNLOAD_SUCCESS] project_id={project_id} segment_id={segment_id} "
        f"bytes_size={len(raw)} mime_type={mime_type} elapsed_seconds={elapsed:.3f}",
        flush=True,
    )
    return [
        {
            "bytesBase64Encoded": base64.b64encode(raw).decode("ascii"),
            "mimeType": mime_type,
        }
    ], None


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
        _log_gemini_failure(
            project_id=project_id,
            segment_id=segment_id,
            error_code="GEMINI_API_KEY_NOT_CONFIGURED",
            error_message="GEMINI_API_KEY is not configured on Railway proxy",
        )
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
    inline_reference_images, reference_error = download_gemini_reference_images(
        project_id=project_id,
        segment_id=segment_id,
        reference_image_urls=reference_image_urls,
        timeout=timeout,
    )
    if reference_error:
        return {
            "ok": False,
            "provider": "gemini",
            "model": resolved_model,
            "error_code": "GEMINI_VEO_REFERENCE_IMAGE_DOWNLOAD_FAILED",
            "error_message": reference_error,
            "request_id": "",
        }

    payload = build_gemini_veo_payload(
        prompt=prompt,
        inline_reference_images=inline_reference_images,
        aspect_ratio=aspect_ratio,
        duration_seconds=duration_seconds,
        resolution=resolution,
    )

    print(
        f"[GEMINI_VEO_REQUEST] project_id={project_id} segment_id={segment_id} model={resolved_model} "
        f"duration={duration_seconds} aspect_ratio={aspect_ratio} resolution={resolution or ''} "
        f"reference_image_count={len(inline_reference_images)} reference_mode=inline_image "
        f"prompt_chars={len(prompt or '')} timeout_seconds={_timeout_seconds()} "
        f"poll_timeout_seconds={_poll_timeout_seconds()} poll_interval_seconds={_poll_interval_seconds()}",
        flush=True,
    )

    started = time.monotonic()
    try:
        with httpx.Client(timeout=timeout, http2=False, verify=True, follow_redirects=True) as client:
            resp = client.post(submit_url, headers={"Content-Type": "application/json"}, json=payload)
    except httpx.TimeoutException as exc:
        _log_gemini_failure(
            project_id=project_id,
            segment_id=segment_id,
            error_code="GEMINI_VEO_SUBMIT_TIMEOUT",
            error_message=str(exc),
            elapsed_seconds=time.monotonic() - started,
        )
        return {
            "ok": False,
            "provider": "gemini",
            "model": resolved_model,
            "error_code": "GEMINI_VEO_SUBMIT_TIMEOUT",
            "error_message": str(exc),
            "request_id": "",
        }
    except httpx.RequestError as exc:
        _log_gemini_failure(
            project_id=project_id,
            segment_id=segment_id,
            error_code="GEMINI_VEO_NETWORK_ERROR",
            error_message=str(exc),
            elapsed_seconds=time.monotonic() - started,
        )
        return {
            "ok": False,
            "provider": "gemini",
            "model": resolved_model,
            "error_code": "GEMINI_VEO_NETWORK_ERROR",
            "error_message": str(exc),
            "request_id": "",
        }

    submit_elapsed = time.monotonic() - started
    submit_body_text = resp.text or ""
    print(
        f"[GEMINI_VEO_SUBMIT_RESPONSE] project_id={project_id} segment_id={segment_id} "
        f"status_code={resp.status_code} elapsed_seconds={submit_elapsed:.3f} "
        f"body_prefix={_truncate(submit_body_text, 500)}",
        flush=True,
    )

    if resp.status_code >= 400:
        _log_gemini_failure(
            project_id=project_id,
            segment_id=segment_id,
            error_code="GEMINI_VEO_HTTP_ERROR",
            error_message=f"HTTP {resp.status_code}: {_truncate(resp.text or '')}",
            elapsed_seconds=submit_elapsed,
        )
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
        _log_gemini_failure(
            project_id=project_id,
            segment_id=segment_id,
            error_code="GEMINI_VEO_INVALID_JSON",
            error_message=str(exc),
            elapsed_seconds=submit_elapsed,
        )
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
        _log_gemini_failure(
            project_id=project_id,
            segment_id=segment_id,
            error_code="GEMINI_VEO_MISSING_OPERATION",
            error_message=f"start response missing operation name: {_truncate(str(start_data))}",
            elapsed_seconds=submit_elapsed,
        )
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
    print(
        f"[GEMINI_VEO_OPERATION_CREATED] project_id={project_id} segment_id={segment_id} "
        f"request_id={operation_name} operation_name={operation_name} "
        f"submit_elapsed_seconds={submit_elapsed:.3f}",
        flush=True,
    )

    poll_attempt = 0
    while time.monotonic() < deadline:
        poll_attempt += 1
        poll_started = time.monotonic()
        print(
            f"[GEMINI_VEO_POLL_START] project_id={project_id} segment_id={segment_id} "
            f"request_id={operation_name} attempt={poll_attempt}",
            flush=True,
        )
        try:
            with httpx.Client(timeout=timeout, http2=False, verify=True, follow_redirects=True) as client:
                poll_resp = client.get(poll_url)
        except httpx.TimeoutException as exc:
            _log_gemini_failure(
                project_id=project_id,
                segment_id=segment_id,
                request_id=operation_name,
                error_code="GEMINI_VEO_POLL_TIMEOUT",
                error_message=str(exc),
                elapsed_seconds=time.monotonic() - poll_started,
            )
            return {
                "ok": False,
                "provider": "gemini",
                "model": resolved_model,
                "error_code": "GEMINI_VEO_POLL_TIMEOUT",
                "error_message": str(exc),
                "request_id": operation_name,
            }
        except httpx.RequestError as exc:
            _log_gemini_failure(
                project_id=project_id,
                segment_id=segment_id,
                request_id=operation_name,
                error_code="GEMINI_VEO_POLL_NETWORK_ERROR",
                error_message=str(exc),
                elapsed_seconds=time.monotonic() - poll_started,
            )
            return {
                "ok": False,
                "provider": "gemini",
                "model": resolved_model,
                "error_code": "GEMINI_VEO_POLL_NETWORK_ERROR",
                "error_message": str(exc),
                "request_id": operation_name,
            }

        poll_elapsed = time.monotonic() - poll_started
        if poll_resp.status_code >= 400:
            _log_gemini_failure(
                project_id=project_id,
                segment_id=segment_id,
                request_id=operation_name,
                error_code="GEMINI_VEO_POLL_HTTP_ERROR",
                error_message=f"HTTP {poll_resp.status_code}: {_truncate(poll_resp.text or '')}",
                elapsed_seconds=poll_elapsed,
            )
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
            _log_gemini_failure(
                project_id=project_id,
                segment_id=segment_id,
                request_id=operation_name,
                error_code="GEMINI_VEO_POLL_INVALID_JSON",
                error_message=str(exc),
                elapsed_seconds=poll_elapsed,
            )
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
            f"request_id={operation_name} attempt={poll_attempt} status_code={poll_resp.status_code} "
            f"elapsed_seconds={poll_elapsed:.3f} done={bool(data.get('done'))} "
            f"body_prefix={_truncate(poll_resp.text or '', 500)}",
            flush=True,
        )

        if bool(data.get("done")):
            operation_error = _extract_operation_error(data)
            if operation_error:
                _log_gemini_failure(
                    project_id=project_id,
                    segment_id=segment_id,
                    request_id=operation_name,
                    error_code="GEMINI_VEO_OPERATION_FAILED",
                    error_message=operation_error,
                )
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
                _log_gemini_failure(
                    project_id=project_id,
                    segment_id=segment_id,
                    request_id=operation_name,
                    error_code="GEMINI_VEO_MISSING_VIDEO_URI",
                    error_message=f"done but no video uri: {_truncate(str(data))}",
                )
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
                _log_gemini_failure(
                    project_id=project_id,
                    segment_id=segment_id,
                    request_id=operation_name,
                    error_code="R2_VIDEO_CONFIG_MISSING",
                    error_message="R2 not configured",
                )
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
            print(
                f"[GEMINI_VEO_DOWNLOAD_START] project_id={project_id} segment_id={segment_id} "
                f"request_id={operation_name} gemini_video_uri={video_uri}",
                flush=True,
            )

            download_started = time.monotonic()
            try:
                with httpx.Client(timeout=timeout, http2=False, verify=True, follow_redirects=True) as dl_client:
                    vid_resp = dl_client.get(download_url)
            except (httpx.TimeoutException, httpx.RequestError) as exc:
                _log_gemini_failure(
                    project_id=project_id,
                    segment_id=segment_id,
                    request_id=operation_name,
                    error_code="GEMINI_VEO_DOWNLOAD_FAILED",
                    error_message=str(exc),
                    elapsed_seconds=time.monotonic() - download_started,
                )
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
                _log_gemini_failure(
                    project_id=project_id,
                    segment_id=segment_id,
                    request_id=operation_name,
                    error_code="GEMINI_VEO_DOWNLOAD_FAILED",
                    error_message=f"HTTP {vid_resp.status_code}: {_truncate(vid_resp.text or '')}",
                    elapsed_seconds=time.monotonic() - download_started,
                )
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
            print(
                f"[GEMINI_VEO_DOWNLOAD_SUCCESS] project_id={project_id} segment_id={segment_id} "
                f"request_id={operation_name} status_code={vid_resp.status_code} "
                f"bytes_size={len(raw_mp4)} elapsed_seconds={time.monotonic() - download_started:.3f}",
                flush=True,
            )
            mx = _max_video_download_bytes()
            if len(raw_mp4) > mx:
                _log_gemini_failure(
                    project_id=project_id,
                    segment_id=segment_id,
                    request_id=operation_name,
                    error_code="GEMINI_VEO_DOWNLOAD_FAILED",
                    error_message=f"video too large: {len(raw_mp4)} bytes (max {mx})",
                )
                return {
                    "ok": False,
                    "provider": "gemini",
                    "model": resolved_model,
                    "request_id": operation_name,
                    "gemini_video_uri": video_uri,
                    "error_code": "GEMINI_VEO_DOWNLOAD_FAILED",
                    "error_message": f"video too large: {len(raw_mp4)} bytes (max {mx})",
                }

            print(
                f"[GEMINI_VEO_R2_UPLOAD_START] project_id={project_id} segment_id={segment_id} "
                f"request_id={operation_name} r2_key={storage_key} bytes_size={len(raw_mp4)}",
                flush=True,
            )
            upload_started = time.monotonic()
            try:
                upload_bytes_to_r2(object_key=storage_key, data=raw_mp4, content_type="video/mp4")
            except ClientError as exc:
                _log_gemini_failure(
                    project_id=project_id,
                    segment_id=segment_id,
                    request_id=operation_name,
                    error_code="R2_VIDEO_UPLOAD_FAILED",
                    error_message=_truncate(str(exc)),
                    elapsed_seconds=time.monotonic() - upload_started,
                )
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
                _log_gemini_failure(
                    project_id=project_id,
                    segment_id=segment_id,
                    request_id=operation_name,
                    error_code="R2_VIDEO_UPLOAD_FAILED",
                    error_message=_truncate(str(exc)),
                    elapsed_seconds=time.monotonic() - upload_started,
                )
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
                f"[GEMINI_VEO_R2_UPLOAD_SUCCESS] project_id={project_id} segment_id={segment_id} "
                f"request_id={operation_name} r2_key={storage_key} "
                f"elapsed_seconds={time.monotonic() - upload_started:.3f}",
                flush=True,
            )
            print(
                f"[GEMINI_VEO_RESPONSE_READY] project_id={project_id} segment_id={segment_id} "
                f"request_id={operation_name} success=true video_url={public_url} "
                f"storage=r2 r2_key={storage_key} model={resolved_model}",
                flush=True,
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

    _log_gemini_failure(
        project_id=project_id,
        segment_id=segment_id,
        request_id=operation_name,
        error_code="GEMINI_VEO_POLL_TIMEOUT",
        error_message=f"timed out after {_poll_timeout_seconds()} seconds",
    )
    return {
        "ok": False,
        "provider": "gemini",
        "model": resolved_model,
        "request_id": operation_name,
        "error_code": "GEMINI_VEO_POLL_TIMEOUT",
        "error_message": f"timed out after {_poll_timeout_seconds()} seconds",
    }
