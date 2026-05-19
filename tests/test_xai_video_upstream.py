"""xAI video upstream (mock HTTP + R2, no real xAI)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

import xai_video_upstream as xvu


@pytest.fixture(autouse=True)
def clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "XAI_API_KEY",
        "OPENAI_API_KEY",
        "XAI_VIDEO_MODEL",
        "XAI_API_BASE",
        "R2_ENDPOINT",
        "R2_ACCOUNT_ID",
        "R2_ACCESS_KEY_ID",
        "R2_ACCESS_KEY",
        "R2_SECRET_ACCESS_KEY",
        "R2_SECRET_KEY",
        "R2_BUCKET_NAME",
        "R2_PUBLIC_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)


def test_generate_success_download_and_r2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XAI_API_KEY", "test-key")
    monkeypatch.setenv("XAI_VIDEO_MODEL", "grok-imagine-video")
    monkeypatch.setenv("R2_ENDPOINT", "https://account.r2.cloudflarestorage.com")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "ak")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "sk")
    monkeypatch.setenv("R2_BUCKET_NAME", "bucket")
    monkeypatch.setenv("R2_PUBLIC_BASE_URL", "https://pub.test.r2.dev")

    vidgen = "https://vidgen.x.ai/out.mp4"
    mp4_bytes = b"\x00\x00\x00\x20ftypmp41" + b"\x00" * 64

    submit_resp = MagicMock()
    submit_resp.status_code = 200
    submit_resp.json.return_value = {"request_id": "req-abc"}

    poll_running = MagicMock()
    poll_running.status_code = 200
    poll_running.json.return_value = {"status": "processing"}

    poll_done = MagicMock()
    poll_done.status_code = 200
    poll_done.json.return_value = {"status": "done", "video": {"url": vidgen}}

    vid_dl = MagicMock()
    vid_dl.status_code = 200
    vid_dl.content = mp4_bytes

    uploads: list[tuple[str, bytes, str]] = []

    def capture_upload(*, object_key: str, data: bytes, content_type: str) -> None:
        uploads.append((object_key, data, content_type))

    monkeypatch.setattr(xvu, "upload_bytes_to_r2", capture_upload)
    monkeypatch.setattr(xvu, "_poll_interval_seconds", lambda: 0.01)
    monkeypatch.setattr(xvu, "_poll_timeout_seconds", lambda: 5.0)

    with patch("httpx.Client") as client_cls:
        inst = client_cls.return_value.__enter__.return_value
        inst.post.return_value = submit_resp
        inst.get.side_effect = [poll_running, poll_done, vid_dl]
        out = xvu.generate_xai_video_sync(
            project_id=2,
            segment_id="seg_1",
            prompt="hello",
            reference_image_urls=["https://img.example/a.jpg"],
            duration_seconds=8,
            aspect_ratio="9:16",
            resolution="720p",
            model="grok-imagine-video",
        )

    assert out["ok"] is True
    assert out["xai_video_url"] == vidgen
    assert out["video_url"] == "https://pub.test.r2.dev/short-drama/videos/2/seg_1/req-abc.mp4"
    assert out["video_url"].startswith("https://pub.test.r2.dev/")
    assert "vidgen.x.ai" not in out["video_url"]
    assert out["storage"] == "r2"
    assert out["r2_key"] == "short-drama/videos/2/seg_1/req-abc.mp4"
    assert out["duration_seconds"] == 8
    assert uploads and uploads[0][0] == out["r2_key"]
    assert uploads[0][1] == mp4_bytes
    assert uploads[0][2] == "video/mp4"


def test_generate_success_when_upstream_returns_non_vidgen_url_still_rehosts_to_r2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XAI_API_KEY", "test-key")
    monkeypatch.setenv("R2_ENDPOINT", "https://account.r2.cloudflarestorage.com")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "ak")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "sk")
    monkeypatch.setenv("R2_BUCKET_NAME", "bucket")
    monkeypatch.setenv("R2_PUBLIC_BASE_URL", "https://pub.test.r2.dev")

    cdn_url = "https://cdn.x.ai/video/out.mp4"
    mp4_bytes = b"x" * 100

    submit_resp = MagicMock(status_code=200)
    submit_resp.json.return_value = {"request_id": "rid-99"}

    poll_done = MagicMock(status_code=200)
    poll_done.json.return_value = {"status": "done", "video": {"url": cdn_url}}

    vid_dl = MagicMock(status_code=200, content=mp4_bytes)

    monkeypatch.setattr(xvu, "upload_bytes_to_r2", lambda **kw: None)
    monkeypatch.setattr(xvu, "_poll_interval_seconds", lambda: 0.01)
    monkeypatch.setattr(xvu, "_poll_timeout_seconds", lambda: 5.0)

    with patch("httpx.Client") as client_cls:
        inst = client_cls.return_value.__enter__.return_value
        inst.post.return_value = submit_resp
        inst.get.side_effect = [poll_done, vid_dl]
        out = xvu.generate_xai_video_sync(
            project_id=1,
            segment_id="s",
            prompt="p",
            reference_image_urls=[],
            duration_seconds=5,
            aspect_ratio="9:16",
            resolution=None,
            model=None,
        )

    assert out["ok"] is True
    assert out["xai_video_url"] == cdn_url
    assert out["video_url"].startswith("https://pub.test.r2.dev/")


def test_r2_config_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XAI_API_KEY", "test-key")

    submit_resp = MagicMock(status_code=200)
    submit_resp.json.return_value = {"request_id": "req-x"}

    poll_done = MagicMock(status_code=200)
    poll_done.json.return_value = {
        "status": "done",
        "video": {"url": "https://vidgen.x.ai/a.mp4"},
    }

    monkeypatch.setattr(xvu, "_poll_interval_seconds", lambda: 0.01)
    monkeypatch.setattr(xvu, "_poll_timeout_seconds", lambda: 5.0)

    with patch("httpx.Client") as client_cls:
        inst = client_cls.return_value.__enter__.return_value
        inst.post.return_value = submit_resp
        inst.get.return_value = poll_done
        out = xvu.generate_xai_video_sync(
            project_id=1,
            segment_id="seg",
            prompt="p",
            reference_image_urls=[],
            duration_seconds=5,
            aspect_ratio="9:16",
            resolution=None,
            model=None,
        )

    assert out["ok"] is False
    assert out["error_code"] == "R2_VIDEO_CONFIG_MISSING"


def test_download_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XAI_API_KEY", "test-key")
    monkeypatch.setenv("R2_ENDPOINT", "https://account.r2.cloudflarestorage.com")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "ak")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "sk")
    monkeypatch.setenv("R2_BUCKET_NAME", "bucket")
    monkeypatch.setenv("R2_PUBLIC_BASE_URL", "https://pub.test.r2.dev")

    submit_resp = MagicMock(status_code=200)
    submit_resp.json.return_value = {"request_id": "req-dl"}

    poll_done = MagicMock(status_code=200)
    poll_done.json.return_value = {
        "status": "done",
        "video": {"url": "https://vidgen.x.ai/b.mp4"},
    }

    vid_dl = MagicMock(status_code=403, text="no")

    monkeypatch.setattr(xvu, "_poll_interval_seconds", lambda: 0.01)
    monkeypatch.setattr(xvu, "_poll_timeout_seconds", lambda: 5.0)

    with patch("httpx.Client") as client_cls:
        inst = client_cls.return_value.__enter__.return_value
        inst.post.return_value = submit_resp
        inst.get.side_effect = [poll_done, vid_dl]
        out = xvu.generate_xai_video_sync(
            project_id=1,
            segment_id="seg",
            prompt="p",
            reference_image_urls=[],
            duration_seconds=5,
            aspect_ratio="9:16",
            resolution=None,
            model=None,
        )

    assert out["ok"] is False
    assert out["error_code"] == "XAI_VIDEO_DOWNLOAD_FAILED"


def test_r2_upload_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XAI_API_KEY", "test-key")
    monkeypatch.setenv("R2_ENDPOINT", "https://account.r2.cloudflarestorage.com")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "ak")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "sk")
    monkeypatch.setenv("R2_BUCKET_NAME", "bucket")
    monkeypatch.setenv("R2_PUBLIC_BASE_URL", "https://pub.test.r2.dev")

    submit_resp = MagicMock(status_code=200)
    submit_resp.json.return_value = {"request_id": "req-up"}

    poll_done = MagicMock(status_code=200)
    poll_done.json.return_value = {
        "status": "done",
        "video": {"url": "https://vidgen.x.ai/c.mp4"},
    }

    vid_dl = MagicMock(status_code=200, content=b"mp4bytes")

    def boom(**kwargs):
        raise ClientError({"Error": {"Code": "AccessDenied", "Message": "no"}}, "PutObject")

    monkeypatch.setattr(xvu, "upload_bytes_to_r2", boom)
    monkeypatch.setattr(xvu, "_poll_interval_seconds", lambda: 0.01)
    monkeypatch.setattr(xvu, "_poll_timeout_seconds", lambda: 5.0)

    with patch("httpx.Client") as client_cls:
        inst = client_cls.return_value.__enter__.return_value
        inst.post.return_value = submit_resp
        inst.get.side_effect = [poll_done, vid_dl]
        out = xvu.generate_xai_video_sync(
            project_id=1,
            segment_id="seg",
            prompt="p",
            reference_image_urls=[],
            duration_seconds=5,
            aspect_ratio="9:16",
            resolution=None,
            model=None,
        )

    assert out["ok"] is False
    assert out["error_code"] == "R2_VIDEO_UPLOAD_FAILED"


def test_generate_failed_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XAI_API_KEY", "test-key")

    submit_resp = MagicMock()
    submit_resp.status_code = 200
    submit_resp.json.return_value = {"request_id": "req-fail"}

    poll_fail = MagicMock()
    poll_fail.status_code = 200
    poll_fail.json.return_value = {"status": "failed", "error": "policy"}

    with patch("httpx.Client") as client_cls:
        inst = client_cls.return_value.__enter__.return_value
        inst.post.return_value = submit_resp
        inst.get.return_value = poll_fail
        out = xvu.generate_xai_video_sync(
            project_id=1,
            segment_id="seg-1",
            prompt="p",
            reference_image_urls=[],
            duration_seconds=5,
            aspect_ratio="9:16",
            resolution=None,
            model=None,
        )

    assert out["ok"] is False
    assert out["error_code"] == "XAI_VIDEO_GENERATION_FAILED"


def test_api_route_returns_proxy_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    import main

    monkeypatch.setenv("PROXY_AUTH_TOKEN", "secret")
    monkeypatch.setattr(
        main,
        "generate_xai_video_sync",
        lambda **kwargs: {
            "ok": True,
            "provider": "xai",
            "model": "grok-imagine-video",
            "request_id": "rid",
            "xai_video_url": "https://vidgen.x.ai/orig.mp4",
            "video_url": "https://pub.example.r2.dev/short-drama/videos/2/seg_1/rid.mp4",
            "storage": "r2",
            "r2_key": "short-drama/videos/2/seg_1/rid.mp4",
            "duration_seconds": 8,
        },
    )
    client = TestClient(main.app)
    resp = client.post(
        "/api/xai/videos/generations",
        headers={"Authorization": "Bearer secret"},
        json={
            "project_id": 2,
            "segment_id": "seg_1",
            "prompt": "test",
            "reference_image_urls": [],
            "duration_seconds": 8,
            "aspect_ratio": "9:16",
            "resolution": "720p",
            "model": "grok-imagine-video",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["video_url"] == "https://pub.example.r2.dev/short-drama/videos/2/seg_1/rid.mp4"
    assert "vidgen.x.ai" not in data["video_url"]
    assert data["storage"] == "r2"
    assert data["r2_key"] == "short-drama/videos/2/seg_1/rid.mp4"
