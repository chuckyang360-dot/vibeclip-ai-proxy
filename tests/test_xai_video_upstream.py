"""xAI video upstream (mock HTTP, no real xAI)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import xai_video_upstream as xvu


@pytest.fixture(autouse=True)
def clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "XAI_API_KEY",
        "OPENAI_API_KEY",
        "XAI_VIDEO_MODEL",
        "XAI_API_BASE",
    ):
        monkeypatch.delenv(key, raising=False)


def test_generate_success_poll_done(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XAI_API_KEY", "test-key")
    monkeypatch.setenv("XAI_VIDEO_MODEL", "grok-imagine-video")

    submit_resp = MagicMock()
    submit_resp.status_code = 200
    submit_resp.json.return_value = {"request_id": "req-abc"}

    poll_running = MagicMock()
    poll_running.status_code = 200
    poll_running.json.return_value = {"status": "processing"}

    poll_done = MagicMock()
    poll_done.status_code = 200
    poll_done.json.return_value = {
        "status": "done",
        "video": {"url": "https://x.ai/video/out.mp4"},
    }

    with patch("httpx.Client") as client_cls:
        inst = client_cls.return_value.__enter__.return_value
        inst.post.return_value = submit_resp
        inst.get.side_effect = [poll_running, poll_done]
        monkeypatch.setattr(xvu, "_poll_interval_seconds", lambda: 0.01)
        monkeypatch.setattr(xvu, "_poll_timeout_seconds", lambda: 5.0)
        out = xvu.generate_xai_video_sync(
            project_id=1,
            segment_id="seg-1",
            prompt="hello",
            reference_image_urls=["https://img.example/a.jpg"],
            duration_seconds=6,
            aspect_ratio="9:16",
            resolution="720p",
            model="grok-imagine-video",
        )

    assert out["ok"] is True
    assert out["video_url"] == "https://x.ai/video/out.mp4"
    assert out["request_id"] == "req-abc"
    post_json = inst.post.call_args.kwargs["json"]
    assert post_json["model"] == "grok-imagine-video"
    assert post_json["duration"] == 6
    assert post_json["reference_images"] == [{"url": "https://img.example/a.jpg"}]


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
            "video_url": "https://cdn.example/v.mp4",
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
    assert data["video_url"] == "https://cdn.example/v.mp4"
