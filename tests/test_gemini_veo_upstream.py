"""Gemini Veo upstream (mock HTTP + R2, no real Google)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import gemini_veo_upstream as gvu


@pytest.fixture(autouse=True)
def clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "GEMINI_API_KEY",
        "GEMINI_VIDEO_MODEL",
        "GEMINI_VIDEO_BASE_URL",
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


def test_generate_success_download_and_r2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setenv("GEMINI_VIDEO_MODEL", "veo-3.1-generate-preview")
    monkeypatch.setenv("R2_ENDPOINT", "https://account.r2.cloudflarestorage.com")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "ak")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "sk")
    monkeypatch.setenv("R2_BUCKET_NAME", "bucket")
    monkeypatch.setenv("R2_PUBLIC_BASE_URL", "https://pub.test.r2.dev")

    submit_resp = MagicMock(status_code=200)
    submit_resp.json.return_value = {"name": "operations/op-123"}

    poll_running = MagicMock(status_code=200)
    poll_running.json.return_value = {"done": False}

    poll_done = MagicMock(status_code=200)
    poll_done.json.return_value = {
        "done": True,
        "response": {
            "generateVideoResponse": {
                "generatedSamples": [{"video": {"uri": "https://generativelanguage.googleapis.com/v1beta/files/v1"}}]
            }
        },
    }

    vid_dl = MagicMock(status_code=200, content=b"\x00\x00\x00\x20ftypmp41" + b"\x00" * 16)

    uploads: list[tuple[str, bytes, str]] = []

    def capture_upload(*, object_key: str, data: bytes, content_type: str) -> None:
        uploads.append((object_key, data, content_type))

    monkeypatch.setattr(gvu, "upload_bytes_to_r2", capture_upload)
    monkeypatch.setattr(gvu, "_poll_interval_seconds", lambda: 0.01)
    monkeypatch.setattr(gvu, "_poll_timeout_seconds", lambda: 5.0)

    with patch("httpx.Client") as client_cls:
        inst = client_cls.return_value.__enter__.return_value
        inst.post.return_value = submit_resp
        inst.get.side_effect = [poll_running, poll_done, vid_dl]
        out = gvu.generate_gemini_veo_video_sync(
            project_id=11,
            segment_id="seg_1",
            prompt="hello",
            reference_image_urls=["https://img.example/a.jpg"],
            duration_seconds=8,
            aspect_ratio="9:16",
            resolution="720p",
            model=None,
        )

    assert out["ok"] is True
    assert out["provider"] == "gemini"
    assert out["model"] == "veo-3.1-generate-preview"
    assert out["request_id"] == "operations/op-123"
    assert out["gemini_video_uri"].endswith("/files/v1")
    assert out["video_url"].startswith("https://pub.test.r2.dev/")
    assert out["storage"] == "r2"
    assert out["r2_key"] == "short-drama/videos/11/seg_1/operations%2Fop-123.mp4"
    assert uploads and uploads[0][0] == out["r2_key"]
    assert uploads[0][2] == "video/mp4"


def test_missing_key_returns_normalized_error() -> None:
    out = gvu.generate_gemini_veo_video_sync(
        project_id=1,
        segment_id="seg_1",
        prompt="hello",
        reference_image_urls=[],
        duration_seconds=8,
        aspect_ratio="9:16",
        resolution=None,
        model=None,
    )
    assert out["ok"] is False
    assert out["error_code"] == "GEMINI_API_KEY_NOT_CONFIGURED"
