"""POST /images/generations with client response_format=r2_url (upstream must not see r2_url)."""

from __future__ import annotations

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import main


@pytest.fixture
def image_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("PROXY_AUTH_TOKEN", "proxy-token")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.test-host.example/v1")
    monkeypatch.setenv("XAI_IMAGE_MODEL", "grok-imagine-image")
    monkeypatch.delenv("AI_PROXY_IMAGE_UPSTREAM_FORMAT_FOR_R2", raising=False)
    return TestClient(main.app)


@pytest.fixture(autouse=True)
def r2_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("R2_ENDPOINT", "https://account.r2.cloudflarestorage.com")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "ak")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "sk")
    monkeypatch.setenv("R2_BUCKET_NAME", "bucket")
    monkeypatch.setenv("R2_PUBLIC_BASE_URL", "https://pub.test.r2.dev")


def test_r2_url_upstream_json_uses_url_not_r2_url(
    monkeypatch: pytest.MonkeyPatch,
    image_client: TestClient,
) -> None:
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 40
    posted: list[dict | None] = []

    def capture_upload(*, object_key: str, data: bytes, content_type: str) -> None:
        assert data.startswith(b"\x89PNG")

    monkeypatch.setattr(main, "upload_bytes_to_r2", capture_upload)

    async def fake_post(url: str, **kwargs: object) -> MagicMock:
        posted.append(kwargs.get("json"))  # type: ignore[assignment]
        mr = MagicMock()
        mr.status_code = 200
        mr.json.return_value = {"data": [{"url": "https://imgs.x.ai/temporary.png"}]}
        return mr

    async def fake_get(url: str, **kwargs: object) -> MagicMock:
        gr = MagicMock()
        gr.status_code = 200
        gr.content = png
        gr.headers = {"content-type": "image/png"}
        return gr

    with patch.object(main.httpx, "AsyncClient") as ac_cls:
        inst = AsyncMock()
        ac_cls.return_value.__aenter__.return_value = inst
        inst.post = AsyncMock(side_effect=fake_post)
        inst.get = AsyncMock(side_effect=fake_get)

        r = image_client.post(
            "/images/generations",
            headers={"Authorization": "Bearer proxy-token"},
            json={
                "prompt": "a cat",
                "response_format": "r2_url",
                "project_id": 7,
                "target_type": "character",
                "target_id": 42,
            },
        )

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["storage"] == "r2"
    assert body["response_format"] == "r2_url"
    assert body["url"] == body["image_url"]
    assert body["url"].startswith("https://pub.test.r2.dev/")
    assert "imgs.x.ai" not in body["url"]
    assert "r2_url" not in posted[0]["response_format"]
    assert posted[0]["response_format"] == "url"
    assert "r2_url" not in str(posted[0])


def test_r2_url_env_b64_json_upstream_no_download(
    monkeypatch: pytest.MonkeyPatch,
    image_client: TestClient,
) -> None:
    monkeypatch.setenv("AI_PROXY_IMAGE_UPSTREAM_FORMAT_FOR_R2", "b64_json")
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 40
    b64 = base64.b64encode(png).decode("ascii")
    posted: list[dict | None] = []

    monkeypatch.setattr(main, "upload_bytes_to_r2", lambda **kw: None)

    async def fake_post(url: str, **kwargs: object) -> MagicMock:
        posted.append(kwargs.get("json"))  # type: ignore[assignment]
        mr = MagicMock()
        mr.status_code = 200
        mr.json.return_value = {"data": [{"b64_json": b64}]}
        return mr

    with patch.object(main.httpx, "AsyncClient") as ac_cls:
        inst = AsyncMock()
        ac_cls.return_value.__aenter__.return_value = inst
        inst.post = AsyncMock(side_effect=fake_post)
        inst.get = AsyncMock()

        r = image_client.post(
            "/images/generations",
            headers={"Authorization": "Bearer proxy-token"},
            json={"prompt": "x", "response_format": "r2_url"},
        )

    assert r.status_code == 200, r.text
    assert posted[0]["response_format"] == "b64_json"
    inst.get.assert_not_called()
    assert r.json()["url"].startswith("https://pub.test.r2.dev/")


def test_resolve_upstream_r2_url_defaults_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AI_PROXY_IMAGE_UPSTREAM_FORMAT_FOR_R2", raising=False)
    assert main.resolve_upstream_image_response_format("r2_url") == "url"


def test_resolve_upstream_r2_url_env_invalid_falls_back_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_PROXY_IMAGE_UPSTREAM_FORMAT_FOR_R2", "nonsense")
    assert main.resolve_upstream_image_response_format("r2_url") == "url"


def test_invalid_client_response_format(image_client: TestClient) -> None:
    r = image_client.post(
        "/images/generations",
        headers={"Authorization": "Bearer proxy-token"},
        json={"prompt": "x", "response_format": "r2_url_typo"},
    )
    assert r.status_code == 400
