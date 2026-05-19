"""Integration-style tests for /images/generations r2_url (mocked HTTP + R2 upload)."""

from __future__ import annotations

import base64

import pytest
from botocore.exceptions import ClientError
from fastapi.testclient import TestClient

import main


@pytest.fixture
def auth_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROXY_AUTH_TOKEN", "proxy-token")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.x.ai/v1")


@pytest.fixture
def r2_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("R2_ENDPOINT", "https://account.r2.cloudflarestorage.com")
    monkeypatch.setenv("R2_ACCESS_KEY", "ak")
    monkeypatch.setenv("R2_SECRET_KEY", "sk")
    monkeypatch.setenv("R2_BUCKET_NAME", "bucket")
    monkeypatch.setenv("R2_PUBLIC_BASE_URL", "https://pub.test.r2.dev")


@pytest.fixture
def immediate_to_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _immediate(fn, /, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr(main.asyncio, "to_thread", _immediate)


class FakeResp:
    def __init__(
        self,
        status_code: int,
        json_data: dict | None = None,
        text: str = "",
        content: bytes = b"",
        headers: dict | None = None,
    ):
        self.status_code = status_code
        self._json = json_data
        self.text = text or ""
        self.content = content
        self.headers = headers or {}

    def json(self) -> dict:
        assert self._json is not None
        return self._json


class FakeAsyncClient:
    def __init__(self, post_resp: FakeResp, get_resp: FakeResp | None = None):
        self._post_resp = post_resp
        self._get_resp = get_resp

    async def __aenter__(self) -> FakeAsyncClient:
        return self

    async def __aexit__(self, *a: object) -> None:
        return None

    async def post(self, url: str, headers=None, json=None) -> FakeResp:
        return self._post_resp

    async def get(self, url: str, follow_redirects=True) -> FakeResp:
        if self._get_resp is None:
            raise AssertionError("unexpected GET")
        return self._get_resp


def test_r2_url_upstream_b64_uploads_and_returns_public_url(
    monkeypatch: pytest.MonkeyPatch,
    auth_env: None,
    r2_env: None,
    immediate_to_thread: None,
) -> None:
    img = b"\xff\xd8\xff\xe0" + b"\x00" * 100
    b64 = base64.standard_b64encode(img).decode("ascii")

    uploaded: list[tuple[str, bytes, str]] = []

    def capture_upload(*, object_key: str, data: bytes, content_type: str) -> None:
        uploaded.append((object_key, data, content_type))

    monkeypatch.setattr(main, "upload_bytes_to_r2_sync", capture_upload)
    monkeypatch.setattr(
        main.httpx,
        "AsyncClient",
        lambda **kw: FakeAsyncClient(FakeResp(200, json_data={"data": [{"b64_json": b64}]})),
    )

    c = TestClient(main.app)
    r = c.post(
        "/images/generations",
        headers={"Authorization": "Bearer proxy-token"},
        json={
            "prompt": "cat",
            "response_format": "r2_url",
            "model": "grok-imagine-image",
            "project_id": 10,
            "target_type": "segment",
            "target_id": 3,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["storage"] == "r2"
    assert body["mime_type"] == "image/jpeg"
    assert body["data"][0]["url"].startswith("https://pub.test.r2.dev/")
    assert uploaded and uploaded[0][1] == img
    key = uploaded[0][0]
    assert key.startswith("short-drama/assets/10/segment/3/")
    assert key.endswith(".jpg")
    assert body["data"][0]["url"] == f"https://pub.test.r2.dev/{key}"


def test_r2_url_upstream_url_downloads_then_uploads(
    monkeypatch: pytest.MonkeyPatch,
    auth_env: None,
    r2_env: None,
    immediate_to_thread: None,
) -> None:
    img = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
    post_json = {"data": [{"url": "https://cdn.example/x.png"}]}
    get_resp = FakeResp(200, content=img, headers={"content-type": "image/png"})

    uploaded: list[tuple[str, bytes, str]] = []

    def capture_upload(*, object_key: str, data: bytes, content_type: str) -> None:
        uploaded.append((object_key, data, content_type))

    monkeypatch.setattr(main, "upload_bytes_to_r2_sync", capture_upload)
    monkeypatch.setattr(
        main.httpx,
        "AsyncClient",
        lambda **kw: FakeAsyncClient(FakeResp(200, json_data=post_json), get_resp=get_resp),
    )

    c = TestClient(main.app)
    r = c.post(
        "/images/generations",
        headers={"Authorization": "Bearer proxy-token"},
        json={
            "prompt": "dog",
            "response_format": "r2_url",
            "project_id": 1,
            "target_type": "asset",
            "target_id": 2,
        },
    )
    assert r.status_code == 200, r.text
    assert r.json()["mime_type"] == "image/png"
    assert uploaded and uploaded[0][1] == img


def test_r2_upload_failure_returns_502(
    monkeypatch: pytest.MonkeyPatch,
    auth_env: None,
    r2_env: None,
    immediate_to_thread: None,
) -> None:
    img = b"\xff\xd8\xff\xe0" + b"x"
    b64 = base64.standard_b64encode(img).decode("ascii")

    def boom(*, object_key: str, data: bytes, content_type: str) -> None:
        raise ClientError({"Error": {"Code": "AccessDenied", "Message": "no"}}, "PutObject")

    monkeypatch.setattr(main, "upload_bytes_to_r2_sync", boom)
    monkeypatch.setattr(
        main.httpx,
        "AsyncClient",
        lambda **kw: FakeAsyncClient(FakeResp(200, json_data={"data": [{"b64_json": b64}]})),
    )

    c = TestClient(main.app)
    r = c.post(
        "/images/generations",
        headers={"Authorization": "Bearer proxy-token"},
        json={"prompt": "z", "response_format": "r2_url"},
    )
    assert r.status_code == 502
    assert "r2_upload_failed" in r.text


def test_b64_json_response_format_unchanged_no_r2(
    monkeypatch: pytest.MonkeyPatch,
    auth_env: None,
    immediate_to_thread: None,
) -> None:
    img = b"\xff\xd8\xff\xe0" + b"y"
    b64 = base64.standard_b64encode(img).decode("ascii")

    called_upload: list[bool] = []

    monkeypatch.setattr(
        main,
        "upload_bytes_to_r2_sync",
        lambda **kw: called_upload.append(True),
    )
    monkeypatch.setattr(
        main.httpx,
        "AsyncClient",
        lambda **kw: FakeAsyncClient(FakeResp(200, json_data={"data": [{"b64_json": b64}]})),
    )

    c = TestClient(main.app)
    r = c.post(
        "/images/generations",
        headers={"Authorization": "Bearer proxy-token"},
        json={"prompt": "q", "response_format": "b64_json"},
    )
    assert r.status_code == 200
    body = r.json()
    assert "b64_json" in body["data"][0]
    assert body["data"][0]["b64_json"] == b64
    assert called_upload == []
