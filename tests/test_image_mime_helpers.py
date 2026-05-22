"""Tests for image MIME sniffing helpers used by /images/generations."""

from __future__ import annotations

import pytest

import main


def test_sniff_png() -> None:
    assert main.sniff_image_mime(b"\x89PNG\r\n\x1a\n" + b"x" * 20) == "image/png"


def test_sniff_jpeg() -> None:
    assert main.sniff_image_mime(b"\xff\xd8\xff\xe0" + b"y" * 20) == "image/jpeg"


def test_sniff_webp() -> None:
    chunk = b"RIFF\x00\x00\x00\x00WEBP" + b"z" * 8
    assert main.sniff_image_mime(chunk) == "image/webp"


def test_resolve_mime_prefers_content_type_image() -> None:
    raw = b"\x89PNG\r\n\x1a\n" + b"x" * 8
    assert main._resolve_mime_from_bytes(raw, "image/jpeg; charset=binary") == "image/jpeg"


def test_resolve_mime_sniffs_when_octet_stream() -> None:
    raw = b"\x89PNG\r\n\x1a\n" + b"x" * 8
    assert main._resolve_mime_from_bytes(raw, "application/octet-stream") == "image/png"


def test_extension_for_mime_image() -> None:
    assert main.extension_for_mime_image("image/jpeg") == ".jpg"
    assert main.extension_for_mime_image("image/png") == ".png"
    assert main.extension_for_mime_image("image/webp") == ".webp"
    assert main.extension_for_mime_image("application/octet-stream") == ".bin"


def test_build_image_r2_object_key() -> None:
    k = main.build_image_r2_object_key(
        project_id=7,
        target_type="scene",
        target_id=99,
        proxy_request_id="abc-uuid",
        mime_type="image/png",
    )
    assert k == "short-drama/assets/7/scene/99/abc-uuid.png"
