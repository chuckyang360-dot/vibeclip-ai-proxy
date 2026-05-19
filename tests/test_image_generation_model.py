"""Unit tests for /images/generations model resolution (no HTTP)."""

from __future__ import annotations

import pytest

import main


@pytest.fixture(autouse=True)
def clear_image_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "XAI_IMAGE_MODEL",
        "SHORT_DRAMA_XAI_IMAGE_MODEL",
        "IMAGE_MODEL",
        "XAI_MODEL",
        "OPENAI_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)


def test_body_model_forwarded_not_overridden_by_xai_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XAI_MODEL", "grok-4.20")
    model, src = main.resolve_image_generation_model("grok-imagine-image")
    assert model == "grok-imagine-image"
    assert src == "request_body"


def test_body_model_whitespace_trimmed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XAI_MODEL", "grok-4.20")
    model, src = main.resolve_image_generation_model("  grok-imagine-image  ")
    assert model == "grok-imagine-image"
    assert src == "request_body"


def test_fallback_xai_image_model_when_body_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XAI_IMAGE_MODEL", "grok-imagine-image")
    monkeypatch.setenv("XAI_MODEL", "grok-4.20")
    model, src = main.resolve_image_generation_model(None)
    assert model == "grok-imagine-image"
    assert src == "XAI_IMAGE_MODEL"

    model2, src2 = main.resolve_image_generation_model("")
    assert model2 == "grok-imagine-image"
    assert src2 == "XAI_IMAGE_MODEL"


def test_image_model_chain_never_reads_xai_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XAI_MODEL", "grok-4.20")
    model, src = main.resolve_image_generation_model(None)
    assert model == "grok-imagine-image"
    assert src == "default_grok_imagine_image"


def test_priority_image_model_over_legacy_short_drama(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XAI_IMAGE_MODEL", "from-xai-image")
    monkeypatch.setenv("SHORT_DRAMA_XAI_IMAGE_MODEL", "from-short-drama")
    model, src = main.resolve_image_generation_model(None)
    assert model == "from-xai-image"
    assert src == "XAI_IMAGE_MODEL"


def test_priority_short_drama_over_image_model_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHORT_DRAMA_XAI_IMAGE_MODEL", "from-short-drama")
    monkeypatch.setenv("IMAGE_MODEL", "from-image-env")
    model, src = main.resolve_image_generation_model(None)
    assert model == "from-short-drama"
    assert src == "SHORT_DRAMA_XAI_IMAGE_MODEL"


def test_image_model_env_generic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMAGE_MODEL", "custom-img")
    model, src = main.resolve_image_generation_model(None)
    assert model == "custom-img"
    assert src == "IMAGE_MODEL"


def test_text_resolve_does_not_use_xai_image_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XAI_TEXT_MODEL", raising=False)
    monkeypatch.delenv("XAI_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.setenv("XAI_IMAGE_MODEL", "grok-imagine-image")
    model, key = main.resolve_text_chat_model()
    assert model == ""
    assert key == ""
