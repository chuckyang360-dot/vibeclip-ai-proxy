"""Minimal Cloudflare R2 (S3-compatible) upload — shared helpers."""

from __future__ import annotations

import os
from typing import NamedTuple

import boto3
from botocore.config import Config


class R2Settings(NamedTuple):
    endpoint: str
    access_key_id: str
    secret_access_key: str
    bucket: str
    public_base_url: str


def _strip(key: str) -> str:
    v = os.getenv(key)
    return (v or "").strip()


def resolve_r2_endpoint() -> str:
    ep = _strip("R2_ENDPOINT")
    if ep:
        return ep.rstrip("/")
    account = _strip("R2_ACCOUNT_ID")
    if account:
        return f"https://{account}.r2.cloudflarestorage.com"
    return ""


def load_r2_settings() -> R2Settings | None:
    """Supports Railway-style R2_* plus legacy R2_ACCESS_KEY / R2_SECRET_KEY."""

    endpoint = resolve_r2_endpoint()
    access = _strip("R2_ACCESS_KEY_ID") or _strip("R2_ACCESS_KEY")
    secret = _strip("R2_SECRET_ACCESS_KEY") or _strip("R2_SECRET_KEY")
    bucket = _strip("R2_BUCKET_NAME")
    public_base = _strip("R2_PUBLIC_BASE_URL").rstrip("/")
    if not all((endpoint, access, secret, bucket, public_base)):
        return None
    return R2Settings(endpoint, access, secret, bucket, public_base)


def upload_bytes_to_r2(*, object_key: str, data: bytes, content_type: str) -> None:
    s = load_r2_settings()
    if s is None:
        raise RuntimeError("r2_not_configured")
    client = boto3.client(
        "s3",
        endpoint_url=s.endpoint,
        aws_access_key_id=s.access_key_id,
        aws_secret_access_key=s.secret_access_key,
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )
    client.put_object(Bucket=s.bucket, Key=object_key, Body=data, ContentType=content_type)
