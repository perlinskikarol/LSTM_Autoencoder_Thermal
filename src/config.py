from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import quote

from dotenv import load_dotenv


def _as_int(value: str, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: str, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    text = value.strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


@dataclass(frozen=True)
class Settings:
    hik_ip: str
    hik_user: str
    hik_pass: str
    rtsp_port: int
    rtsp_rgb: str
    rtsp_th: str
    channel_id_for_metadata: int
    reconnect_delay_sec: float = 2.0
    enable_metadata: bool = False
    metadata_mode: str = "legacy"
    metadata_legacy_uri: str = ""
    metadata_http_endpoint: str = ""
    metadata_retry_sec: float = 20.0
    metadata_auth_lockout_sec: float = 1200.0
    metadata_max_auth_failures: int = 1

    @property
    def base_http_url(self) -> str:
        return f"http://{self.hik_ip}"

    @property
    def base_rtsp_url(self) -> str:
        return f"rtsp://{self.hik_ip}:{self.rtsp_port}"


def _default_rtsp(user: str, password: str, ip: str, port: int, channel_id: int) -> str:
    user_q = quote(user, safe="")
    pass_q = quote(password, safe="")
    return f"rtsp://{user_q}:{pass_q}@{ip}:{port}/Streaming/Channels/{channel_id}"


def _is_template_rtsp(value: str) -> bool:
    upper = value.upper()
    if "RTSP://USER:PASS@IP" in upper:
        return True
    if "//USER:" in upper and "@IP:" in upper:
        return True
    return False


def load_settings() -> Settings:
    load_dotenv()

    hik_ip = os.getenv("HIK_IP", "")
    hik_user = os.getenv("HIK_USER", "")
    hik_pass = os.getenv("HIK_PASS", "")
    rtsp_port = _as_int(os.getenv("RTSP_PORT", "554"), 554)
    rtsp_rgb_env = (os.getenv("RTSP_RGB") or "").strip()
    rtsp_th_env = (os.getenv("RTSP_TH") or "").strip()
    channel_id_for_metadata = _as_int(os.getenv("CHANNEL_ID_FOR_METADATA", "101"), 101)
    enable_metadata = _as_bool(os.getenv("ENABLE_METADATA"), False)
    metadata_mode = (os.getenv("METADATA_MODE") or "legacy").strip().lower()
    if metadata_mode not in {"legacy", "auto", "http_thermal", "http_thermal_p2p"}:
        metadata_mode = "legacy"
    metadata_legacy_uri = (os.getenv("METADATA_LEGACY_URI") or "").strip()
    metadata_http_endpoint = (os.getenv("METADATA_HTTP_ENDPOINT") or "").strip()
    metadata_retry_sec = _as_float(os.getenv("METADATA_RETRY_SEC", "20"), 20.0)
    metadata_auth_lockout_sec = _as_float(os.getenv("METADATA_AUTH_LOCKOUT_SEC", "1200"), 1200.0)
    metadata_max_auth_failures = max(1, _as_int(os.getenv("METADATA_MAX_AUTH_FAILURES", "1"), 1))

    missing = [
        name
        for name, value in {
            "HIK_IP": hik_ip,
            "HIK_USER": hik_user,
            "HIK_PASS": hik_pass,
        }.items()
        if not value
    ]
    if missing:
        raise ValueError(f"Missing env values: {', '.join(missing)}")

    if hik_pass.lower() in {"your_password", "password", "pass", "changeme"}:
        raise ValueError("Invalid HIK_PASS in .env (placeholder value detected). Set real camera password.")

    rtsp_rgb = rtsp_rgb_env
    rtsp_th = rtsp_th_env
    if not rtsp_rgb or _is_template_rtsp(rtsp_rgb):
        rtsp_rgb = _default_rtsp(hik_user, hik_pass, hik_ip, rtsp_port, 101)
    if not rtsp_th or _is_template_rtsp(rtsp_th):
        rtsp_th = _default_rtsp(hik_user, hik_pass, hik_ip, rtsp_port, 201)

    return Settings(
        hik_ip=hik_ip,
        hik_user=hik_user,
        hik_pass=hik_pass,
        rtsp_port=rtsp_port,
        rtsp_rgb=rtsp_rgb,
        rtsp_th=rtsp_th,
        channel_id_for_metadata=channel_id_for_metadata,
        enable_metadata=enable_metadata,
        metadata_mode=metadata_mode,
        metadata_legacy_uri=metadata_legacy_uri,
        metadata_http_endpoint=metadata_http_endpoint,
        metadata_retry_sec=metadata_retry_sec,
        metadata_auth_lockout_sec=metadata_auth_lockout_sec,
        metadata_max_auth_failures=metadata_max_auth_failures,
    )
