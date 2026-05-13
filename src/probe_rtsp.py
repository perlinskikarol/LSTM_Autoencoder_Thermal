from __future__ import annotations

import cv2
from urllib.parse import quote

from src.config import load_settings


def build_rtsp_url(user: str, password: str, ip: str, port: int, channel_id: int) -> str:
    user_q = quote(user, safe="")
    pass_q = quote(password, safe="")
    return f"rtsp://{user_q}:{pass_q}@{ip}:{port}/Streaming/Channels/{channel_id}"


def probe() -> None:
    cfg = load_settings()
    candidates = [101, 102, 201, 202]

    print("RTSP probe results:")
    for channel_id in candidates:
        url = build_rtsp_url(cfg.hik_user, cfg.hik_pass, cfg.hik_ip, cfg.rtsp_port, channel_id)
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        ok = cap.isOpened()
        frame_ok = False
        if ok:
            frame_ok, _ = cap.read()
        cap.release()
        status = "OK" if ok and frame_ok else "FAIL"
        print(f"  {channel_id}: {status} -> {url}")


if __name__ == "__main__":
    probe()
