from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import cv2


class RtspStream:
    def __init__(self, name: str, rtsp_url: str, reconnect_delay_sec: float = 2.0) -> None:
        self.name = name
        self.rtsp_url = rtsp_url
        self.reconnect_delay_sec = reconnect_delay_sec

        self._cap: Optional[cv2.VideoCapture] = None
        self._last_frame = None
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name=f"RtspStream-{self.name}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        try:
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=0.5)
                if self._thread.is_alive():
                    # Last resort: release capture to unblock lingering read().
                    self._release_capture()
                    self._thread.join(timeout=0.5)
        except KeyboardInterrupt:
            # Allow fast exit when user presses Ctrl+C repeatedly.
            pass
        self._release_capture()

    def get_last_frame(self):
        with self._lock:
            if self._last_frame is None:
                return None
            return self._last_frame.copy()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            if not self._open_capture():
                time.sleep(self.reconnect_delay_sec)
                continue

            while not self._stop_event.is_set():
                if self._cap is None:
                    break
                try:
                    ok, frame = self._cap.read()
                except cv2.error as exc:
                    logging.warning("[%s] OpenCV read exception, reconnecting: %s", self.name, exc)
                    break
                except Exception as exc:  # noqa: BLE001
                    logging.warning("[%s] unexpected read exception, reconnecting: %s", self.name, exc)
                    break
                if not ok or frame is None:
                    logging.warning("[%s] frame read failed, reconnecting", self.name)
                    break

                with self._lock:
                    self._last_frame = frame

            self._release_capture()
            if not self._stop_event.is_set():
                time.sleep(self.reconnect_delay_sec)

    def _open_capture(self) -> bool:
        self._release_capture()
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            logging.warning("[%s] cannot open RTSP: %s", self.name, self.rtsp_url)
            cap.release()
            return False

        # Best effort: keep lag low by forcing small buffer.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap
        logging.info("[%s] connected", self.name)
        return True

    def _release_capture(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
