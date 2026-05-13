from __future__ import annotations

import csv
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import messagebox, ttk

from src.config import load_settings
from src.hik_isapi_metadata import HikMetadataClient, TempReading, ThermalPixelFrame
from src.mp_forehead_capture import (
    HikThermometryRegionSync,
    _load_homography_from_env,
    _map_rgb_roi_to_thermal,
    _parse_bool_env,
    _parse_float_env,
    _parse_int_env,
    _select_temperature_for_roi,
)
from src.roi_provider import MediaPipeForeheadRoiProvider, RoiBox
from src.rtsp_stream import RtspStream


SESSION_STATE_OPTIONS = ("normalny", "prysznic", "trening")


def _fit_bgr_to_tk(frame_bgr: np.ndarray, target_w: int, target_h: int) -> ImageTk.PhotoImage:
    src_h, src_w = frame_bgr.shape[:2]
    if src_w <= 0 or src_h <= 0:
        raise ValueError("Invalid frame size.")
    target_w = max(1, int(target_w))
    target_h = max(1, int(target_h))
    scale = min(target_w / float(src_w), target_h / float(src_h))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    frame_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    return ImageTk.PhotoImage(image=img)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _draw_hud_text(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    font_scale: float,
    thickness: int,
) -> None:
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        max(1, thickness + 2),
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _sanitize_path_component(value: str) -> str:
    text = value.strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r'[<>:"/\\|?*]', "_", text)
    text = re.sub(r"_+", "_", text)
    text = text.strip(" ._")
    return text or "unknown"


def _build_session_csv_path(subject_id: str, session_state: str, session_date: str, start_time: str) -> Path:
    subject_dir_name = _sanitize_path_component(subject_id)
    state_name = _sanitize_path_component(session_state.lower())
    date_name = _sanitize_path_component(session_date)
    time_name = _sanitize_path_component(start_time)

    base_dir = Path("Pomiary") / subject_dir_name
    base_dir.mkdir(parents=True, exist_ok=True)

    stem_base = f"{subject_dir_name}_{state_name}_{date_name}_{time_name}"
    csv_path = base_dir / f"{stem_base}.csv"
    suffix_idx = 1
    while csv_path.exists():
        suffix_idx += 1
        csv_path = base_dir / f"{stem_base}_{suffix_idx:02d}.csv"
    return csv_path


def _clip_roi(box: RoiBox, width: int, height: int) -> Optional[RoiBox]:
    x1 = max(0, min(width - 1, box.x1))
    y1 = max(0, min(height - 1, box.y1))
    x2 = max(0, min(width - 1, box.x2))
    y2 = max(0, min(height - 1, box.y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return RoiBox(x1=x1, y1=y1, x2=x2, y2=y2)


@dataclass(frozen=True)
class UiSession:
    session_date: str
    session_start_time: str
    subject_id: str
    session_state: str
    csv_path: Path


@dataclass(frozen=True)
class RoiTemperatureStats:
    timestamp: str
    temp_unit: str
    mean_temp_c: float
    min_temp_c: float
    max_temp_c: float


class SessionTrainingCsvSink:
    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self.csv_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fp)
        self._writer.writerow(
            [
                "sample_index",
                "date",
                "time",
                "elapsed_sec",
                "subject_id",
                "session_state",
                "mean_temp_c",
                "min_temp_c",
                "max_temp_c",
            ]
        )
        self._fp.flush()

    def append(
        self,
        session: UiSession,
        sample_index: int,
        sample_dt: datetime,
        elapsed_sec: float,
        stats: Optional[RoiTemperatureStats],
    ) -> None:
        self._writer.writerow(
            [
                sample_index,
                sample_dt.strftime("%d.%m.%Y"),
                sample_dt.strftime("%H:%M:%S"),
                f"{elapsed_sec:.3f}",
                session.subject_id,
                session.session_state,
                "" if stats is None else f"{stats.mean_temp_c:.3f}",
                "" if stats is None else f"{stats.min_temp_c:.3f}",
                "" if stats is None else f"{stats.max_temp_c:.3f}",
            ]
        )
        self._fp.flush()

    def close(self) -> None:
        self._fp.close()


def _extract_p2p_roi_stats(
    pixel_frame: Optional[ThermalPixelFrame],
    thermal_roi: RoiBox,
    thermal_shape: tuple[int, int, int],
    temp_min_c: float,
    temp_max_c: float,
    trim_low_pct: float,
    trim_high_pct: float,
    min_valid_pixels: int,
) -> tuple[Optional[RoiTemperatureStats], str]:
    if pixel_frame is None:
        return None, "no_p2p_frame"

    th_h, th_w = thermal_shape[:2]
    px_h, px_w = pixel_frame.temp_c.shape[:2]
    if th_w <= 1 or th_h <= 1 or px_w <= 1 or px_h <= 1:
        return None, "invalid_p2p_shape"

    def map_x(x: int) -> int:
        return int(round((max(0, min(th_w - 1, x)) / float(th_w - 1)) * (px_w - 1)))

    def map_y(y: int) -> int:
        return int(round((max(0, min(th_h - 1, y)) / float(th_h - 1)) * (px_h - 1)))

    px_roi = RoiBox(
        x1=map_x(thermal_roi.x1),
        y1=map_y(thermal_roi.y1),
        x2=map_x(thermal_roi.x2),
        y2=map_y(thermal_roi.y2),
    )
    px_roi = _clip_roi(px_roi, width=px_w, height=px_h)
    if px_roi is None:
        return None, "p2p_roi_oob"

    roi_matrix = pixel_frame.temp_c[px_roi.y1 : px_roi.y2, px_roi.x1 : px_roi.x2]
    finite = roi_matrix[np.isfinite(roi_matrix)]
    finite = finite[(finite >= float(temp_min_c)) & (finite <= float(temp_max_c))]

    min_valid_pixels = max(1, int(min_valid_pixels))
    if finite.size >= max(20, min_valid_pixels):
        lo = float(np.percentile(finite, max(0.0, min(49.0, float(trim_low_pct)))))
        hi = float(np.percentile(finite, min(100.0, max(51.0, 100.0 - float(trim_high_pct)))))
        if hi > lo:
            finite = finite[(finite >= lo) & (finite <= hi)]

    if finite.size == 0:
        return None, "no_p2p_pixels"
    if finite.size < min_valid_pixels:
        return None, "insufficient_p2p_pixels"

    return (
        RoiTemperatureStats(
            timestamp=pixel_frame.timestamp,
            temp_unit="centigrade",
            mean_temp_c=float(np.mean(finite)),
            min_temp_c=float(np.min(finite)),
            max_temp_c=float(np.max(finite)),
        ),
        "ok",
    )


def _extract_rule_roi_stats(
    readings: list[TempReading],
    thermal_roi: RoiBox,
    thermal_shape: tuple[int, int, int],
    min_iou_threshold: float,
    strict_roi_only: bool,
) -> tuple[Optional[RoiTemperatureStats], str]:
    avg_match = _select_temperature_for_roi(
        readings=readings,
        thermal_roi=thermal_roi,
        thermal_shape=thermal_shape,
        preferred_property="average",
        min_iou_threshold=min_iou_threshold,
        strict_roi_only=strict_roi_only,
    )
    if avg_match.reading is None:
        return None, avg_match.source or "no_temperature_match"

    hi_match = _select_temperature_for_roi(
        readings=readings,
        thermal_roi=thermal_roi,
        thermal_shape=thermal_shape,
        preferred_property="highest",
        min_iou_threshold=min_iou_threshold,
        strict_roi_only=strict_roi_only,
    )
    lo_match = _select_temperature_for_roi(
        readings=readings,
        thermal_roi=thermal_roi,
        thermal_shape=thermal_shape,
        preferred_property="lowest",
        min_iou_threshold=min_iou_threshold,
        strict_roi_only=strict_roi_only,
    )

    mean_value = float(avg_match.reading.temp_value)
    max_value = float(hi_match.reading.temp_value) if hi_match.reading is not None else mean_value
    min_value = float(lo_match.reading.temp_value) if lo_match.reading is not None else mean_value

    return (
        RoiTemperatureStats(
            timestamp=avg_match.reading.timestamp,
            temp_unit=avg_match.reading.temp_unit,
            mean_temp_c=mean_value,
            min_temp_c=min_value,
            max_temp_c=max_value,
        ),
        "ok",
    )


class MeasurementEngine:
    def __init__(self, session: UiSession) -> None:
        self.session = session
        self.cfg = load_settings()

        self.min_detection_conf = _parse_float_env("MP_MIN_DETECTION_CONF", 0.60)
        self.sample_period_sec = max(0.2, _parse_float_env("SAMPLE_PERIOD_SEC", 1.0))
        self.roi_match_min_iou = max(0.0, min(1.0, _parse_float_env("ROI_MATCH_MIN_IOU", 0.15)))
        self.strict_roi_only = _parse_bool_env("STRICT_ROI_ONLY", True)
        self.mp_face_model_path = (os.getenv("MP_FACE_MODEL_PATH") or "").strip()
        self.mp_model_selection = max(0, min(1, _parse_int_env("MP_MODEL_SELECTION", 1)))
        self.mp_detection_input_scale = max(1.0, _parse_float_env("MP_DETECTION_INPUT_SCALE", 1.5))
        self.mp_face_top_expand_ratio = _parse_float_env("MP_FACE_TOP_EXPAND_RATIO", 0.35)
        self.thermal_coverage_x = _parse_float_env("THERMAL_COVERAGE_X", 1.0)
        self.thermal_coverage_y = _parse_float_env("THERMAL_COVERAGE_Y", 1.0)
        self.thermal_shift_x_ratio = _parse_float_env("THERMAL_ROI_SHIFT_X_RATIO", 0.0)
        self.thermal_shift_y_ratio = _parse_float_env("THERMAL_ROI_SHIFT_Y_RATIO", 0.0)
        self.thermal_scale_x = _parse_float_env("THERMAL_ROI_SCALE_X", 1.0)
        self.thermal_scale_y = _parse_float_env("THERMAL_ROI_SCALE_Y", 1.0)
        self.p2p_temp_min_c = _parse_float_env("P2P_TEMP_MIN_C", 32.0)
        self.p2p_temp_max_c = _parse_float_env("P2P_TEMP_MAX_C", 43.5)
        self.p2p_trim_low_pct = _parse_float_env("P2P_TRIM_LOW_PCT", 5.0)
        self.p2p_trim_high_pct = _parse_float_env("P2P_TRIM_HIGH_PCT", 5.0)
        self.p2p_min_valid_pixels = max(1, _parse_int_env("P2P_MIN_VALID_PIXELS", 25))
        self.dynamic_region_enabled = _parse_bool_env("DYNAMIC_THERMOMETRY_REGION_ENABLED", False)
        self.dynamic_region_scene_id = _parse_int_env("DYNAMIC_THERMOMETRY_SCENE_ID", 1)
        self.dynamic_region_id = _parse_int_env("DYNAMIC_THERMOMETRY_REGION_ID", 1)
        self.dynamic_region_update_sec = max(0.2, _parse_float_env("DYNAMIC_THERMOMETRY_UPDATE_SEC", 2.0))
        self.dynamic_region_min_move_norm = max(0.0, _parse_float_env("DYNAMIC_THERMOMETRY_MIN_MOVE_NORM", 0.03))
        self.dynamic_region_max_failures = max(1, _parse_int_env("DYNAMIC_THERMOMETRY_MAX_FAILURES", 1))
        self.dynamic_region_failure_backoff_sec = max(
            1.0, _parse_float_env("DYNAMIC_THERMOMETRY_FAILURE_BACKOFF_SEC", 10.0)
        )
        self.homography = _load_homography_from_env()

        self.current_shift_x_ratio = self.thermal_shift_x_ratio
        self.current_shift_y_ratio = self.thermal_shift_y_ratio
        self.current_scale_x = self.thermal_scale_x
        self.current_scale_y = self.thermal_scale_y
        self.tune_step_shift = 0.005
        self.tune_step_scale = 0.02

        self.rgb_stream = RtspStream(name="RGB", rtsp_url=self.cfg.rtsp_rgb, reconnect_delay_sec=self.cfg.reconnect_delay_sec)
        self.th_stream = RtspStream(name="THERMAL", rtsp_url=self.cfg.rtsp_th, reconnect_delay_sec=self.cfg.reconnect_delay_sec)
        self.metadata = HikMetadataClient(
            ip=self.cfg.hik_ip,
            user=self.cfg.hik_user,
            password=self.cfg.hik_pass,
            channel_id=self.cfg.channel_id_for_metadata,
            rtsp_port=self.cfg.rtsp_port,
            reconnect_delay_sec=self.cfg.metadata_retry_sec,
            mode=self.cfg.metadata_mode,
            forced_legacy_uri=self.cfg.metadata_legacy_uri,
            forced_http_endpoint=self.cfg.metadata_http_endpoint,
            auth_lockout_sleep_sec=self.cfg.metadata_auth_lockout_sec,
            max_auth_failures=self.cfg.metadata_max_auth_failures,
        )
        self.roi_provider = MediaPipeForeheadRoiProvider(
            min_detection_confidence=self.min_detection_conf,
            model_selection=self.mp_model_selection,
            model_asset_path=self.mp_face_model_path,
            face_top_expand_ratio=self.mp_face_top_expand_ratio,
            detection_input_scale=self.mp_detection_input_scale,
        )
        self.training_csv_sink = SessionTrainingCsvSink(self.session.csv_path)
        self.dynamic_region_sync = HikThermometryRegionSync(
            ip=self.cfg.hik_ip,
            user=self.cfg.hik_user,
            password=self.cfg.hik_pass,
            channel_id=self.cfg.channel_id_for_metadata,
            scene_id=self.dynamic_region_scene_id,
            region_id=self.dynamic_region_id,
            enabled=self.dynamic_region_enabled,
            min_update_period_sec=self.dynamic_region_update_sec,
            min_move_norm=self.dynamic_region_min_move_norm,
            max_consecutive_failures=self.dynamic_region_max_failures,
            failure_backoff_sec=self.dynamic_region_failure_backoff_sec,
        )

        self.sample_index = 0
        self.session_start_dt: Optional[datetime] = None
        self.session_start_monotonic = 0.0
        self.next_sample_due_monotonic = 0.0
        self.overlay_text = "temp: n/a"
        self.last_rgb_frame: Optional[np.ndarray] = None
        self.last_th_frame: Optional[np.ndarray] = None
        self.last_status = "idle"
        self.running = False

        logging.info("Training CSV path: %s", self.session.csv_path)

    def start(self) -> None:
        if self.running:
            return
        now = datetime.now()
        session_date_value = datetime.strptime(self.session.session_date, "%d.%m.%Y").date()
        self.session_start_dt = datetime.combine(session_date_value, now.time())
        self.session_start_monotonic = time.monotonic()
        self.next_sample_due_monotonic = self.session_start_monotonic
        self.sample_index = 0
        self.rgb_stream.start()
        self.th_stream.start()
        self.metadata.start()
        self.running = True
        self.last_status = "running"

    def stop(self) -> None:
        if not self.running:
            return
        self.running = False
        for stopper in (self.metadata.stop, self.rgb_stream.stop, self.th_stream.stop):
            try:
                stopper()
            except Exception:
                pass
        try:
            self.roi_provider.close()
        except Exception:
            pass
        try:
            self.dynamic_region_sync.close()
        except Exception:
            pass
        self.training_csv_sink.close()
        self.last_status = "stopped"

    def handle_key(self, key: str) -> bool:
        if not key:
            return False
        ch = key.lower()
        if ch == "w":
            self.current_shift_y_ratio = _clamp(self.current_shift_y_ratio - self.tune_step_shift, -0.5, 0.5)
        elif ch == "s":
            self.current_shift_y_ratio = _clamp(self.current_shift_y_ratio + self.tune_step_shift, -0.5, 0.5)
        elif ch == "a":
            self.current_shift_x_ratio = _clamp(self.current_shift_x_ratio - self.tune_step_shift, -0.5, 0.5)
        elif ch == "d":
            self.current_shift_x_ratio = _clamp(self.current_shift_x_ratio + self.tune_step_shift, -0.5, 0.5)
        elif ch == "j":
            self.current_scale_x = _clamp(self.current_scale_x - self.tune_step_scale, 0.2, 2.5)
        elif ch == "l":
            self.current_scale_x = _clamp(self.current_scale_x + self.tune_step_scale, 0.2, 2.5)
        elif ch == "k":
            self.current_scale_y = _clamp(self.current_scale_y - self.tune_step_scale, 0.2, 2.5)
        elif ch == "i":
            self.current_scale_y = _clamp(self.current_scale_y + self.tune_step_scale, 0.2, 2.5)
        elif ch == "[":
            self.tune_step_shift = _clamp(self.tune_step_shift * 0.5, 0.001, 0.05)
        elif ch == "]":
            self.tune_step_shift = _clamp(self.tune_step_shift * 2.0, 0.001, 0.05)
        elif ch == "-":
            self.tune_step_scale = _clamp(self.tune_step_scale * 0.5, 0.005, 0.20)
        elif ch == "=":
            self.tune_step_scale = _clamp(self.tune_step_scale * 2.0, 0.005, 0.20)
        elif ch == "r":
            self.current_shift_x_ratio = self.thermal_shift_x_ratio
            self.current_shift_y_ratio = self.thermal_shift_y_ratio
            self.current_scale_x = self.thermal_scale_x
            self.current_scale_y = self.thermal_scale_y
            self.tune_step_shift = 0.005
            self.tune_step_scale = 0.02
        elif ch == "p":
            logging.info("Paste this to .env:")
            logging.info("THERMAL_ROI_SHIFT_X_RATIO=%.4f", self.current_shift_x_ratio)
            logging.info("THERMAL_ROI_SHIFT_Y_RATIO=%.4f", self.current_shift_y_ratio)
            logging.info("THERMAL_ROI_SCALE_X=%.4f", self.current_scale_x)
            logging.info("THERMAL_ROI_SCALE_Y=%.4f", self.current_scale_y)
        else:
            return False
        return True

    def step(self) -> None:
        if not self.running:
            return

        rgb_frame = self.rgb_stream.get_last_frame()
        th_frame = self.th_stream.get_last_frame()

        rgb_roi: Optional[RoiBox] = None
        thermal_roi: Optional[RoiBox] = None
        roi_stats: Optional[RoiTemperatureStats] = None
        stats_status = "no_temperature_match"

        if rgb_frame is not None:
            rgb_roi = self.roi_provider.get_forehead_roi(rgb_frame)
            if self.roi_provider.last_face_box is not None:
                face = self.roi_provider.last_face_box
                cv2.rectangle(rgb_frame, (face.x1, face.y1), (face.x2, face.y2), (255, 170, 0), 2)
            if rgb_roi is not None:
                cv2.rectangle(rgb_frame, (rgb_roi.x1, rgb_roi.y1), (rgb_roi.x2, rgb_roi.y2), (0, 255, 0), 2)
            self.last_rgb_frame = rgb_frame.copy()

        if th_frame is not None and rgb_frame is not None and rgb_roi is not None:
            thermal_roi = _map_rgb_roi_to_thermal(
                rgb_roi=rgb_roi,
                rgb_shape=rgb_frame.shape,
                thermal_shape=th_frame.shape,
                homography=self.homography,
                thermal_coverage_x=self.thermal_coverage_x,
                thermal_coverage_y=self.thermal_coverage_y,
                shift_x_ratio=self.current_shift_x_ratio,
                shift_y_ratio=self.current_shift_y_ratio,
                thermal_scale_x=self.current_scale_x,
                thermal_scale_y=self.current_scale_y,
            )
            if thermal_roi is not None:
                self.dynamic_region_sync.sync_roi_if_needed(
                    roi=thermal_roi,
                    thermal_shape=th_frame.shape,
                    now_sec=time.time(),
                )
                if self.cfg.metadata_mode == "http_thermal_p2p":
                    roi_stats, stats_status = _extract_p2p_roi_stats(
                        pixel_frame=self.metadata.get_latest_pixel_frame(),
                        thermal_roi=thermal_roi,
                        thermal_shape=th_frame.shape,
                        temp_min_c=self.p2p_temp_min_c,
                        temp_max_c=self.p2p_temp_max_c,
                        trim_low_pct=self.p2p_trim_low_pct,
                        trim_high_pct=self.p2p_trim_high_pct,
                        min_valid_pixels=self.p2p_min_valid_pixels,
                    )
                else:
                    roi_stats, stats_status = _extract_rule_roi_stats(
                        readings=self.metadata.get_latest_temperatures(),
                        thermal_roi=thermal_roi,
                        thermal_shape=th_frame.shape,
                        min_iou_threshold=self.roi_match_min_iou,
                        strict_roi_only=self.strict_roi_only,
                    )
                cv2.rectangle(th_frame, (thermal_roi.x1, thermal_roi.y1), (thermal_roi.x2, thermal_roi.y2), (0, 255, 0), 2)

            if roi_stats is not None:
                self.overlay_text = (
                    f"mean={roi_stats.mean_temp_c:.2f}C  "
                    f"min={roi_stats.min_temp_c:.2f}C  "
                    f"max={roi_stats.max_temp_c:.2f}C"
                )
            else:
                self.overlay_text = f"temp: n/a ({stats_status})"

        if th_frame is not None:
            _draw_hud_text(
                th_frame,
                self.overlay_text,
                x=20,
                y=86,
                color=(0, 255, 255),
                font_scale=0.75,
                thickness=2,
            )
            _draw_hud_text(
                th_frame,
                (
                    f"shift=({self.current_shift_x_ratio:+.3f},{self.current_shift_y_ratio:+.3f}) "
                    f"scale=({self.current_scale_x:.3f},{self.current_scale_y:.3f})"
                ),
                x=20,
                y=114,
                color=(200, 255, 200),
                font_scale=0.52,
                thickness=1,
            )
            self.last_th_frame = th_frame.copy()

        now_monotonic = time.monotonic()
        while now_monotonic >= self.next_sample_due_monotonic:
            status = "ok"
            if rgb_frame is None:
                status = "no_rgb_frame"
            elif th_frame is None:
                status = "no_thermal_frame"
            elif rgb_roi is None:
                status = "no_face_roi"
            elif thermal_roi is None:
                status = "no_mapped_thermal_roi"
            elif roi_stats is None:
                status = stats_status

            elapsed_sec = self.sample_index * self.sample_period_sec
            assert self.session_start_dt is not None
            sample_dt = self.session_start_dt + timedelta(seconds=elapsed_sec)
            self.training_csv_sink.append(
                session=self.session,
                sample_index=self.sample_index,
                sample_dt=sample_dt,
                elapsed_sec=elapsed_sec,
                stats=roi_stats if status == "ok" else None,
            )

            self.last_status = status
            self.sample_index += 1
            self.next_sample_due_monotonic = self.session_start_monotonic + (
                self.sample_index * self.sample_period_sec
            )


class MpForeheadUiApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Forehead Thermometry UI")
        self.geometry("1760x1020")
        self.configure(bg="#1e1e1e")
        self.rgb_view_w = 960
        self.rgb_view_h = 250
        self.th_view_w = 960
        self.th_view_h = 625

        self.engine: Optional[MeasurementEngine] = None
        self.current_session: Optional[UiSession] = None
        self._rgb_img: Optional[ImageTk.PhotoImage] = None
        self._th_img: Optional[ImageTk.PhotoImage] = None

        self._build_ui()
        self.bind_all("<KeyPress>", self._on_key)
        self.after(50, self._tick)

    def _build_ui(self) -> None:
        top = tk.Frame(self, bg="#1e1e1e")
        top.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(top, text="Data (DD.MM.YYYY):", fg="white", bg="#1e1e1e").pack(side=tk.LEFT)
        self.date_var = tk.StringVar(value=datetime.now().strftime("%d.%m.%Y"))
        self.date_entry = tk.Entry(top, textvariable=self.date_var, width=14)
        self.date_entry.pack(side=tk.LEFT, padx=6)

        tk.Label(top, text="Osoba / ID:", fg="white", bg="#1e1e1e").pack(side=tk.LEFT, padx=(12, 0))
        self.person_var = tk.StringVar(value="")
        self.person_entry = tk.Entry(top, textvariable=self.person_var, width=24)
        self.person_entry.pack(side=tk.LEFT, padx=6)

        tk.Label(top, text="Stan:", fg="white", bg="#1e1e1e").pack(side=tk.LEFT, padx=(12, 0))
        self.state_var = tk.StringVar(value=SESSION_STATE_OPTIONS[0])
        self.state_combo = ttk.Combobox(
            top,
            textvariable=self.state_var,
            values=SESSION_STATE_OPTIONS,
            state="readonly",
            width=14,
        )
        self.state_combo.pack(side=tk.LEFT, padx=6)

        self.start_btn = tk.Button(top, text="Rozpocznij pomiar", command=self._start_measurement, bg="#2d8a4e", fg="white")
        self.start_btn.pack(side=tk.LEFT, padx=(12, 6))
        self.stop_btn = tk.Button(top, text="Zatrzymaj pomiar", command=self._stop_measurement, bg="#b34747", fg="white", state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=6)

        self.status_var = tk.StringVar(value="Status: idle")
        tk.Label(top, textvariable=self.status_var, fg="#d0d0d0", bg="#1e1e1e").pack(side=tk.LEFT, padx=16)

        body = tk.Frame(self, bg="#1e1e1e")
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        left = tk.Frame(body, bg="#1e1e1e")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right = tk.Frame(body, bg="#1e1e1e")
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=14)

        tk.Label(left, text="RGB", fg="white", bg="#1e1e1e", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.rgb_container = tk.Frame(left, bg="black", width=self.rgb_view_w, height=self.rgb_view_h)
        self.rgb_container.pack(anchor="w", pady=(4, 12))
        self.rgb_container.pack_propagate(False)
        self.rgb_label = tk.Label(self.rgb_container, bg="black")
        self.rgb_label.pack(fill=tk.BOTH, expand=True)

        tk.Label(left, text="Thermal", fg="white", bg="#1e1e1e", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.th_container = tk.Frame(left, bg="black", width=self.th_view_w, height=self.th_view_h)
        self.th_container.pack(anchor="w", pady=(4, 0))
        self.th_container.pack_propagate(False)
        self.th_label = tk.Label(self.th_container, bg="black")
        self.th_label.pack(fill=tk.BOTH, expand=True)

        controls_text = (
            "Sterowanie reczne ROI (dziala w trakcie pomiaru):\n"
            "W/S: przesuniecie gora/dol\n"
            "A/D: przesuniecie lewo/prawo\n"
            "J/L: skala X - / +\n"
            "K/I: skala Y - / +\n"
            "[ / ]: krok przesuniecia - / +\n"
            "- / =: krok skali - / +\n"
            "R: reset strojenia do wartosci z .env\n"
            "P: wypisz aktualne THERMAL_ROI_* do logu\n"
            "Q: zatrzymaj pomiar\n"
        )
        tk.Label(
            right,
            text=controls_text,
            justify=tk.LEFT,
            fg="#d7f0d7",
            bg="#1e1e1e",
            font=("Consolas", 11),
        ).pack(anchor="n")

    def _set_session_controls_enabled(self, enabled: bool) -> None:
        entry_state = tk.NORMAL if enabled else tk.DISABLED
        combo_state = "readonly" if enabled else "disabled"
        self.date_entry.configure(state=entry_state)
        self.person_entry.configure(state=entry_state)
        self.state_combo.configure(state=combo_state)

    def _validate_form(self) -> UiSession:
        date_text = self.date_var.get().strip()
        person_text = self.person_var.get().strip()
        state_text = self.state_var.get().strip().lower()

        if not date_text:
            raise ValueError("Podaj date sesji.")
        if not person_text:
            raise ValueError("Podaj osobe / ID.")
        if state_text not in SESSION_STATE_OPTIONS:
            raise ValueError("Wybierz poprawny stan sesji.")

        datetime.strptime(date_text, "%d.%m.%Y")
        session_start_time = datetime.now().strftime("%H.%M.%S")
        csv_path = _build_session_csv_path(
            subject_id=person_text,
            session_state=state_text,
            session_date=date_text,
            start_time=session_start_time,
        )
        return UiSession(
            session_date=date_text,
            session_start_time=session_start_time,
            subject_id=person_text,
            session_state=state_text,
            csv_path=csv_path,
        )

    def _start_measurement(self) -> None:
        try:
            session = self._validate_form()
        except Exception as exc:
            messagebox.showerror("Blad formularza", str(exc))
            return
        if self.engine is not None:
            return
        try:
            self.engine = MeasurementEngine(session=session)
            self.engine.start()
        except Exception as exc:
            self.engine = None
            messagebox.showerror("Blad startu", str(exc))
            return

        self.current_session = session
        self._set_session_controls_enabled(False)
        self.start_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self.focus_set()
        self.status_var.set(
            f"Status: running | osoba={session.subject_id} | stan={session.session_state} | plik={session.csv_path.name}"
        )

    def _stop_measurement(self) -> None:
        if self.engine is None:
            return
        self.engine.stop()
        self.engine = None
        self.current_session = None
        self._set_session_controls_enabled(True)
        self.start_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self.status_var.set("Status: stopped")
        self.person_entry.focus_set()

    def _on_key(self, event):
        if self.engine is None:
            return
        key = (event.char or "").strip()
        if not key:
            return
        if key.lower() == "q":
            self._stop_measurement()
            return "break"
        if self.engine.handle_key(key):
            return "break"
        return None

    def _tick(self) -> None:
        if self.engine is not None:
            try:
                self.engine.step()
                if self.engine.last_rgb_frame is not None:
                    self._rgb_img = _fit_bgr_to_tk(
                        self.engine.last_rgb_frame,
                        self.rgb_view_w,
                        self.rgb_view_h,
                    )
                    self.rgb_label.configure(image=self._rgb_img)
                if self.engine.last_th_frame is not None:
                    self._th_img = _fit_bgr_to_tk(
                        self.engine.last_th_frame,
                        self.th_view_w,
                        self.th_view_h,
                    )
                    self.th_label.configure(image=self._th_img)
                if self.current_session is not None:
                    self.status_var.set(
                        "Status: "
                        f"{self.engine.last_status} | osoba={self.current_session.subject_id} "
                        f"| stan={self.current_session.session_state} | plik={self.current_session.csv_path.name}"
                    )
                else:
                    self.status_var.set(f"Status: {self.engine.last_status}")
            except Exception as exc:
                logging.exception("UI loop error: %s", exc)
                messagebox.showerror("Blad pomiaru", str(exc))
                self._stop_measurement()
        self.after(50, self._tick)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    app = MpForeheadUiApp()
    app.mainloop()


if __name__ == "__main__":
    main()
