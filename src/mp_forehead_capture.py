from __future__ import annotations

import csv
import logging
import os
import statistics
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests
from lxml import etree
from requests.auth import HTTPDigestAuth

from src.config import load_settings
from src.hik_isapi_metadata import HikMetadataClient, TempReading, ThermalPixelFrame
from src.roi_provider import MediaPipeForeheadRoiProvider, RoiBox
from src.rtsp_stream import RtspStream


def _parse_float_env(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logging.warning("Invalid %s=%r. Using default=%s", name, raw, default)
        return default


def _parse_int_env(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logging.warning("Invalid %s=%r. Using default=%s", name, raw, default)
        return default


def _parse_bool_env(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    logging.warning("Invalid %s=%r. Using default=%s", name, raw, default)
    return default


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
    # Outline keeps text readable on bright thermal overlays and camera OSD.
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


def _load_homography_from_env() -> Optional[np.ndarray]:
    return _load_homography_from_env_var("RGB_TO_THERMAL_H")


def _load_homography_from_env_var(env_name: str) -> Optional[np.ndarray]:
    raw = (os.getenv(env_name) or "").strip()
    if not raw:
        return None
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    if len(tokens) != 9:
        logging.warning("%s must contain exactly 9 comma-separated values.", env_name)
        return None
    try:
        values = [float(token) for token in tokens]
    except ValueError:
        logging.warning("%s contains non-numeric values.", env_name)
        return None
    return np.array(values, dtype=np.float32).reshape(3, 3)


def _select_active_homography(
    *,
    face_box: Optional[RoiBox],
    rgb_shape: Optional[tuple[int, int, int]],
    default_h: Optional[np.ndarray],
    near_h: Optional[np.ndarray],
    far_h: Optional[np.ndarray],
    switch_enabled: bool,
    near_min_face_width_ratio: float,
    far_max_face_width_ratio: float,
    last_mode: str,
) -> tuple[Optional[np.ndarray], str]:
    if not switch_enabled:
        return default_h, "default"
    if near_h is None and far_h is None:
        return default_h, "default"
    if face_box is None or rgb_shape is None:
        if last_mode == "near" and near_h is not None:
            return near_h, "near"
        if last_mode == "far" and far_h is not None:
            return far_h, "far"
        return default_h, "default"

    rgb_h, rgb_w = rgb_shape[:2]
    if rgb_w <= 0:
        return default_h, "default"
    face_width_ratio = max(0.0, min(1.0, float(face_box.x2 - face_box.x1) / float(rgb_w)))

    # Hysteresis:
    # - switch FAR -> NEAR only above near_min_face_width_ratio
    # - switch NEAR -> FAR only below far_max_face_width_ratio
    if near_h is not None and far_h is not None:
        if last_mode == "near":
            if face_width_ratio <= far_max_face_width_ratio:
                return far_h, "far"
            return near_h, "near"
        if last_mode == "far":
            if face_width_ratio >= near_min_face_width_ratio:
                return near_h, "near"
            return far_h, "far"
        if face_width_ratio >= near_min_face_width_ratio:
            return near_h, "near"
        if face_width_ratio <= far_max_face_width_ratio:
            return far_h, "far"
        return default_h, "default"

    if near_h is not None:
        if face_width_ratio >= near_min_face_width_ratio:
            return near_h, "near"
        return default_h, "default"

    if far_h is not None:
        if face_width_ratio <= far_max_face_width_ratio:
            return far_h, "far"
        return default_h, "default"

    return default_h, "default"


def _clip_box(box: RoiBox, width: int, height: int) -> Optional[RoiBox]:
    x1 = max(0, min(width - 1, box.x1))
    y1 = max(0, min(height - 1, box.y1))
    x2 = max(0, min(width - 1, box.x2))
    y2 = max(0, min(height - 1, box.y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return RoiBox(x1=x1, y1=y1, x2=x2, y2=y2)


def _map_rgb_roi_to_thermal(
    rgb_roi: RoiBox,
    rgb_shape: tuple[int, int, int],
    thermal_shape: tuple[int, int, int],
    homography: Optional[np.ndarray],
    thermal_coverage_x: float = 1.0,
    thermal_coverage_y: float = 1.0,
    shift_x_ratio: float = 0.0,
    shift_y_ratio: float = 0.0,
    thermal_scale_x: float = 1.0,
    thermal_scale_y: float = 1.0,
) -> Optional[RoiBox]:
    rgb_h, rgb_w = rgb_shape[:2]
    th_h, th_w = thermal_shape[:2]
    if rgb_w <= 0 or rgb_h <= 0 or th_w <= 0 or th_h <= 0:
        return None

    if homography is not None:
        points = np.array(
            [
                [rgb_roi.x1, rgb_roi.y1],
                [rgb_roi.x2, rgb_roi.y1],
                [rgb_roi.x2, rgb_roi.y2],
                [rgb_roi.x1, rgb_roi.y2],
            ],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        mapped = cv2.perspectiveTransform(points, homography).reshape(-1, 2)
        xs = mapped[:, 0]
        ys = mapped[:, 1]
        mapped_box = RoiBox(
            x1=int(np.floor(np.min(xs))),
            y1=int(np.floor(np.min(ys))),
            x2=int(np.ceil(np.max(xs))),
            y2=int(np.ceil(np.max(ys))),
        )
        return _apply_thermal_roi_adjustment(
            mapped_box=mapped_box,
            width=th_w,
            height=th_h,
            shift_x_ratio=shift_x_ratio,
            shift_y_ratio=shift_y_ratio,
            scale_x=thermal_scale_x,
            scale_y=thermal_scale_y,
        )

    cov_x = max(0.05, min(1.0, float(thermal_coverage_x)))
    cov_y = max(0.05, min(1.0, float(thermal_coverage_y)))

    crop_left = (1.0 - cov_x) * 0.5
    crop_top = (1.0 - cov_y) * 0.5

    def _map_x(px: float) -> float:
        norm = px / float(rgb_w)
        return ((norm - crop_left) / cov_x) * float(th_w)

    def _map_y(py: float) -> float:
        norm = py / float(rgb_h)
        return ((norm - crop_top) / cov_y) * float(th_h)

    mapped_box = RoiBox(
        x1=int(_map_x(float(rgb_roi.x1))),
        y1=int(_map_y(float(rgb_roi.y1))),
        x2=int(_map_x(float(rgb_roi.x2))),
        y2=int(_map_y(float(rgb_roi.y2))),
    )
    return _apply_thermal_roi_adjustment(
        mapped_box=mapped_box,
        width=th_w,
        height=th_h,
        shift_x_ratio=shift_x_ratio,
        shift_y_ratio=shift_y_ratio,
        scale_x=thermal_scale_x,
        scale_y=thermal_scale_y,
    )


def _apply_thermal_roi_adjustment(
    mapped_box: RoiBox,
    width: int,
    height: int,
    shift_x_ratio: float,
    shift_y_ratio: float,
    scale_x: float,
    scale_y: float,
) -> Optional[RoiBox]:
    clipped = _clip_box(mapped_box, width=width, height=height)
    if clipped is None:
        return None

    cx = (clipped.x1 + clipped.x2) * 0.5 + (shift_x_ratio * width)
    cy = (clipped.y1 + clipped.y2) * 0.5 + (shift_y_ratio * height)
    bw = max(2.0, (clipped.x2 - clipped.x1) * max(0.2, scale_x))
    bh = max(2.0, (clipped.y2 - clipped.y1) * max(0.2, scale_y))

    adjusted = RoiBox(
        x1=int(round(cx - 0.5 * bw)),
        y1=int(round(cy - 0.5 * bh)),
        x2=int(round(cx + 0.5 * bw)),
        y2=int(round(cy + 0.5 * bh)),
    )
    return _clip_box(adjusted, width=width, height=height)


def _box_iou(a: RoiBox, b: RoiBox) -> float:
    inter_x1 = max(a.x1, b.x1)
    inter_y1 = max(a.y1, b.y1)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = float(iw * ih)

    area_a = float(max(0, a.x2 - a.x1) * max(0, a.y2 - a.y1))
    area_b = float(max(0, b.x2 - b.x1) * max(0, b.y2 - b.y1))
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _reading_region_box(reading: TempReading, thermal_shape: tuple[int, int, int]) -> Optional[RoiBox]:
    if not reading.region:
        return None
    th_h, th_w = thermal_shape[:2]
    xs = [point.x for point in reading.region]
    ys = [point.y for point in reading.region]
    if all(0.0 <= x <= 1.0 for x in xs) and all(0.0 <= y <= 1.0 for y in ys):
        xs = [x * (th_w - 1) for x in xs]
        ys = [y * (th_h - 1) for y in ys]
    region_box = RoiBox(
        x1=int(np.floor(min(xs))),
        y1=int(np.floor(min(ys))),
        x2=int(np.ceil(max(xs))),
        y2=int(np.ceil(max(ys))),
    )
    return _clip_box(region_box, width=th_w, height=th_h)


def _dedupe_readings(readings: list[TempReading]) -> list[TempReading]:
    seen: set[tuple[str, str, str, float]] = set()
    unique: list[TempReading] = []
    for item in readings:
        key = (
            item.timestamp,
            item.rule_id or "",
            item.temp_property.lower(),
            round(item.temp_value, 3),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


@dataclass(frozen=True)
class TemperatureMatch:
    reading: Optional[TempReading]
    source: str
    matched_rule_id: Optional[str]
    best_iou: float
    best_distance_norm: float
    region_area_px: float
    min_iou_threshold: float
    strict_roi_only: bool
    passed_iou: bool


def _select_temperature_from_p2p_roi(
    pixel_frame: Optional[ThermalPixelFrame],
    thermal_roi: RoiBox,
    thermal_shape: tuple[int, int, int],
    preferred_property: str,
    temp_min_c: float,
    temp_max_c: float,
    trim_low_pct: float,
    trim_high_pct: float,
    min_valid_pixels: int,
) -> TemperatureMatch:
    if pixel_frame is None:
        return TemperatureMatch(
            reading=None,
            source="no_p2p_frame",
            matched_rule_id=None,
            best_iou=0.0,
            best_distance_norm=0.0,
            region_area_px=0.0,
            min_iou_threshold=0.0,
            strict_roi_only=False,
            passed_iou=False,
        )

    th_h, th_w = thermal_shape[:2]
    px_h, px_w = pixel_frame.temp_c.shape[:2]
    if th_w <= 1 or th_h <= 1 or px_w <= 1 or px_h <= 1:
        return TemperatureMatch(
            reading=None,
            source="invalid_p2p_shape",
            matched_rule_id=None,
            best_iou=0.0,
            best_distance_norm=0.0,
            region_area_px=0.0,
            min_iou_threshold=0.0,
            strict_roi_only=False,
            passed_iou=False,
        )

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
    px_roi = _clip_box(px_roi, width=px_w, height=px_h)
    if px_roi is None:
        return TemperatureMatch(
            reading=None,
            source="p2p_roi_oob",
            matched_rule_id=None,
            best_iou=0.0,
            best_distance_norm=0.0,
            region_area_px=0.0,
            min_iou_threshold=0.0,
            strict_roi_only=False,
            passed_iou=False,
        )

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
        return TemperatureMatch(
            reading=None,
            source="no_p2p_pixels",
            matched_rule_id=None,
            best_iou=0.0,
            best_distance_norm=0.0,
            region_area_px=float(max(0, px_roi.x2 - px_roi.x1) * max(0, px_roi.y2 - px_roi.y1)),
            min_iou_threshold=0.0,
            strict_roi_only=False,
            passed_iou=False,
        )
    if finite.size < min_valid_pixels:
        return TemperatureMatch(
            reading=None,
            source="insufficient_p2p_pixels",
            matched_rule_id=None,
            best_iou=0.0,
            best_distance_norm=0.0,
            region_area_px=float(max(0, px_roi.x2 - px_roi.x1) * max(0, px_roi.y2 - px_roi.y1)),
            min_iou_threshold=0.0,
            strict_roi_only=False,
            passed_iou=False,
        )

    preferred = preferred_property.lower()
    if preferred == "highest":
        value = float(np.max(finite))
    elif preferred == "lowest":
        value = float(np.min(finite))
    else:
        value = float(np.mean(finite))
        preferred = "average"

    reading = TempReading(
        timestamp=pixel_frame.timestamp,
        sub_type="thermometry",
        rule_id="p2p_roi",
        temp_value=value,
        temp_unit="centigrade",
        temp_property=preferred,
        region=[],
    )
    return TemperatureMatch(
        reading=reading,
        source="p2p_roi",
        matched_rule_id="p2p_roi",
        best_iou=1.0,
        best_distance_norm=0.0,
        region_area_px=float(max(0, px_roi.x2 - px_roi.x1) * max(0, px_roi.y2 - px_roi.y1)),
        min_iou_threshold=0.0,
        strict_roi_only=False,
        passed_iou=True,
    )


def _select_temperature_for_roi(
    readings: list[TempReading],
    thermal_roi: RoiBox,
    thermal_shape: tuple[int, int, int],
    preferred_property: str,
    min_iou_threshold: float = 0.0,
    strict_roi_only: bool = False,
) -> TemperatureMatch:
    unique = _dedupe_readings(readings)
    if not unique:
        return TemperatureMatch(
            reading=None,
            source="no_candidates",
            matched_rule_id=None,
            best_iou=0.0,
            best_distance_norm=0.0,
            region_area_px=0.0,
            min_iou_threshold=min_iou_threshold,
            strict_roi_only=strict_roi_only,
            passed_iou=False,
        )

    preferred = [item for item in unique if item.temp_property.lower() == preferred_property]
    candidates = preferred or unique
    with_region = [item for item in candidates if item.region]

    if with_region:
        roi_cx = (thermal_roi.x1 + thermal_roi.x2) / 2.0
        roi_cy = (thermal_roi.y1 + thermal_roi.y2) / 2.0
        th_h, th_w = thermal_shape[:2]
        best: Optional[TempReading] = None
        best_region: Optional[RoiBox] = None
        best_score = -1e9
        best_iou = 0.0
        best_dist = 0.0
        for item in with_region:
            region_box = _reading_region_box(item, thermal_shape)
            if region_box is None:
                continue
            iou = _box_iou(thermal_roi, region_box)
            reg_cx = (region_box.x1 + region_box.x2) / 2.0
            reg_cy = (region_box.y1 + region_box.y2) / 2.0
            norm_dx = (roi_cx - reg_cx) / max(1.0, float(th_w))
            norm_dy = (roi_cy - reg_cy) / max(1.0, float(th_h))
            dist = float(np.sqrt(norm_dx * norm_dx + norm_dy * norm_dy))
            score = (2.0 * iou) - dist
            if score > best_score:
                best_score = score
                best = item
                best_region = region_box
                best_iou = iou
                best_dist = dist
        if best is not None:
            passed_iou = best_iou >= min_iou_threshold
            region_area_px = 0.0
            if best_region is not None:
                region_area_px = float(max(0, best_region.x2 - best_region.x1) * max(0, best_region.y2 - best_region.y1))
            if strict_roi_only and not passed_iou:
                return TemperatureMatch(
                    reading=None,
                    source="rejected_iou",
                    matched_rule_id=best.rule_id,
                    best_iou=best_iou,
                    best_distance_norm=best_dist,
                    region_area_px=region_area_px,
                    min_iou_threshold=min_iou_threshold,
                    strict_roi_only=strict_roi_only,
                    passed_iou=False,
                )
            return TemperatureMatch(
                reading=best,
                source="region_best",
                matched_rule_id=best.rule_id,
                best_iou=best_iou,
                best_distance_norm=best_dist,
                region_area_px=region_area_px,
                min_iou_threshold=min_iou_threshold,
                strict_roi_only=strict_roi_only,
                passed_iou=passed_iou,
            )

    if strict_roi_only:
        return TemperatureMatch(
            reading=None,
            source="no_region_strict",
            matched_rule_id=None,
            best_iou=0.0,
            best_distance_norm=0.0,
            region_area_px=0.0,
            min_iou_threshold=min_iou_threshold,
            strict_roi_only=strict_roi_only,
            passed_iou=False,
        )
    fallback = candidates[0] if candidates else None
    return TemperatureMatch(
        reading=fallback,
        source="fallback_no_region",
        matched_rule_id=fallback.rule_id if fallback is not None else None,
        best_iou=0.0,
        best_distance_norm=0.0,
        region_area_px=0.0,
        min_iou_threshold=min_iou_threshold,
        strict_roi_only=strict_roi_only,
        passed_iou=False,
    )


@dataclass(frozen=True)
class AggregatedStats:
    count: int
    mean: float
    median: float
    std: float
    min: float
    max: float
    ewma: float


class RollingTemperatureAggregator:
    def __init__(self, window_sec: float, median_window: int = 5, ewma_alpha: float = 0.2) -> None:
        self.window_sec = max(1.0, float(window_sec))
        self.median_window = max(1, int(median_window))
        self.ewma_alpha = min(1.0, max(0.01, float(ewma_alpha)))

        self._values: deque[tuple[float, float]] = deque()
        self._recent: deque[float] = deque(maxlen=self.median_window)
        self._ewma: Optional[float] = None

    def add(self, timestamp_sec: float, value: float) -> None:
        self._recent.append(float(value))
        smoothed_value = float(statistics.median(self._recent))
        self._values.append((timestamp_sec, smoothed_value))
        self._trim(timestamp_sec)

        if self._ewma is None:
            self._ewma = smoothed_value
        else:
            self._ewma = (self.ewma_alpha * smoothed_value) + ((1.0 - self.ewma_alpha) * self._ewma)

    def get_stats(self, now_sec: float) -> Optional[AggregatedStats]:
        self._trim(now_sec)
        if not self._values:
            return None

        values = [value for _, value in self._values]
        mean_value = sum(values) / len(values)
        median_value = float(statistics.median(values))
        std_value = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
        min_value = min(values)
        max_value = max(values)
        ewma_value = self._ewma if self._ewma is not None else mean_value

        return AggregatedStats(
            count=len(values),
            mean=mean_value,
            median=median_value,
            std=std_value,
            min=min_value,
            max=max_value,
            ewma=float(ewma_value),
        )

    def _trim(self, now_sec: float) -> None:
        cutoff = now_sec - self.window_sec
        while self._values and self._values[0][0] < cutoff:
            self._values.popleft()


class RawCsvSink:
    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        is_new_file = (not self.csv_path.exists()) or self.csv_path.stat().st_size == 0
        self._fp = self.csv_path.open("a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fp)
        if is_new_file:
            self._writer.writerow(
                [
                    "recorded_at_utc",
                    "status",
                    "match_source",
                    "matched_rule_id",
                    "match_iou",
                    "match_distance_norm",
                    "match_region_area_px",
                    "match_min_iou_threshold",
                    "strict_roi_only",
                    "temp_timestamp",
                    "temp_value",
                    "temp_unit",
                    "temp_property",
                    "rule_id",
                    "face_confidence",
                    "rgb_forehead_x1",
                    "rgb_forehead_y1",
                    "rgb_forehead_x2",
                    "rgb_forehead_y2",
                    "thermal_forehead_x1",
                    "thermal_forehead_y1",
                    "thermal_forehead_x2",
                    "thermal_forehead_y2",
                    "metadata_mode",
                ]
            )
            self._fp.flush()

    def append(
        self,
        status: str,
        match: Optional[TemperatureMatch],
        temp: Optional[TempReading],
        face_confidence: Optional[float],
        rgb_forehead: Optional[RoiBox],
        thermal_forehead: Optional[RoiBox],
        metadata_mode: str,
    ) -> None:
        self._writer.writerow(
            [
                datetime.now(timezone.utc).isoformat(),
                status,
                "" if match is None else match.source,
                "" if match is None else (match.matched_rule_id or ""),
                "" if match is None else f"{match.best_iou:.4f}",
                "" if match is None else f"{match.best_distance_norm:.4f}",
                "" if match is None else f"{match.region_area_px:.1f}",
                "" if match is None else f"{match.min_iou_threshold:.4f}",
                "" if match is None else str(match.strict_roi_only).lower(),
                "" if temp is None else temp.timestamp,
                "" if temp is None else f"{temp.temp_value:.3f}",
                "" if temp is None else temp.temp_unit,
                "" if temp is None else temp.temp_property,
                "" if temp is None else (temp.rule_id or ""),
                "" if face_confidence is None else f"{face_confidence:.4f}",
                "" if rgb_forehead is None else rgb_forehead.x1,
                "" if rgb_forehead is None else rgb_forehead.y1,
                "" if rgb_forehead is None else rgb_forehead.x2,
                "" if rgb_forehead is None else rgb_forehead.y2,
                "" if thermal_forehead is None else thermal_forehead.x1,
                "" if thermal_forehead is None else thermal_forehead.y1,
                "" if thermal_forehead is None else thermal_forehead.x2,
                "" if thermal_forehead is None else thermal_forehead.y2,
                metadata_mode,
            ]
        )
        self._fp.flush()

    def close(self) -> None:
        self._fp.close()


class AggregatedCsvSink:
    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        is_new_file = (not self.csv_path.exists()) or self.csv_path.stat().st_size == 0
        self._fp = self.csv_path.open("a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fp)
        if is_new_file:
            self._writer.writerow(
                [
                    "window_end_utc",
                    "window_sec",
                    "samples_total",
                    "samples_valid",
                    "mean_temp",
                    "median_temp",
                    "std_temp",
                    "min_temp",
                    "max_temp",
                    "ewma_temp",
                    "preferred_property",
                    "metadata_mode",
                ]
            )
            self._fp.flush()

    def append(
        self,
        window_sec: float,
        samples_total: int,
        samples_valid: int,
        stats: Optional[AggregatedStats],
        preferred_property: str,
        metadata_mode: str,
    ) -> None:
        self._writer.writerow(
            [
                datetime.now(timezone.utc).isoformat(),
                f"{window_sec:.2f}",
                samples_total,
                samples_valid,
                "" if stats is None else f"{stats.mean:.3f}",
                "" if stats is None else f"{stats.median:.3f}",
                "" if stats is None else f"{stats.std:.3f}",
                "" if stats is None else f"{stats.min:.3f}",
                "" if stats is None else f"{stats.max:.3f}",
                "" if stats is None else f"{stats.ewma:.3f}",
                preferred_property,
                metadata_mode,
            ]
        )
        self._fp.flush()

    def close(self) -> None:
        self._fp.close()


class HikThermometryRegionSync:
    """
    Dynamically writes a rectangular thermometry region to ISAPI endpoint:
    /ISAPI/Thermal/channels/<channel>/thermometry/<scene_id>
    """

    def __init__(
        self,
        ip: str,
        user: str,
        password: str,
        channel_id: int,
        scene_id: int = 1,
        region_id: int = 1,
        enabled: bool = True,
        min_update_period_sec: float = 1.0,
        min_move_norm: float = 0.01,
        timeout_sec: float = 4.0,
        max_consecutive_failures: int = 3,
        failure_backoff_sec: float = 10.0,
    ) -> None:
        self.enabled = bool(enabled)
        self.ip = ip
        self.channel_id = int(channel_id)
        self.scene_id = int(scene_id)
        self.region_id = int(region_id)
        self.min_update_period_sec = max(0.2, float(min_update_period_sec))
        self.min_move_norm = max(0.0, float(min_move_norm))
        self.timeout_sec = max(1.0, float(timeout_sec))
        self.max_consecutive_failures = max(1, int(max_consecutive_failures))
        self.failure_backoff_sec = max(1.0, float(failure_backoff_sec))

        self._session = requests.Session()
        # Ignore system proxy settings (e.g. stale Fiddler proxy).
        self._session.trust_env = False
        self._session.auth = HTTPDigestAuth(user, password)
        self._endpoint = (
            f"http://{self.ip}/ISAPI/Thermal/channels/{self.channel_id}/thermometry/{self.scene_id}"
        )
        self._xml_root: Optional[etree._Element] = None
        self._norm_w = 1000
        self._norm_h = 1000
        self._next_allowed_update_ts = 0.0
        self._last_points: Optional[list[tuple[int, int]]] = None
        self._had_success = False
        self._disabled_by_error = False
        self._consecutive_failures = 0

    def close(self) -> None:
        self._session.close()

    def sync_roi_if_needed(self, roi: RoiBox, thermal_shape: tuple[int, int, int], now_sec: float) -> None:
        if not self.enabled or self._disabled_by_error:
            return
        if now_sec < self._next_allowed_update_ts:
            return
        if not self._ensure_scene_loaded():
            return

        th_h, th_w = thermal_shape[:2]
        if th_w <= 1 or th_h <= 1:
            return

        points = self._roi_to_normalized_polygon(roi=roi, thermal_w=th_w, thermal_h=th_h)
        if not points:
            return

        if self._last_points is not None and not self._is_significant_change(points):
            return

        if not self._apply_region_points(points):
            return
        if not self._put_scene():
            self._next_allowed_update_ts = now_sec + self.failure_backoff_sec
            return

        self._last_points = points
        self._next_allowed_update_ts = now_sec + self.min_update_period_sec
        self._consecutive_failures = 0
        if not self._had_success:
            self._had_success = True
            logging.info(
                "Dynamic thermometry region sync enabled: channel=%d scene=%d region=%d endpoint=%s",
                self.channel_id,
                self.scene_id,
                self.region_id,
                self._endpoint,
            )

    def _ensure_scene_loaded(self) -> bool:
        if self._xml_root is not None:
            return True
        try:
            response = self._session.get(
                self._endpoint,
                timeout=self.timeout_sec,
                headers={"Accept": "application/xml, text/xml, */*"},
            )
            response.raise_for_status()
            root = etree.fromstring(response.content)
            root_name = etree.QName(root).localname
            if root_name != "ThermometryScene":
                logging.warning(
                    "Unexpected thermometry scene root '%s' at %s (check scene/channel).",
                    root_name,
                    self._endpoint,
                )
                return False
            self._xml_root = root
            self._norm_w, self._norm_h = self._read_normalized_screen_size(root)
            return True
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            logging.warning("Thermometry scene GET failed (status=%s): %s", status, self._endpoint)
            if status in (401, 403):
                # Avoid lockout loops on auth/permission errors.
                self._disabled_by_error = True
            return False
        except Exception as exc:  # noqa: BLE001
            logging.warning("Thermometry scene GET parse/network error: %s", exc)
            return False

    def _read_normalized_screen_size(self, root: etree._Element) -> tuple[int, int]:
        norm = self._find_first_by_local_name(root, "normalizedScreenSize")
        if norm is None:
            return 1000, 1000
        w_text = self._find_text_by_local_name(norm, "normalizedScreenWidth") or "1000"
        h_text = self._find_text_by_local_name(norm, "normalizedScreenHeight") or "1000"
        try:
            w = max(10, int(float(w_text)))
            h = max(10, int(float(h_text)))
        except ValueError:
            return 1000, 1000
        return w, h

    def _roi_to_normalized_polygon(
        self,
        roi: RoiBox,
        thermal_w: int,
        thermal_h: int,
    ) -> list[tuple[int, int]]:
        def norm_x(x: int) -> int:
            return int(round((max(0, min(thermal_w - 1, x)) / float(thermal_w - 1)) * self._norm_w))

        def norm_y(y: int) -> int:
            return int(round((max(0, min(thermal_h - 1, y)) / float(thermal_h - 1)) * self._norm_h))

        x1 = norm_x(roi.x1)
        y1 = norm_y(roi.y1)
        x2 = norm_x(roi.x2)
        y2 = norm_y(roi.y2)
        if x2 <= x1 or y2 <= y1:
            return []
        return [
            (x1, y1),  # top-left
            (x2, y1),  # top-right
            (x2, y2),  # bottom-right
            (x1, y2),  # bottom-left
        ]

    def _is_significant_change(self, new_points: list[tuple[int, int]]) -> bool:
        assert self._last_points is not None
        scale = float(max(self._norm_w, self._norm_h))
        max_delta = 0.0
        for (ax, ay), (bx, by) in zip(self._last_points, new_points):
            max_delta = max(max_delta, abs(ax - bx), abs(ay - by))
        return (max_delta / scale) >= self.min_move_norm

    def _apply_region_points(self, points: list[tuple[int, int]]) -> bool:
        assert self._xml_root is not None
        region_list = self._find_first_by_local_name(self._xml_root, "ThermometryRegionList")
        if region_list is None:
            logging.warning("Thermometry scene XML has no ThermometryRegionList.")
            self._disabled_by_error = True
            return False

        target_region: Optional[etree._Element] = None
        for region in self._find_children_by_local_name(region_list, "ThermometryRegion"):
            region_id = (self._find_text_by_local_name(region, "id") or "").strip()
            if region_id == str(self.region_id):
                target_region = region
                break
        if target_region is None:
            logging.warning("Thermometry region id=%d not found in scene XML.", self.region_id)
            self._disabled_by_error = True
            return False

        self._upsert_text(target_region, "enabled", "true")
        self._upsert_text(target_region, "type", "region")

        region_node = self._find_first_by_local_name(target_region, "Region")
        if region_node is None:
            region_node = etree.SubElement(target_region, self._ns_tag(target_region, "Region"))
        coords_list = self._find_first_by_local_name(region_node, "RegionCoordinatesList")
        if coords_list is None:
            coords_list = etree.SubElement(region_node, self._ns_tag(region_node, "RegionCoordinatesList"))

        for child in list(coords_list):
            coords_list.remove(child)

        for px, py in points:
            rc = etree.SubElement(coords_list, self._ns_tag(coords_list, "RegionCoordinates"))
            etree.SubElement(rc, self._ns_tag(rc, "positionX")).text = str(px)
            etree.SubElement(rc, self._ns_tag(rc, "positionY")).text = str(py)
        return True

    def _upsert_text(self, parent: etree._Element, tag: str, text: str) -> None:
        node = self._find_first_by_local_name(parent, tag)
        if node is None:
            node = etree.SubElement(parent, self._ns_tag(parent, tag))
        node.text = text

    def _find_first_by_local_name(self, parent: etree._Element, local_name: str) -> Optional[etree._Element]:
        if etree.QName(parent).localname == local_name:
            return parent
        matches = parent.xpath(f".//*[local-name()='{local_name}']")
        if not matches:
            return None
        first = matches[0]
        if isinstance(first, etree._Element):
            return first
        return None

    def _find_children_by_local_name(self, parent: etree._Element, local_name: str) -> list[etree._Element]:
        matches = parent.xpath(f"./*[local-name()='{local_name}']")
        out: list[etree._Element] = []
        for item in matches:
            if isinstance(item, etree._Element):
                out.append(item)
        return out

    def _find_text_by_local_name(self, parent: etree._Element, local_name: str) -> Optional[str]:
        node = self._find_first_by_local_name(parent, local_name)
        if node is None:
            return None
        return node.text

    def _ns_tag(self, parent: etree._Element, local_name: str) -> str:
        namespace = etree.QName(parent).namespace
        if namespace:
            return f"{{{namespace}}}{local_name}"
        return local_name

    def _put_scene(self) -> bool:
        assert self._xml_root is not None
        payload = etree.tostring(self._xml_root, encoding="utf-8", xml_declaration=True)
        content_types = [
            "application/xml; charset=UTF-8",
            "application/x-www-form-urlencoded; charset=UTF-8",
        ]
        last_status: Optional[int] = None

        for content_type in content_types:
            try:
                response = self._session.put(
                    self._endpoint,
                    data=payload,
                    headers={"Content-Type": content_type},
                    timeout=self.timeout_sec,
                )
                response.raise_for_status()
                return True
            except requests.HTTPError as exc:
                last_status = exc.response.status_code if exc.response is not None else None
                if last_status in (401, 403):
                    logging.warning("Thermometry scene PUT auth failed (status=%s): %s", last_status, self._endpoint)
                    self._disabled_by_error = True
                    return False
            except Exception as exc:  # noqa: BLE001
                self._register_failure(f"Thermometry scene PUT network error: {exc}")
                return False

        self._register_failure(f"Thermometry scene PUT failed (status={last_status}): {self._endpoint}")
        return False

    def _register_failure(self, message: str) -> None:
        self._consecutive_failures += 1
        logging.warning("%s", message)
        if self._consecutive_failures >= self.max_consecutive_failures:
            self._disabled_by_error = True
            logging.warning(
                "Dynamic thermometry sync auto-disabled after %d consecutive failures. "
                "Set DYNAMIC_THERMOMETRY_REGION_ENABLED=false or tune update/backoff settings.",
                self._consecutive_failures,
            )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    cfg = load_settings()
    min_detection_conf = _parse_float_env("MP_MIN_DETECTION_CONF", 0.60)
    preferred_property = (os.getenv("TEMP_PROPERTY") or "average").strip().lower() or "average"
    raw_csv_path = Path(os.getenv("RAW_CSV_PATH") or os.getenv("FOREHEAD_CSV_PATH") or "logs/forehead_raw.csv")
    agg_csv_path = Path(os.getenv("AGG_CSV_PATH") or "logs/forehead_aggregated.csv")
    mp_face_model_path = (os.getenv("MP_FACE_MODEL_PATH") or "").strip()
    mp_model_selection = max(0, min(1, _parse_int_env("MP_MODEL_SELECTION", 1)))
    mp_detection_input_scale = max(1.0, _parse_float_env("MP_DETECTION_INPUT_SCALE", 1.5))
    mp_face_top_expand_ratio = _parse_float_env("MP_FACE_TOP_EXPAND_RATIO", 0.35)
    sample_period_sec = max(0.2, _parse_float_env("SAMPLE_PERIOD_SEC", 1.0))
    agg_window_sec = max(sample_period_sec, _parse_float_env("AGG_WINDOW_SEC", 60.0))
    agg_emit_sec = max(sample_period_sec, _parse_float_env("AGG_EMIT_SEC", 5.0))
    median_window = max(1, _parse_int_env("MEDIAN_WINDOW", 5))
    ewma_alpha = _parse_float_env("EWMA_ALPHA", 0.2)
    roi_match_min_iou = max(0.0, min(1.0, _parse_float_env("ROI_MATCH_MIN_IOU", 0.15)))
    strict_roi_only = _parse_bool_env("STRICT_ROI_ONLY", True)
    thermal_coverage_x = _parse_float_env("THERMAL_COVERAGE_X", 1.0)
    thermal_coverage_y = _parse_float_env("THERMAL_COVERAGE_Y", 1.0)
    thermal_shift_x_ratio = _parse_float_env("THERMAL_ROI_SHIFT_X_RATIO", 0.0)
    thermal_shift_y_ratio = _parse_float_env("THERMAL_ROI_SHIFT_Y_RATIO", 0.0)
    thermal_scale_x = _parse_float_env("THERMAL_ROI_SCALE_X", 1.0)
    thermal_scale_y = _parse_float_env("THERMAL_ROI_SCALE_Y", 1.0)
    homography_switch_enabled = _parse_bool_env("HOMOGRAPHY_SWITCH_ENABLED", False)
    homography_near = _load_homography_from_env_var("RGB_TO_THERMAL_H_NEAR")
    homography_far = _load_homography_from_env_var("RGB_TO_THERMAL_H_FAR")
    homography_near_min_ratio = _parse_float_env("HOMOGRAPHY_NEAR_MIN_FACE_WIDTH_RATIO", 0.32)
    homography_far_max_ratio = _parse_float_env("HOMOGRAPHY_FAR_MAX_FACE_WIDTH_RATIO", 0.24)
    p2p_temp_min_c = _parse_float_env("P2P_TEMP_MIN_C", 32.0)
    p2p_temp_max_c = _parse_float_env("P2P_TEMP_MAX_C", 43.5)
    p2p_trim_low_pct = _parse_float_env("P2P_TRIM_LOW_PCT", 5.0)
    p2p_trim_high_pct = _parse_float_env("P2P_TRIM_HIGH_PCT", 5.0)
    p2p_min_valid_pixels = max(1, _parse_int_env("P2P_MIN_VALID_PIXELS", 25))
    dynamic_region_enabled = _parse_bool_env("DYNAMIC_THERMOMETRY_REGION_ENABLED", False)
    dynamic_region_scene_id = _parse_int_env("DYNAMIC_THERMOMETRY_SCENE_ID", 1)
    dynamic_region_id = _parse_int_env("DYNAMIC_THERMOMETRY_REGION_ID", 1)
    dynamic_region_update_sec = max(0.2, _parse_float_env("DYNAMIC_THERMOMETRY_UPDATE_SEC", 1.0))
    dynamic_region_min_move_norm = max(0.0, _parse_float_env("DYNAMIC_THERMOMETRY_MIN_MOVE_NORM", 0.01))
    dynamic_region_max_failures = max(1, _parse_int_env("DYNAMIC_THERMOMETRY_MAX_FAILURES", 1))
    dynamic_region_failure_backoff_sec = max(1.0, _parse_float_env("DYNAMIC_THERMOMETRY_FAILURE_BACKOFF_SEC", 10.0))
    homography = _load_homography_from_env()
    # Runtime manual tuning (keyboard) starts from .env values.
    current_shift_x_ratio = thermal_shift_x_ratio
    current_shift_y_ratio = thermal_shift_y_ratio
    current_scale_x = thermal_scale_x
    current_scale_y = thermal_scale_y
    tune_step_shift = 0.005
    tune_step_scale = 0.02

    if homography is None:
        logging.info("ROI mapping mode: scale RGB->thermal (set RGB_TO_THERMAL_H for calibrated homography).")
    else:
        logging.info("ROI mapping mode: calibrated homography from RGB_TO_THERMAL_H.")
    if homography_switch_enabled:
        logging.info(
            "Homography switch: enabled near=%s far=%s near_min_ratio=%.3f far_max_ratio=%.3f",
            "yes" if homography_near is not None else "no",
            "yes" if homography_far is not None else "no",
            homography_near_min_ratio,
            homography_far_max_ratio,
        )
        if homography_far_max_ratio >= homography_near_min_ratio:
            logging.warning(
                "HOMOGRAPHY_FAR_MAX_FACE_WIDTH_RATIO (%.3f) should be lower than "
                "HOMOGRAPHY_NEAR_MIN_FACE_WIDTH_RATIO (%.3f) for stable hysteresis.",
                homography_far_max_ratio,
                homography_near_min_ratio,
            )
    if homography is None and (abs(thermal_coverage_x - 1.0) > 1e-9 or abs(thermal_coverage_y - 1.0) > 1e-9):
        logging.info(
            "Thermal FOV coverage enabled: coverage=(%.4f, %.4f) of RGB frame (central crop mapping).",
            thermal_coverage_x,
            thermal_coverage_y,
        )
    if (
        abs(thermal_shift_x_ratio) > 1e-9
        or abs(thermal_shift_y_ratio) > 1e-9
        or abs(thermal_scale_x - 1.0) > 1e-9
        or abs(thermal_scale_y - 1.0) > 1e-9
    ):
        logging.info(
            "Thermal ROI correction enabled: shift=(%.4f, %.4f), scale=(%.4f, %.4f)",
            thermal_shift_x_ratio,
            thermal_shift_y_ratio,
            thermal_scale_x,
            thermal_scale_y,
        )
    logging.info(
        "Workflow CSV: raw=%s, agg=%s | sample=%.2fs, agg_window=%.2fs, agg_emit=%.2fs, median_window=%d, ewma_alpha=%.3f",
        raw_csv_path,
        agg_csv_path,
        sample_period_sec,
        agg_window_sec,
        agg_emit_sec,
        median_window,
        ewma_alpha,
    )
    logging.info(
        "ROI match policy: strict=%s min_iou=%.3f",
        str(strict_roi_only).lower(),
        roi_match_min_iou,
    )
    if cfg.metadata_mode == "http_thermal_p2p":
        logging.info(
            "P2P ROI filter: temp_range=[%.2f, %.2f]C trim_low=%.1f%% trim_high=%.1f%% min_valid_pixels=%d",
            p2p_temp_min_c,
            p2p_temp_max_c,
            p2p_trim_low_pct,
            p2p_trim_high_pct,
            p2p_min_valid_pixels,
        )
    logging.info(
        "Dynamic thermometry region sync: enabled=%s scene_id=%d region_id=%d update_sec=%.2f min_move_norm=%.4f max_failures=%d failure_backoff=%.1fs",
        str(dynamic_region_enabled).lower(),
        dynamic_region_scene_id,
        dynamic_region_id,
        dynamic_region_update_sec,
        dynamic_region_min_move_norm,
        dynamic_region_max_failures,
        dynamic_region_failure_backoff_sec,
    )
    logging.info(
        "Manual thermal ROI tuning keys: WASD=shift, I/K=scaleY +/- , J/L=scaleX +/- , "
        "[/] shift step -, + ;  -/= scale step -, + ; R=reset tuning ; P=print .env snippet"
    )

    rgb_stream = RtspStream(name="RGB", rtsp_url=cfg.rtsp_rgb, reconnect_delay_sec=cfg.reconnect_delay_sec)
    th_stream = RtspStream(name="THERMAL", rtsp_url=cfg.rtsp_th, reconnect_delay_sec=cfg.reconnect_delay_sec)

    if not cfg.enable_metadata:
        raise ValueError(
            "ENABLE_METADATA=false in .env. "
            "Set ENABLE_METADATA=true before running src.mp_forehead_capture."
        )

    metadata = HikMetadataClient(
        ip=cfg.hik_ip,
        user=cfg.hik_user,
        password=cfg.hik_pass,
        channel_id=cfg.channel_id_for_metadata,
        rtsp_port=cfg.rtsp_port,
        reconnect_delay_sec=cfg.metadata_retry_sec,
        mode=cfg.metadata_mode,
        forced_legacy_uri=cfg.metadata_legacy_uri,
        forced_http_endpoint=cfg.metadata_http_endpoint,
        auth_lockout_sleep_sec=cfg.metadata_auth_lockout_sec,
        max_auth_failures=cfg.metadata_max_auth_failures,
    )

    roi_provider = MediaPipeForeheadRoiProvider(
        min_detection_confidence=min_detection_conf,
        model_selection=mp_model_selection,
        model_asset_path=mp_face_model_path,
        face_top_expand_ratio=mp_face_top_expand_ratio,
        detection_input_scale=mp_detection_input_scale,
    )
    raw_csv_sink = RawCsvSink(raw_csv_path)
    agg_csv_sink = AggregatedCsvSink(agg_csv_path)
    rolling_agg = RollingTemperatureAggregator(
        window_sec=agg_window_sec,
        median_window=median_window,
        ewma_alpha=ewma_alpha,
    )
    dynamic_region_sync = HikThermometryRegionSync(
        ip=cfg.hik_ip,
        user=cfg.hik_user,
        password=cfg.hik_pass,
        channel_id=cfg.channel_id_for_metadata,
        scene_id=dynamic_region_scene_id,
        region_id=dynamic_region_id,
        enabled=dynamic_region_enabled,
        min_update_period_sec=dynamic_region_update_sec,
        min_move_norm=dynamic_region_min_move_norm,
        max_consecutive_failures=dynamic_region_max_failures,
        failure_backoff_sec=dynamic_region_failure_backoff_sec,
    )
    sample_history: deque[tuple[float, str]] = deque()

    rgb_stream.start()
    th_stream.start()
    metadata.start()

    display_w = 1280
    display_h = 720
    cv2.namedWindow("RGB Forehead ROI", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Thermal ROI", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RGB Forehead ROI", display_w, display_h)
    cv2.resizeWindow("Thermal ROI", display_w, display_h)

    last_sample_ts = 0.0
    last_agg_emit_ts = 0.0
    overlay_text = "temp: n/a"
    active_h_mode = "default"

    try:
        while True:
            rgb_frame = rgb_stream.get_last_frame()
            th_frame = th_stream.get_last_frame()

            rgb_roi: Optional[RoiBox] = None
            thermal_roi: Optional[RoiBox] = None
            selected_temp: Optional[TempReading] = None
            selected_match: Optional[TemperatureMatch] = None

            if rgb_frame is not None:
                rgb_roi = roi_provider.get_forehead_roi(rgb_frame)
                if roi_provider.last_face_box is not None:
                    face = roi_provider.last_face_box
                    cv2.rectangle(rgb_frame, (face.x1, face.y1), (face.x2, face.y2), (255, 170, 0), 2)
                    label = f"face conf={roi_provider.last_score:.2f}" if roi_provider.last_score is not None else "face"
                    cv2.putText(
                        rgb_frame,
                        label,
                        (face.x1, max(20, face.y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 170, 0),
                        2,
                    )
                if rgb_roi is not None:
                    cv2.rectangle(rgb_frame, (rgb_roi.x1, rgb_roi.y1), (rgb_roi.x2, rgb_roi.y2), (0, 255, 0), 2)
                    cv2.putText(
                        rgb_frame,
                        "forehead ROI",
                        (rgb_roi.x1, max(20, rgb_roi.y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2,
                    )
                rgb_view = cv2.resize(rgb_frame, (display_w, display_h), interpolation=cv2.INTER_AREA)
                cv2.imshow("RGB Forehead ROI", rgb_view)

            if th_frame is not None and rgb_frame is not None and rgb_roi is not None:
                active_h, next_h_mode = _select_active_homography(
                    face_box=roi_provider.last_face_box,
                    rgb_shape=rgb_frame.shape,
                    default_h=homography,
                    near_h=homography_near,
                    far_h=homography_far,
                    switch_enabled=homography_switch_enabled,
                    near_min_face_width_ratio=homography_near_min_ratio,
                    far_max_face_width_ratio=homography_far_max_ratio,
                    last_mode=active_h_mode,
                )
                if next_h_mode != active_h_mode:
                    logging.info("Homography mode switched: %s -> %s", active_h_mode, next_h_mode)
                    active_h_mode = next_h_mode
                thermal_roi = _map_rgb_roi_to_thermal(
                    rgb_roi=rgb_roi,
                    rgb_shape=rgb_frame.shape,
                    thermal_shape=th_frame.shape,
                    homography=active_h,
                    thermal_coverage_x=thermal_coverage_x,
                    thermal_coverage_y=thermal_coverage_y,
                    shift_x_ratio=current_shift_x_ratio,
                    shift_y_ratio=current_shift_y_ratio,
                    thermal_scale_x=current_scale_x,
                    thermal_scale_y=current_scale_y,
                )

                if thermal_roi is not None:
                    dynamic_region_sync.sync_roi_if_needed(
                        roi=thermal_roi,
                        thermal_shape=th_frame.shape,
                        now_sec=time.time(),
                    )

                    if cfg.metadata_mode == "http_thermal_p2p":
                        selected_match = _select_temperature_from_p2p_roi(
                            pixel_frame=metadata.get_latest_pixel_frame(),
                            thermal_roi=thermal_roi,
                            thermal_shape=th_frame.shape,
                            preferred_property=preferred_property,
                            temp_min_c=p2p_temp_min_c,
                            temp_max_c=p2p_temp_max_c,
                            trim_low_pct=p2p_trim_low_pct,
                            trim_high_pct=p2p_trim_high_pct,
                            min_valid_pixels=p2p_min_valid_pixels,
                        )
                    else:
                        readings = metadata.get_latest_temperatures()
                        selected_match = _select_temperature_for_roi(
                            readings=readings,
                            thermal_roi=thermal_roi,
                            thermal_shape=th_frame.shape,
                            preferred_property=preferred_property,
                            min_iou_threshold=roi_match_min_iou,
                            strict_roi_only=strict_roi_only,
                        )
                    selected_temp = selected_match.reading

                    cv2.rectangle(
                        th_frame,
                        (thermal_roi.x1, thermal_roi.y1),
                        (thermal_roi.x2, thermal_roi.y2),
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        th_frame,
                        "mapped forehead ROI",
                        (thermal_roi.x1, max(20, thermal_roi.y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2,
                    )
                if selected_temp is not None:
                    overlay_text = (
                        f"{selected_temp.temp_value:.2f} {selected_temp.temp_unit} "
                        f"({selected_temp.temp_property}) rule={selected_temp.rule_id or '-'} "
                        f"iou={(selected_match.best_iou if selected_match else 0.0):.2f}"
                    )
                else:
                    overlay_text = "temp: n/a"

            if th_frame is not None:
                _draw_hud_text(
                    th_frame,
                    overlay_text,
                    x=20,
                    y=86,
                    color=(0, 255, 255),
                    font_scale=0.75,
                    thickness=2,
                )
                _draw_hud_text(
                    th_frame,
                    (
                        f"tune shift=({current_shift_x_ratio:+.3f},{current_shift_y_ratio:+.3f}) "
                        f"scale=({current_scale_x:.3f},{current_scale_y:.3f}) "
                        f"step_shift={tune_step_shift:.3f} step_scale={tune_step_scale:.3f}"
                    ),
                    x=20,
                    y=114,
                    color=(200, 255, 200),
                    font_scale=0.52,
                    thickness=1,
                )
                th_view = cv2.resize(th_frame, (display_w, display_h), interpolation=cv2.INTER_AREA)
                cv2.imshow("Thermal ROI", th_view)

            now_sec = time.time()
            if (now_sec - last_sample_ts) >= sample_period_sec:
                last_sample_ts = now_sec

                status = "ok"
                if rgb_frame is None:
                    status = "no_rgb_frame"
                elif th_frame is None:
                    status = "no_thermal_frame"
                elif rgb_roi is None:
                    status = "no_face_roi"
                elif thermal_roi is None:
                    status = "no_mapped_thermal_roi"
                elif selected_temp is None:
                    if selected_match is not None and selected_match.source == "rejected_iou":
                        status = "below_iou_threshold"
                    elif selected_match is not None and selected_match.source == "no_region_strict":
                        status = "no_region_for_strict_mode"
                    elif selected_match is not None and selected_match.source == "no_p2p_frame":
                        status = "no_p2p_frame"
                    elif selected_match is not None and selected_match.source == "no_p2p_pixels":
                        status = "no_p2p_pixels"
                    elif selected_match is not None and selected_match.source == "insufficient_p2p_pixels":
                        status = "insufficient_p2p_pixels"
                    else:
                        status = "no_temperature_match"

                if status != "ok":
                    overlay_text = f"temp: n/a ({status})"

                raw_csv_sink.append(
                    status=status,
                    match=selected_match,
                    temp=selected_temp if status == "ok" else None,
                    face_confidence=roi_provider.last_score,
                    rgb_forehead=rgb_roi,
                    thermal_forehead=thermal_roi,
                    metadata_mode=cfg.metadata_mode,
                )

                if status == "ok" and selected_temp is not None:
                    rolling_agg.add(timestamp_sec=now_sec, value=selected_temp.temp_value)

                sample_history.append((now_sec, status))
                cutoff = now_sec - agg_window_sec
                while sample_history and sample_history[0][0] < cutoff:
                    sample_history.popleft()

            if (now_sec - last_agg_emit_ts) >= agg_emit_sec:
                last_agg_emit_ts = now_sec

                stats = rolling_agg.get_stats(now_sec=now_sec)
                samples_total = len(sample_history)
                samples_valid = sum(1 for _, item_status in sample_history if item_status == "ok")

                agg_csv_sink.append(
                    window_sec=agg_window_sec,
                    samples_total=samples_total,
                    samples_valid=samples_valid,
                    stats=stats,
                    preferred_property=preferred_property,
                    metadata_mode=cfg.metadata_mode,
                )

                if stats is None:
                    logging.info("AGG %.0fs: samples=%d valid=%d temp=n/a", agg_window_sec, samples_total, samples_valid)
                else:
                    logging.info(
                        "AGG %.0fs: samples=%d valid=%d mean=%.2f median=%.2f std=%.2f ewma=%.2f",
                        agg_window_sec,
                        samples_total,
                        samples_valid,
                        stats.mean,
                        stats.median,
                        stats.std,
                        stats.ewma,
                    )

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break
            if key == ord("w"):
                current_shift_y_ratio = _clamp(current_shift_y_ratio - tune_step_shift, -0.5, 0.5)
            elif key == ord("s"):
                current_shift_y_ratio = _clamp(current_shift_y_ratio + tune_step_shift, -0.5, 0.5)
            elif key == ord("a"):
                current_shift_x_ratio = _clamp(current_shift_x_ratio - tune_step_shift, -0.5, 0.5)
            elif key == ord("d"):
                current_shift_x_ratio = _clamp(current_shift_x_ratio + tune_step_shift, -0.5, 0.5)
            elif key == ord("j"):
                current_scale_x = _clamp(current_scale_x - tune_step_scale, 0.2, 2.5)
            elif key == ord("l"):
                current_scale_x = _clamp(current_scale_x + tune_step_scale, 0.2, 2.5)
            elif key == ord("k"):
                current_scale_y = _clamp(current_scale_y - tune_step_scale, 0.2, 2.5)
            elif key == ord("i"):
                current_scale_y = _clamp(current_scale_y + tune_step_scale, 0.2, 2.5)
            elif key == ord("["):
                tune_step_shift = _clamp(tune_step_shift * 0.5, 0.001, 0.05)
            elif key == ord("]"):
                tune_step_shift = _clamp(tune_step_shift * 2.0, 0.001, 0.05)
            elif key == ord("-"):
                tune_step_scale = _clamp(tune_step_scale * 0.5, 0.005, 0.20)
            elif key == ord("="):
                tune_step_scale = _clamp(tune_step_scale * 2.0, 0.005, 0.20)
            elif key == ord("r"):
                current_shift_x_ratio = thermal_shift_x_ratio
                current_shift_y_ratio = thermal_shift_y_ratio
                current_scale_x = thermal_scale_x
                current_scale_y = thermal_scale_y
                tune_step_shift = 0.005
                tune_step_scale = 0.02
                logging.info("Manual tuning reset to .env defaults.")
            elif key == ord("p"):
                logging.info("Paste this to .env:")
                logging.info("THERMAL_ROI_SHIFT_X_RATIO=%.4f", current_shift_x_ratio)
                logging.info("THERMAL_ROI_SHIFT_Y_RATIO=%.4f", current_shift_y_ratio)
                logging.info("THERMAL_ROI_SCALE_X=%.4f", current_scale_x)
                logging.info("THERMAL_ROI_SCALE_Y=%.4f", current_scale_y)
            time.sleep(0.001)
    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
    finally:
        for stopper in (metadata.stop, rgb_stream.stop, th_stream.stop):
            try:
                stopper()
            except Exception as exc:  # noqa: BLE001
                logging.debug("shutdown warning: %s", exc)
        try:
            roi_provider.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            dynamic_region_sync.close()
        except Exception:  # noqa: BLE001
            pass
        raw_csv_sink.close()
        agg_csv_sink.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
