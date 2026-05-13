from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.config import load_settings


def _format_h_for_env(h: np.ndarray) -> str:
    flat = h.reshape(-1)
    return ",".join(f"{value:.8f}" for value in flat)


@dataclass
class CalibrationState:
    rgb_points: list[tuple[float, float]] = field(default_factory=list)
    th_points: list[tuple[float, float]] = field(default_factory=list)
    rgb_frame_shape: tuple[int, int] = (0, 0)  # h, w
    th_frame_shape: tuple[int, int] = (0, 0)  # h, w
    rgb_view_shape: tuple[int, int] = (0, 0)  # h, w
    th_view_shape: tuple[int, int] = (0, 0)  # h, w
    homography: np.ndarray | None = None
    reprojection_error_px: float | None = None

    def clear(self) -> None:
        self.rgb_points.clear()
        self.th_points.clear()
        self.homography = None
        self.reprojection_error_px = None

    def paired_count(self) -> int:
        return min(len(self.rgb_points), len(self.th_points))


def _to_frame_coords(
    x_view: int,
    y_view: int,
    view_shape: tuple[int, int],
    frame_shape: tuple[int, int],
) -> tuple[float, float]:
    view_h, view_w = view_shape
    frame_h, frame_w = frame_shape
    if view_w <= 0 or view_h <= 0 or frame_w <= 0 or frame_h <= 0:
        return float(x_view), float(y_view)
    x = (float(x_view) / float(view_w)) * float(frame_w)
    y = (float(y_view) / float(view_h)) * float(frame_h)
    return x, y


def _rgb_callback(state: CalibrationState):
    def _cb(event, x, y, _flags, _userdata):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        point = _to_frame_coords(x, y, state.rgb_view_shape, state.rgb_frame_shape)
        state.rgb_points.append(point)
        state.homography = None
        state.reprojection_error_px = None
        logging.info("RGB point #%d: (%.1f, %.1f)", len(state.rgb_points), point[0], point[1])

    return _cb


def _th_callback(state: CalibrationState):
    def _cb(event, x, y, _flags, _userdata):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        point = _to_frame_coords(x, y, state.th_view_shape, state.th_frame_shape)
        state.th_points.append(point)
        state.homography = None
        state.reprojection_error_px = None
        logging.info("THERMAL point #%d: (%.1f, %.1f)", len(state.th_points), point[0], point[1])

    return _cb


def _draw_points(
    frame: np.ndarray,
    points: list[tuple[float, float]],
    color: tuple[int, int, int],
    label: str,
) -> np.ndarray:
    out = frame.copy()
    for idx, (x, y) in enumerate(points, start=1):
        cx = int(round(x))
        cy = int(round(y))
        cv2.circle(out, (cx, cy), 5, color, -1)
        cv2.putText(
            out,
            f"{label}{idx}",
            (cx + 8, cy - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )
    return out


def _compute_homography(state: CalibrationState) -> None:
    n = state.paired_count()
    if n < 4:
        logging.warning("Need at least 4 point pairs. Now: %d", n)
        return
    rgb = np.array(state.rgb_points[:n], dtype=np.float32).reshape(-1, 1, 2)
    th = np.array(state.th_points[:n], dtype=np.float32).reshape(-1, 1, 2)
    h, mask = cv2.findHomography(rgb, th, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if h is None:
        logging.warning("findHomography failed.")
        return
    projected = cv2.perspectiveTransform(rgb, h)
    errors = np.linalg.norm(projected.reshape(-1, 2) - th.reshape(-1, 2), axis=1)
    inlier_mask = (mask.reshape(-1) > 0) if mask is not None else np.ones((n,), dtype=bool)
    if np.any(inlier_mask):
        err = float(np.mean(errors[inlier_mask]))
    else:
        err = float(np.mean(errors))
    state.homography = h.astype(np.float64)
    state.reprojection_error_px = err
    inliers = int(np.sum(inlier_mask))
    logging.info("Homography computed. Inliers=%d/%d, mean reprojection error=%.2f px", inliers, n, err)
    logging.info("RGB_TO_THERMAL_H=%s", _format_h_for_env(state.homography))


def _save_homography(state: CalibrationState, out_path: Path) -> None:
    if state.homography is None:
        logging.warning("No homography to save. Press 'c' first.")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "rgb_to_thermal_h": state.homography.tolist(),
        "reprojection_error_px": state.reprojection_error_px,
        "point_pairs": state.paired_count(),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logging.info("Saved homography to %s", out_path)
    logging.info("Paste to .env:")
    logging.info("RGB_TO_THERMAL_H=%s", _format_h_for_env(state.homography))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = load_settings()

    cap_rgb = cv2.VideoCapture(cfg.rtsp_rgb)
    cap_th = cv2.VideoCapture(cfg.rtsp_th)
    if not cap_rgb.isOpened():
        raise RuntimeError(f"Cannot open RGB RTSP: {cfg.rtsp_rgb}")
    if not cap_th.isOpened():
        raise RuntimeError(f"Cannot open THERMAL RTSP: {cfg.rtsp_th}")

    cv2.namedWindow("Calib RGB", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Calib THERMAL", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calib RGB", 1280, 720)
    cv2.resizeWindow("Calib THERMAL", 1280, 720)

    state = CalibrationState()
    cv2.setMouseCallback("Calib RGB", _rgb_callback(state))
    cv2.setMouseCallback("Calib THERMAL", _th_callback(state))

    logging.info("Click matching points in BOTH windows in the same order.")
    logging.info("Keys: c=compute homography, r=reset points, s=save, q/ESC=quit")
    out_file = Path("logs/rgb_to_thermal_homography.json")

    try:
        while True:
            ok_rgb, frame_rgb = cap_rgb.read()
            ok_th, frame_th = cap_th.read()
            if not ok_rgb or frame_rgb is None:
                continue
            if not ok_th or frame_th is None:
                continue

            state.rgb_frame_shape = frame_rgb.shape[:2]
            state.th_frame_shape = frame_th.shape[:2]

            draw_rgb = _draw_points(frame_rgb, state.rgb_points, (0, 255, 0), "R")
            draw_th = _draw_points(frame_th, state.th_points, (0, 255, 255), "T")
            pair_count = state.paired_count()
            info = f"pairs={pair_count} rgb={len(state.rgb_points)} th={len(state.th_points)}"
            cv2.putText(draw_rgb, info, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            cv2.putText(draw_th, info, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
            if state.reprojection_error_px is not None:
                cv2.putText(
                    draw_th,
                    f"reproj err: {state.reprojection_error_px:.2f}px",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                )

            rgb_h, rgb_w = draw_rgb.shape[:2]
            th_h, th_w = draw_th.shape[:2]
            state.rgb_view_shape = (720, 1280)
            state.th_view_shape = (720, 1280)
            view_rgb = cv2.resize(draw_rgb, (1280, 720), interpolation=cv2.INTER_AREA)
            view_th = cv2.resize(draw_th, (1280, 720), interpolation=cv2.INTER_AREA)
            if rgb_w <= 0 or rgb_h <= 0:
                state.rgb_view_shape = (0, 0)
            if th_w <= 0 or th_h <= 0:
                state.th_view_shape = (0, 0)

            cv2.imshow("Calib RGB", view_rgb)
            cv2.imshow("Calib THERMAL", view_th)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("r"):
                state.clear()
                logging.info("Points cleared.")
            elif key == ord("c"):
                _compute_homography(state)
            elif key == ord("s"):
                _save_homography(state, out_file)
    finally:
        cap_rgb.release()
        cap_th.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
