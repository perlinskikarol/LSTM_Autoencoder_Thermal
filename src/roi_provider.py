from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

import cv2


@dataclass(frozen=True)
class RoiBox:
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(frozen=True)
class RoiPolygon:
    points: list[tuple[int, int]]


class ForeheadRoiProvider(Protocol):
    def get_forehead_roi(self, rgb_frame) -> Optional[RoiBox | RoiPolygon]:
        """Return forehead ROI detected on RGB frame."""


class MediaPipeForeheadRoiProvider:
    """
    Face detector based on MediaPipe.
    Forehead ROI is estimated as top-center subregion of detected face box.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        model_selection: int = 0,
        model_asset_path: str = "",
        face_top_expand_ratio: float = 0.35,
        detection_input_scale: float = 1.0,
    ) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:  # pragma: no cover - depends on local env
            raise RuntimeError("mediapipe is required. Install with: pip install mediapipe") from exc

        self._backend = ""
        self._face_detection = None
        self._tasks_detector = None
        self._tasks_image_cls = None
        self._tasks_image_format = None
        self._face_top_expand_ratio = max(0.0, min(1.0, face_top_expand_ratio))
        self._detection_input_scale = max(1.0, min(3.0, float(detection_input_scale)))

        if hasattr(mp, "solutions"):
            self._backend = "solutions"
            self._face_detection = mp.solutions.face_detection.FaceDetection(
                model_selection=model_selection,
                min_detection_confidence=min_detection_confidence,
            )
        else:
            self._init_tasks_backend(
                min_detection_confidence=min_detection_confidence,
                model_asset_path=model_asset_path,
            )

        self.last_face_box: Optional[RoiBox] = None
        self.last_forehead_box: Optional[RoiBox] = None
        self.last_score: Optional[float] = None

    def close(self) -> None:
        if self._face_detection is not None:
            self._face_detection.close()
        if self._tasks_detector is not None:
            self._tasks_detector.close()

    def get_forehead_roi(self, rgb_frame) -> Optional[RoiBox]:
        h, w = rgb_frame.shape[:2]
        det_scale = self._detection_input_scale
        if abs(det_scale - 1.0) > 1e-6:
            det_w = max(2, int(round(w * det_scale)))
            det_h = max(2, int(round(h * det_scale)))
            det_frame = cv2.resize(rgb_frame, (det_w, det_h), interpolation=cv2.INTER_LINEAR)
        else:
            det_w = w
            det_h = h
            det_frame = rgb_frame

        if self._backend == "solutions":
            frame_rgb = cv2.cvtColor(det_frame, cv2.COLOR_BGR2RGB)
            results = self._face_detection.process(frame_rgb)
            if not results.detections:
                self.last_face_box = None
                self.last_forehead_box = None
                self.last_score = None
                return None

            best = max(results.detections, key=lambda det: float(det.score[0]) if det.score else 0.0)
            rel = best.location_data.relative_bounding_box
            score = float(best.score[0]) if best.score else 0.0
            x1 = int(rel.xmin * det_w)
            y1 = int(rel.ymin * det_h)
            x2 = int((rel.xmin + rel.width) * det_w)
            y2 = int((rel.ymin + rel.height) * det_h)
            eyes = self._extract_eye_points(best, w=det_w, h=det_h)
        else:
            frame_rgb = cv2.cvtColor(det_frame, cv2.COLOR_BGR2RGB)
            mp_image = self._tasks_image_cls(image_format=self._tasks_image_format.SRGB, data=frame_rgb)
            result = self._tasks_detector.detect(mp_image)
            if not result.detections:
                self.last_face_box = None
                self.last_forehead_box = None
                self.last_score = None
                return None

            def _score(det) -> float:
                if det.categories:
                    return float(det.categories[0].score)
                return 0.0

            best = max(result.detections, key=_score)
            score = _score(best)
            box = best.bounding_box
            x1 = int(box.origin_x)
            y1 = int(box.origin_y)
            x2 = int(box.origin_x + box.width)
            y2 = int(box.origin_y + box.height)
            eyes = self._extract_eye_points(best, w=det_w, h=det_h)

        if abs(det_scale - 1.0) > 1e-6:
            x1 = int(round(x1 / det_scale))
            y1 = int(round(y1 / det_scale))
            x2 = int(round(x2 / det_scale))
            y2 = int(round(y2 / det_scale))
            if eyes is not None:
                rx, ry, lx, ly = eyes
                eyes = (
                    rx / det_scale,
                    ry / det_scale,
                    lx / det_scale,
                    ly / det_scale,
                )

        raw_face = self._clip_box(RoiBox(x1=x1, y1=y1, x2=x2, y2=y2), w=w, h=h)
        if raw_face is None:
            self.last_face_box = None
            self.last_forehead_box = None
            self.last_score = None
            return None

        face = self._expand_face_up(raw_face, w=w, h=h)
        if face is None:
            self.last_face_box = None
            self.last_forehead_box = None
            self.last_score = None
            return None

        forehead = self._estimate_forehead_box(face=face, eyes=eyes)
        forehead = self._clip_box(forehead, w=w, h=h)
        if forehead is None:
            self.last_face_box = None
            self.last_forehead_box = None
            self.last_score = None
            return None

        self.last_face_box = face
        self.last_forehead_box = forehead
        self.last_score = score
        return forehead

    def _clip_box(self, box: RoiBox, w: int, h: int) -> Optional[RoiBox]:
        x1 = max(0, min(w - 1, box.x1))
        y1 = max(0, min(h - 1, box.y1))
        x2 = max(0, min(w - 1, box.x2))
        y2 = max(0, min(h - 1, box.y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return RoiBox(x1=x1, y1=y1, x2=x2, y2=y2)

    def _expand_face_up(self, face: RoiBox, w: int, h: int) -> Optional[RoiBox]:
        fh = face.y2 - face.y1
        expand_px = int(round(self._face_top_expand_ratio * fh))
        expanded = RoiBox(
            x1=face.x1,
            y1=face.y1 - expand_px,
            x2=face.x2,
            y2=face.y2,
        )
        return self._clip_box(expanded, w=w, h=h)

    def _init_tasks_backend(self, min_detection_confidence: float, model_asset_path: str) -> None:
        try:
            from mediapipe.tasks.python.core.base_options import BaseOptions
            from mediapipe.tasks.python.vision import FaceDetector
            from mediapipe.tasks.python.vision import FaceDetectorOptions
            from mediapipe.tasks.python.vision.core.image import Image
            from mediapipe.tasks.python.vision.core.image import ImageFormat
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "This mediapipe build has no 'solutions' and tasks API import failed. "
                "Install mediapipe==0.10.14 or compatible."
            ) from exc

        model_path = (model_asset_path or "").strip()
        if not model_path:
            raise RuntimeError(
                "This mediapipe build does not expose mp.solutions. "
                "Set MP_FACE_MODEL_PATH in .env to a face detector model "
                "(e.g. face_detection_short_range.tflite), or install mediapipe==0.10.14."
            )

        model_file = Path(model_path)
        if not model_file.exists():
            raise RuntimeError(f"MP_FACE_MODEL_PATH does not exist: {model_file}")

        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(model_file)),
            min_detection_confidence=min_detection_confidence,
        )
        self._tasks_detector = FaceDetector.create_from_options(options)
        self._tasks_image_cls = Image
        self._tasks_image_format = ImageFormat
        self._backend = "tasks"

    def _extract_eye_points(self, detection, w: int, h: int) -> Optional[tuple[float, float, float, float]]:
        if self._backend == "solutions":
            location = getattr(detection, "location_data", None)
            rel_points = getattr(location, "relative_keypoints", None) if location is not None else None
            if rel_points and len(rel_points) >= 2:
                right_eye = rel_points[0]
                left_eye = rel_points[1]
                return (
                    float(right_eye.x) * w,
                    float(right_eye.y) * h,
                    float(left_eye.x) * w,
                    float(left_eye.y) * h,
                )
            return None

        keypoints = getattr(detection, "keypoints", None)
        if keypoints and len(keypoints) >= 2:
            right_eye = keypoints[0]
            left_eye = keypoints[1]
            rx, ry = self._to_pixel_point(right_eye, w=w, h=h)
            lx, ly = self._to_pixel_point(left_eye, w=w, h=h)
            return (rx, ry, lx, ly)
        return None

    def _to_pixel_point(self, point_obj, w: int, h: int) -> tuple[float, float]:
        x = float(getattr(point_obj, "x", 0.0))
        y = float(getattr(point_obj, "y", 0.0))
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            return x * w, y * h
        return x, y

    def _estimate_forehead_box(
        self,
        face: RoiBox,
        eyes: Optional[tuple[float, float, float, float]],
    ) -> RoiBox:
        fw = face.x2 - face.x1
        fh = face.y2 - face.y1

        if eyes is not None:
            rx, ry, lx, ly = eyes
            eye_y = (ry + ly) * 0.5
            eye_center_x = (rx + lx) * 0.5
            eye_span = abs(rx - lx)

            top = int(face.y1 + 0.04 * fh)
            bottom = int(min(face.y1 + 0.30 * fh, eye_y - 0.08 * fh))
            if bottom <= top + int(0.08 * fh):
                bottom = int(face.y1 + 0.26 * fh)

            half_w = max(0.28 * fw, 0.70 * eye_span)
            x1 = int(eye_center_x - half_w)
            x2 = int(eye_center_x + half_w)
            return RoiBox(x1=x1, y1=top, x2=x2, y2=bottom)

        return RoiBox(
            x1=int(face.x1 + 0.25 * fw),
            y1=int(face.y1 + 0.04 * fh),
            x2=int(face.x1 + 0.75 * fw),
            y2=int(face.y1 + 0.26 * fh),
        )


class NullRoiProvider:
    """Default placeholder until YOLO/MediaPipe ROI provider is integrated."""

    def get_forehead_roi(self, rgb_frame) -> Optional[RoiBox | RoiPolygon]:
        return None
