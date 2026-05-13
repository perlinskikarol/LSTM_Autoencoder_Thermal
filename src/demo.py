from __future__ import annotations

import logging
import time

import cv2

from src.config import load_settings
from src.hik_isapi_metadata import HikMetadataClient
from src.roi_provider import NullRoiProvider
from src.rtsp_stream import RtspStream


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    try:
        cfg = load_settings()
    except ValueError as exc:
        logging.error("%s", exc)
        return

    rgb_stream = RtspStream(name="RGB", rtsp_url=cfg.rtsp_rgb, reconnect_delay_sec=cfg.reconnect_delay_sec)
    th_stream = RtspStream(name="THERMAL", rtsp_url=cfg.rtsp_th, reconnect_delay_sec=cfg.reconnect_delay_sec)

    metadata = None
    if cfg.enable_metadata:
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
    else:
        logging.info("Metadata is disabled (ENABLE_METADATA=false). Running video-only mode.")

    roi_provider = NullRoiProvider()

    rgb_stream.start()
    th_stream.start()
    if metadata is not None:
        metadata.start()

    display_w = 1280
    display_h = 720
    cv2.namedWindow("RGB", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Thermal", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RGB", display_w, display_h)
    cv2.resizeWindow("Thermal", display_w, display_h)

    last_print_ts = 0.0

    try:
        while True:
            rgb_frame = rgb_stream.get_last_frame()
            th_frame = th_stream.get_last_frame()

            if rgb_frame is not None:
                # TODO(ROI): plug YOLO/MediaPipe forehead detector here.
                roi = roi_provider.get_forehead_roi(rgb_frame)
                if roi is not None:
                    pass
                    # TODO(ROI): map RGB ROI -> thermal coordinates.
                    # TODO(ROI): choose matching thermometry ruleID/target for ROI.
                    # TODO(ROI): pick tempProperty (average/highest/lowest) for ROI output.

                rgb_view = cv2.resize(rgb_frame, (display_w, display_h), interpolation=cv2.INTER_AREA)
                cv2.imshow("RGB", rgb_view)

            if th_frame is not None:
                th_view = cv2.resize(th_frame, (display_w, display_h), interpolation=cv2.INTER_AREA)
                cv2.imshow("Thermal", th_view)

            now = time.time()
            if now - last_print_ts > 1.0:
                last_print_ts = now
                if metadata is not None:
                    latest = metadata.get_latest_temperatures()
                    if latest:
                        for item in latest:
                            print(
                                f"[{item.timestamp}] rule={item.rule_id} "
                                f"temp={item.temp_value:.2f} {item.temp_unit} "
                                f"property={item.temp_property} region_points={len(item.region)}"
                            )

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
    finally:
        stoppers = [rgb_stream.stop, th_stream.stop]
        if metadata is not None:
            stoppers.insert(0, metadata.stop)
        for stopper in stoppers:
            try:
                stopper()
            except Exception as exc:  # noqa: BLE001
                logging.debug("shutdown warning: %s", exc)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
