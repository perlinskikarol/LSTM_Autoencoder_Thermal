from __future__ import annotations

import hashlib
import json
import logging
import re
import secrets
import socket
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote, urlparse

import numpy as np
import requests
from lxml import etree
from requests.auth import HTTPBasicAuth, HTTPDigestAuth
from requests.utils import parse_dict_header


@dataclass(frozen=True)
class RegionPoint:
    x: float
    y: float


@dataclass(frozen=True)
class TempReading:
    timestamp: str
    sub_type: str
    rule_id: Optional[str]
    temp_value: float
    temp_unit: str
    temp_property: str
    region: list[RegionPoint]


@dataclass(frozen=True)
class ThermalPixelFrame:
    timestamp: str
    width: int
    height: int
    temp_c: np.ndarray


class HikMetadataClient:
    def __init__(
        self,
        ip: str,
        user: str,
        password: str,
        channel_id: int = 101,
        rtsp_port: int = 554,
        reconnect_delay_sec: float = 2.0,
        raw_log_path: str = "logs/raw_thermo.xml",
        request_timeout_sec: float = 8.0,
        mode: str = "legacy",
        forced_legacy_uri: str = "",
        forced_http_endpoint: str = "",
        auth_lockout_sleep_sec: float = 1200.0,
        max_auth_failures: int = 1,
    ) -> None:
        self.ip = ip
        self.user = user
        self.password = password
        self.channel_id = channel_id
        self.rtsp_port = rtsp_port
        self.reconnect_delay_sec = reconnect_delay_sec
        self.request_timeout_sec = request_timeout_sec
        self.mode = mode.lower().strip()
        if self.mode not in {"legacy", "auto", "http_thermal", "http_thermal_p2p"}:
            self.mode = "legacy"
        self.forced_legacy_uri = forced_legacy_uri.strip()
        self.forced_http_endpoint = forced_http_endpoint.strip()
        self.auth_lockout_sleep_sec = max(20.0, auth_lockout_sleep_sec)
        self.max_auth_failures = max(1, int(max_auth_failures))

        self._base_http = f"http://{self.ip}"
        self._session = requests.Session()
        # Ignore system proxy settings (e.g. stale Fiddler 127.0.0.1:8888).
        self._session.trust_env = False
        self._digest_auth = HTTPDigestAuth(self.user, self.password)
        self._basic_auth = HTTPBasicAuth(self.user, self.password)
        self._session.auth = self._digest_auth

        self._raw_log_path = Path(raw_log_path)
        self._raw_log_path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._latest_readings: list[TempReading] = []
        self._latest_pixel_frame: Optional[ThermalPixelFrame] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._rtsp_digest: dict[str, str] = {}
        self._rtsp_nonce_counter = 0
        self._fallback_rtsp_index = 0
        self._consecutive_auth_failures = 0
        self._http_rules_endpoint_index = 0
        self._selected_http_endpoint: str | None = None
        self._selected_http_endpoint_empty_count = 0
        self._http_p2p_endpoint_index = 0
        self._selected_http_p2p_endpoint: str | None = None
        self._selected_http_p2p_empty_count = 0
        self._last_empty_payload_dump_ts = 0.0
        self._empty_payload_dump_interval_sec = 15.0
        self._empty_payload_log_path = Path("logs/raw_http_thermo_empty.log")
        self._empty_payload_log_path.parent.mkdir(parents=True, exist_ok=True)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="HikMetadataClient", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        # Closing session helps unblock pending HTTP calls during shutdown.
        self._session.close()
        try:
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=0.5)
        except KeyboardInterrupt:
            # Allow fast exit when user hits Ctrl+C during shutdown.
            pass

    def get_latest_temperatures(self) -> list[TempReading]:
        with self._lock:
            return list(self._latest_readings)

    def get_latest_pixel_frame(self) -> Optional[ThermalPixelFrame]:
        with self._lock:
            return self._latest_pixel_frame

    def get_temperature_for_rule(self, rule_id, property: str = "average") -> float | None:
        rule_text = str(rule_id)
        target_property = property.lower()
        for item in self.get_latest_temperatures():
            if item.rule_id == rule_text and item.temp_property.lower() == target_property:
                return item.temp_value
        return None

    def get_latest_by_rule(self, rule_id=None) -> list[TempReading]:
        if rule_id is None:
            return self.get_latest_temperatures()
        rule_text = str(rule_id)
        return [r for r in self.get_latest_temperatures() if r.rule_id == rule_text]

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                if self.mode == "http_thermal":
                    self._poll_http_thermal_once()
                    self._consecutive_auth_failures = 0
                    if not self._stop_event.is_set():
                        time.sleep(max(1.0, self.reconnect_delay_sec))
                    continue
                if self.mode == "http_thermal_p2p":
                    self._poll_http_thermal_p2p_once()
                    self._consecutive_auth_failures = 0
                    if not self._stop_event.is_set():
                        time.sleep(max(1.0, self.reconnect_delay_sec))
                    continue

                rtsp_uri = self._prepare_rtsp_uri()
                self._read_metadata_stream(rtsp_uri)
                self._consecutive_auth_failures = 0
            except RuntimeError as exc:
                msg = str(exc)
                if "DESCRIBE failed with status 401" in msg:
                    sleep_s = self._handle_auth_failure(
                        "Metadata RTSP rejected auth (401). Camera may be locked or metadata RTSP path is blocked."
                    )
                elif "SETUP failed with status 401" in msg or "PLAY failed with status 401" in msg:
                    sleep_s = self._handle_auth_failure("Metadata RTSP auth failed after DESCRIBE (401).")
                elif "Metadata track not found in SDP" in msg:
                    logging.warning("%s", msg)
                    sleep_s = max(5.0, self.reconnect_delay_sec)
                else:
                    logging.exception("metadata runtime error: %s", exc)
                    sleep_s = self.reconnect_delay_sec
                if not self._stop_event.is_set():
                    time.sleep(sleep_s)
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                if status in (401, 403):
                    sleep_s = self._handle_auth_failure(
                        "ISAPI auth failed (%s). Check HIK_USER/HIK_PASS and camera user permissions for ISAPI/metadata."
                        % status
                    )
                else:
                    logging.exception("metadata HTTP error: %s", exc)
                    sleep_s = self.reconnect_delay_sec
                if not self._stop_event.is_set():
                    time.sleep(sleep_s)
            except Exception as exc:  # noqa: BLE001
                logging.exception("metadata loop error: %s", exc)
                if not self._stop_event.is_set():
                    time.sleep(self.reconnect_delay_sec)

    def _prepare_rtsp_uri(self) -> str:
        if self.mode == "legacy":
            if self.forced_legacy_uri:
                logging.info("Metadata mode=legacy, using forced RTSP URI: %s", self.forced_legacy_uri)
                return self.forced_legacy_uri
            candidates = self._default_metadata_rtsp_uris()
            uri = candidates[self._fallback_rtsp_index % len(candidates)]
            self._fallback_rtsp_index += 1
            logging.info("Metadata mode=legacy, using fallback RTSP URI: %s", uri)
            return uri

        self._get_metadata_capabilities()
        self._try_enable_thermometry()
        rtsp_uri = self._subscribe_thermometry_or_default()
        if "@" not in rtsp_uri:
            parsed = urlparse(rtsp_uri)
            host = parsed.hostname or self.ip
            port = parsed.port or self.rtsp_port
            path = parsed.path or ""
            query = f"?{parsed.query}" if parsed.query else ""
            user_q = quote(self.user, safe="")
            pass_q = quote(self.password, safe="")
            rtsp_uri = f"rtsp://{user_q}:{pass_q}@{host}:{port}{path}{query}"
        return rtsp_uri

    def _poll_http_thermal_once(self) -> None:
        endpoint = self._next_http_rules_endpoint()
        response = self._isapi_request(
            "GET",
            endpoint,
            stream=True,
            timeout=(3.0, 3.0),
            headers={"Connection": "close", "Accept": "application/json, application/xml, */*"},
        )
        try:
            if response.status_code == 401:
                response.raise_for_status()
            if response.status_code in (403, 404, 405):
                logging.warning(
                    "HTTP thermometry endpoint unavailable/forbidden: %s (status=%s)",
                    endpoint,
                    response.status_code,
                )
                if self._selected_http_endpoint == endpoint:
                    self._selected_http_endpoint = None
                return
            response.raise_for_status()
            self._selected_http_endpoint = endpoint

            payload = self._read_http_payload(response, max_wait_sec=3.0, max_bytes=512 * 1024)
            if not payload:
                logging.warning("HTTP thermometry returned no parsable payload yet: %s", endpoint)
                return

            readings = self._parse_http_thermometry_payload(payload, response.headers.get("Content-Type", ""))
        finally:
            response.close()

        if not readings:
            self._dump_empty_http_payload(endpoint=endpoint, content_type=response.headers.get("Content-Type", ""), payload=payload)
            if not self.forced_http_endpoint and self._selected_http_endpoint == endpoint:
                self._selected_http_endpoint_empty_count += 1
                if self._selected_http_endpoint_empty_count >= 1:
                    logging.info(
                        "No thermometry values on endpoint %s for %d polls. Rotating HTTP endpoint.",
                        endpoint,
                        self._selected_http_endpoint_empty_count,
                    )
                    self._selected_http_endpoint = None
                    self._selected_http_endpoint_empty_count = 0
            logging.info("No thermometry values found in HTTP response from %s", endpoint)
            return
        self._selected_http_endpoint_empty_count = 0

        with self._lock:
            self._latest_readings = readings

    def _poll_http_thermal_p2p_once(self) -> None:
        endpoint = self._next_http_p2p_endpoint()
        response = self._isapi_request(
            "GET",
            endpoint,
            stream=False,
            timeout=(3.0, 10.0),
            headers={"Connection": "close", "Accept": "multipart/form-data, application/json, */*"},
        )
        try:
            if response.status_code == 401:
                response.raise_for_status()
            if response.status_code in (403, 404, 405):
                logging.warning(
                    "HTTP thermal P2P endpoint unavailable/forbidden: %s (status=%s)",
                    endpoint,
                    response.status_code,
                )
                if self._selected_http_p2p_endpoint == endpoint:
                    self._selected_http_p2p_endpoint = None
                return
            response.raise_for_status()
            self._selected_http_p2p_endpoint = endpoint
            payload = response.content or b""
            content_type = response.headers.get("Content-Type", "")
            frame = self._parse_http_p2p_payload(payload=payload, content_type=content_type)
        finally:
            response.close()

        if frame is None:
            self._dump_empty_http_payload(endpoint=endpoint, content_type=content_type, payload=payload)
            if not self.forced_http_endpoint and self._selected_http_p2p_endpoint == endpoint:
                self._selected_http_p2p_empty_count += 1
                if self._selected_http_p2p_empty_count >= 1:
                    logging.info(
                        "No P2P thermal frame on endpoint %s for %d polls. Rotating HTTP endpoint.",
                        endpoint,
                        self._selected_http_p2p_empty_count,
                    )
                    self._selected_http_p2p_endpoint = None
                    self._selected_http_p2p_empty_count = 0
            logging.info("No thermal P2P frame found in HTTP response from %s", endpoint)
            return

        self._selected_http_p2p_empty_count = 0
        readings = self._build_summary_readings_from_p2p(frame)
        with self._lock:
            self._latest_pixel_frame = frame
            self._latest_readings = readings

    def _next_http_p2p_endpoint(self) -> str:
        if self.forced_http_endpoint:
            logging.info("HTTP thermal P2P forced endpoint: %s", self.forced_http_endpoint)
            return self.forced_http_endpoint
        if self._selected_http_p2p_endpoint:
            return self._selected_http_p2p_endpoint
        candidates = self._http_p2p_endpoints()
        endpoint = candidates[self._http_p2p_endpoint_index % len(candidates)]
        self._http_p2p_endpoint_index += 1
        logging.info("HTTP thermal P2P endpoint: %s", endpoint)
        return endpoint

    def _http_p2p_endpoints(self) -> list[str]:
        channel_candidates: list[int] = []
        for candidate in (self.channel_id, 2, 1):
            if candidate > 0 and candidate not in channel_candidates:
                channel_candidates.append(candidate)

        suffixes = [
            "thermometry/jpegPicWithAppendData?format=json",
            "thermometry/jpegPicWithAppendData",
        ]
        out: list[str] = []
        for channel in channel_candidates:
            for suffix in suffixes:
                out.append(f"{self._base_http}/ISAPI/Thermal/channels/{channel}/{suffix}")
        return out

    def _read_http_payload(self, response, max_wait_sec: float, max_bytes: int) -> bytes:
        start = time.time()
        data = bytearray()
        iterator = response.iter_content(chunk_size=4096)
        while (time.time() - start) < max_wait_sec and len(data) < max_bytes:
            try:
                chunk = next(iterator)
            except StopIteration:
                break
            except Exception:
                break
            if not chunk:
                continue
            data.extend(chunk)

            blob = bytes(data)
            if self._looks_like_complete_json(blob):
                break
            if b"</Metadata>" in blob or b"</EventNotificationAlert>" in blob:
                break

            content_length = response.headers.get("Content-Length")
            if content_length:
                try:
                    if len(blob) >= int(content_length):
                        break
                except ValueError:
                    pass
        return bytes(data)

    def _looks_like_complete_json(self, blob: bytes) -> bool:
        text = blob.decode("utf-8", errors="ignore").strip()
        if not text:
            return False
        if text[0] not in "[{":
            return False
        try:
            json.loads(text)
            return True
        except Exception:
            return False

    def _next_http_rules_endpoint(self) -> str:
        if self.forced_http_endpoint:
            logging.info("HTTP thermometry forced endpoint: %s", self.forced_http_endpoint)
            return self.forced_http_endpoint
        if self._selected_http_endpoint:
            return self._selected_http_endpoint
        candidates = self._http_rules_endpoints()
        endpoint = candidates[self._http_rules_endpoint_index % len(candidates)]
        self._http_rules_endpoint_index += 1
        logging.info("HTTP thermometry endpoint: %s", endpoint)
        return endpoint

    def _http_rules_endpoints(self) -> list[str]:
        channel_candidates: list[int] = []
        for candidate in (self.channel_id, 2, 1):
            if candidate > 0 and candidate not in channel_candidates:
                channel_candidates.append(candidate)

        paths = [
            "thermometry/realTimethermometry/rules?format=json",
            "thermometry/realTimethermometry/rules",
            "thermometry/realTimethermometry?format=json",
            "thermometry/realTimethermometry",
            "thermometry/bodyTemperature?format=json",
            "thermometry/bodyTemperature",
            "bodyTemperature?format=json",
            "bodyTemperature",
            "thermometry/temperature?format=json",
            "thermometry/temperature",
        ]

        out: list[str] = []
        for channel in channel_candidates:
            for path in paths:
                out.append(f"{self._base_http}/ISAPI/Thermal/channels/{channel}/{path}")
        # Additional firmware fallback: generic event stream may carry thermometry metadata.
        out.append(f"{self._base_http}/ISAPI/Event/notification/alertStream")
        out.append(f"{self._base_http}/ISAPI/Event/notification/alertStream?format=json")
        return out

    def _parse_http_p2p_payload(self, payload: bytes, content_type: str) -> Optional[ThermalPixelFrame]:
        if not payload:
            return None

        parts = self._parse_multipart_payload(payload=payload, content_type=content_type)
        json_docs: list[dict[str, Any]] = []
        binary_parts: list[tuple[dict[str, str], bytes]] = []

        for headers, body in parts:
            ctype = headers.get("content-type", "").lower()
            if "json" in ctype:
                text = body.decode("utf-8", errors="ignore")
                for fragment in self._extract_all_json_documents(text):
                    try:
                        parsed = json.loads(fragment)
                    except Exception:
                        continue
                    if isinstance(parsed, dict):
                        json_docs.append(parsed)
            elif "image" in ctype or "pjpeg" in ctype or "jpeg" in ctype:
                continue
            else:
                if body:
                    binary_parts.append((headers, body))

        if not json_docs:
            # Non-standard response fallback: try scanning full payload for JSON metadata.
            text = payload.decode("utf-8", errors="ignore")
            for fragment in self._extract_all_json_documents(text):
                try:
                    parsed = json.loads(fragment)
                except Exception:
                    continue
                if isinstance(parsed, dict):
                    json_docs.append(parsed)
            if not json_docs:
                return None

        descriptor = self._find_p2p_descriptor(json_docs)
        if descriptor is None:
            return None

        width = int(descriptor.get("jpegPicWidth") or 0)
        height = int(descriptor.get("jpegPicHeight") or 0)
        p2p_len = int(descriptor.get("p2pDataLen") or 0)
        temp_len = int(descriptor.get("temperatureDataLength") or 4)
        if width <= 0 or height <= 0 or p2p_len <= 0:
            return None

        p2p_bytes = self._select_p2p_binary(binary_parts=binary_parts, expected_len=p2p_len)
        if p2p_bytes is None:
            # Last fallback: detect likely p2p tail by exact length.
            p2p_bytes = self._fallback_find_binary_chunk(payload=payload, expected_len=p2p_len)
            if p2p_bytes is None:
                return None

        matrix = self._decode_p2p_matrix(
            blob=p2p_bytes,
            width=width,
            height=height,
            temperature_data_len=temp_len,
        )
        if matrix is None:
            return None

        return ThermalPixelFrame(
            timestamp=self._now_iso(),
            width=width,
            height=height,
            temp_c=matrix,
        )

    def _find_p2p_descriptor(self, docs: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
        for doc in docs:
            node = doc.get("JpegPictureWithAppendData")
            if isinstance(node, dict):
                return node
            # Some firmwares nest descriptor in wrappers.
            for value in doc.values():
                if isinstance(value, dict) and "JpegPictureWithAppendData" in value:
                    nested = value.get("JpegPictureWithAppendData")
                    if isinstance(nested, dict):
                        return nested
        return None

    def _select_p2p_binary(self, binary_parts: list[tuple[dict[str, str], bytes]], expected_len: int) -> Optional[bytes]:
        if not binary_parts:
            return None

        exact_match = [body for _, body in binary_parts if len(body) == expected_len]
        if exact_match:
            return exact_match[0]

        # Prefer non-image, largest chunk as best candidate.
        largest = max(binary_parts, key=lambda item: len(item[1]))
        if len(largest[1]) >= expected_len:
            return largest[1][:expected_len]
        return None

    def _fallback_find_binary_chunk(self, payload: bytes, expected_len: int) -> Optional[bytes]:
        if expected_len <= 0 or len(payload) < expected_len:
            return None
        # P2P blob is usually at the end of multipart body.
        tail = payload[-expected_len:]
        if len(tail) == expected_len:
            return tail
        return None

    def _decode_p2p_matrix(
        self,
        blob: bytes,
        width: int,
        height: int,
        temperature_data_len: int,
    ) -> Optional[np.ndarray]:
        pixel_count = width * height
        if pixel_count <= 0:
            return None

        expected = pixel_count * max(1, temperature_data_len)
        if len(blob) < expected:
            return None
        blob = blob[:expected]

        try:
            if temperature_data_len == 4:
                arr = np.frombuffer(blob, dtype="<f4", count=pixel_count).astype(np.float32, copy=False)
            elif temperature_data_len == 2:
                arr_u16 = np.frombuffer(blob, dtype="<u2", count=pixel_count).astype(np.float32, copy=False)
                # Common Hikvision format: decide between direct Celsius or 0.1 Celsius.
                median_val = float(np.median(arr_u16)) if arr_u16.size else 0.0
                arr = arr_u16 / 10.0 if median_val > 200.0 else arr_u16
            elif temperature_data_len == 1:
                arr = np.frombuffer(blob, dtype=np.uint8, count=pixel_count).astype(np.float32, copy=False)
            else:
                return None
        except Exception:
            return None

        if arr.size != pixel_count:
            return None
        matrix = arr.reshape((height, width))
        finite_mask = np.isfinite(matrix)
        if not np.any(finite_mask):
            return None
        return matrix

    def _build_summary_readings_from_p2p(self, frame: ThermalPixelFrame) -> list[TempReading]:
        matrix = frame.temp_c
        finite = matrix[np.isfinite(matrix)]
        if finite.size == 0:
            return []

        avg = float(np.mean(finite))
        hi = float(np.max(finite))
        lo = float(np.min(finite))

        return [
            TempReading(
                timestamp=frame.timestamp,
                sub_type="thermometry",
                rule_id=None,
                temp_value=avg,
                temp_unit="centigrade",
                temp_property="average",
                region=[],
            ),
            TempReading(
                timestamp=frame.timestamp,
                sub_type="thermometry",
                rule_id=None,
                temp_value=hi,
                temp_unit="centigrade",
                temp_property="highest",
                region=[],
            ),
            TempReading(
                timestamp=frame.timestamp,
                sub_type="thermometry",
                rule_id=None,
                temp_value=lo,
                temp_unit="centigrade",
                temp_property="lowest",
                region=[],
            ),
        ]

    def _parse_http_thermometry_payload(self, payload: bytes, content_type: str) -> list[TempReading]:
        content_lower = content_type.lower()
        collected: list[TempReading] = []

        # JSON path is preferred for HTTP thermal rules.
        if "json" in content_lower:
            try:
                data = json.loads(payload.decode("utf-8", errors="ignore"))
            except Exception:
                data = None
            if data is not None:
                readings = self._extract_json_readings(data)
                if readings:
                    collected.extend(readings)

        # Some firmwares return XML despite JSON request.
        if "xml" in content_lower or payload.lstrip().startswith(b"<"):
            xml_readings = self._extract_xml_rule_readings(payload)
            if xml_readings:
                collected.extend(xml_readings)

        # Fallback: try JSON first, then XML.
        try:
            data = json.loads(payload.decode("utf-8", errors="ignore"))
            readings = self._extract_json_readings(data)
            if readings:
                collected.extend(readings)
        except Exception:
            pass

        # multipart/form-data streams often embed multiple JSON parts.
        text_payload = payload.decode("utf-8", errors="ignore")
        for json_fragment in self._extract_all_json_documents(text_payload):
            try:
                data = json.loads(json_fragment)
                readings = self._extract_json_readings(data)
                if readings:
                    collected.extend(readings)
            except Exception:
                pass

        xml_readings = self._extract_xml_rule_readings(payload)
        if xml_readings:
            collected.extend(xml_readings)
        xml_fragment_readings = self._extract_xml_fragment_readings(payload)
        if xml_fragment_readings:
            collected.extend(xml_fragment_readings)
        return self._dedupe_temp_readings(collected)

    def _extract_json_readings(self, data: Any) -> list[TempReading]:
        fallback_timestamp = self._now_iso()
        readings: list[TempReading] = []
        seen: set[tuple[str, str, float]] = set()

        property_map = {
            "tempvalue": "average",
            "temperature": "average",
            "averagetemperature": "average",
            "avgtemperature": "average",
            "bodytemperature": "average",
            "facetemperature": "average",
            "skintemperature": "average",
            "surfaceTemperature".lower(): "average",
            "correctedtemperature": "average",
            "targettemperature": "average",
            "realtimebodytemperature": "average",
            "temperaturevalue": "average",
            "maxtemperature": "highest",
            "highesttemperature": "highest",
            "mintemperature": "lowest",
            "lowesttemperature": "lowest",
        }

        def walk(node: Any) -> None:
            if isinstance(node, dict):
                # Property list style:
                # [{"description":"tempValue","value":"30"}, ...]
                if "description" in node and "value" in node:
                    desc = str(node.get("description", "")).strip().lower()
                    if desc in property_map:
                        val = self._to_float(node.get("value"))
                        if val is not None:
                            prop = property_map.get(desc, "average")
                            key = ("", prop, val)
                            if key not in seen:
                                seen.add(key)
                                readings.append(
                                    TempReading(
                                        timestamp=fallback_timestamp,
                                        sub_type="thermometry",
                                        rule_id=None,
                                        temp_value=val,
                                        temp_unit="centigrade",
                                        temp_property=prop,
                                        region=[],
                                    )
                                )

                rule_id = self._json_first_present(node, ["ruleID", "ruleId", "ruleNo", "id", "rule", "presetNo"])
                unit_raw = self._json_first_present(node, ["tempUnit", "temperatureUnit", "unit", "thermometryUnit"])
                unit = self._normalize_temp_unit(unit_raw)
                ts_raw = self._json_first_present(node, ["time", "timestamp", "measureTime", "absTime", "relativeTime"])
                ts = self._normalize_timestamp(ts_raw, fallback_timestamp)
                region = self._extract_region_points_from_json_node(node)

                for key, prop in property_map.items():
                    raw_value = self._json_lookup_case_insensitive(node, key)
                    if raw_value is None:
                        continue
                    temp = self._to_float(raw_value)
                    if temp is None:
                        continue
                    dedupe_key = (str(rule_id or ""), prop, temp)
                    if dedupe_key in seen:
                        continue
                    seen.add(dedupe_key)
                    readings.append(
                        TempReading(
                            timestamp=str(ts),
                            sub_type="thermometry",
                            rule_id=str(rule_id) if rule_id is not None else None,
                            temp_value=temp,
                            temp_unit=str(unit),
                            temp_property=prop,
                            region=region,
                        )
                    )

                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(data)
        return readings

    def _extract_xml_rule_readings(self, payload: bytes) -> list[TempReading]:
        try:
            root = etree.fromstring(payload)
        except Exception:
            return []

        timestamp = self._text_by_local_name(root, "time") or self._now_iso()
        unit = self._normalize_temp_unit(self._text_by_local_name(root, "tempUnit") or "centigrade")
        rule_id = self._text_by_local_name(root, "ruleID") or None

        out: list[TempReading] = []
        for name, prop in (
            ("tempValue", "average"),
            ("temperature", "average"),
            ("averageTemperature", "average"),
            ("avgTemperature", "average"),
            ("bodyTemperature", "average"),
            ("faceTemperature", "average"),
            ("skinTemperature", "average"),
            ("correctedTemperature", "average"),
            ("targetTemperature", "average"),
            ("maxTemperature", "highest"),
            ("highestTemperature", "highest"),
            ("minTemperature", "lowest"),
            ("lowestTemperature", "lowest"),
        ):
            for node in root.xpath(f".//*[local-name()='{name}']"):
                if node.text is None:
                    continue
                temp_value = self._to_float(node.text.strip())
                if temp_value is None:
                    continue
                out.append(
                    TempReading(
                        timestamp=timestamp,
                        sub_type="thermometry",
                        rule_id=rule_id,
                        temp_value=temp_value,
                        temp_unit=unit,
                        temp_property=prop,
                        region=[],
                    )
                )

        # XML PropertyList style:
        # <Property><description>tempValue</description><value>36.7</value></Property>
        desc_map = {
            "tempvalue": "average",
            "temperature": "average",
            "averagetemperature": "average",
            "avgtemperature": "average",
            "bodytemperature": "average",
            "facetemperature": "average",
            "skintemperature": "average",
            "correctedtemperature": "average",
            "targettemperature": "average",
            "maxtemperature": "highest",
            "highesttemperature": "highest",
            "mintemperature": "lowest",
            "lowesttemperature": "lowest",
        }
        for prop_node in root.xpath(".//*[local-name()='Property']"):
            desc_node = self._children_by_local_name(prop_node, "description")
            val_node = self._children_by_local_name(prop_node, "value")
            if not desc_node or not val_node:
                continue
            desc_text = (desc_node[0].text or "").strip().lower()
            if desc_text not in desc_map:
                continue
            temp_value = self._to_float((val_node[0].text or "").strip())
            if temp_value is None:
                continue
            out.append(
                TempReading(
                    timestamp=timestamp,
                    sub_type="thermometry",
                    rule_id=rule_id,
                    temp_value=temp_value,
                    temp_unit=unit,
                    temp_property=desc_map[desc_text],
                    region=[],
                )
            )

        return out

    def _extract_xml_fragment_readings(self, payload: bytes) -> list[TempReading]:
        out: list[TempReading] = []
        pos = 0
        while True:
            start = payload.find(b"<?xml", pos)
            if start < 0:
                break
            end = payload.find(b"</Metadata>", start)
            if end < 0:
                break
            end += len(b"</Metadata>")
            fragment = payload[start:end]
            out.extend(self._parse_thermometry(fragment))
            pos = end
        return out

    def _extract_region_points_from_json_node(self, node: dict[str, Any]) -> list[RegionPoint]:
        region_value = self._json_lookup_case_insensitive(node, "region")
        if region_value is None:
            return []

        out: list[RegionPoint] = []

        def parse_point(point_obj: Any) -> None:
            if not isinstance(point_obj, dict):
                return
            x_val = self._json_first_present(point_obj, ["positionX", "x"])
            y_val = self._json_first_present(point_obj, ["positionY", "y"])
            try:
                if x_val is not None and y_val is not None:
                    out.append(RegionPoint(x=float(x_val), y=float(y_val)))
            except Exception:
                return

        if isinstance(region_value, list):
            for item in region_value:
                if isinstance(item, dict):
                    point = self._json_lookup_case_insensitive(item, "point")
                    if point is not None:
                        parse_point(point)
                    else:
                        parse_point(item)
        elif isinstance(region_value, dict):
            point = self._json_lookup_case_insensitive(region_value, "point")
            if point is not None:
                parse_point(point)
            else:
                parse_point(region_value)

        return out

    def _normalize_timestamp(self, ts_value: Any, fallback: str) -> str:
        if ts_value is None:
            return fallback
        try:
            # absTime/relativeTime are often unix timestamps.
            if isinstance(ts_value, (int, float)):
                return datetime.fromtimestamp(float(ts_value), tz=timezone.utc).isoformat()
            text = str(ts_value).strip()
            if not text:
                return fallback
            as_num = float(text)
            return datetime.fromtimestamp(as_num, tz=timezone.utc).isoformat()
        except Exception:
            return str(ts_value)

    def _to_float(self, value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)

        text = str(value).strip()
        if not text:
            return None

        # Common variants: "36.7", "36,7", "36.7 centigrade", "temp:36.7C"
        text = text.replace(",", ".")
        match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
        if not match:
            return None
        try:
            return float(match.group(0))
        except Exception:
            return None

    def _json_first_present(self, node: dict[str, Any], keys: list[str]) -> Any:
        for key in keys:
            value = self._json_lookup_case_insensitive(node, key.lower())
            if value is not None:
                return value
        return None

    def _json_lookup_case_insensitive(self, node: dict[str, Any], lookup_key_lower: str) -> Any:
        for key, value in node.items():
            if str(key).lower() == lookup_key_lower:
                return value
        return None

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _dump_empty_http_payload(self, endpoint: str, content_type: str, payload: bytes) -> None:
        now = time.time()
        if now - self._last_empty_payload_dump_ts < self._empty_payload_dump_interval_sec:
            return
        self._last_empty_payload_dump_ts = now

        snippet = payload[:65536]
        header = (
            f"\n--- {self._now_iso()} endpoint={endpoint} content_type={content_type} "
            f"bytes={len(payload)} ---\n"
        )
        try:
            with self._empty_payload_log_path.open("ab") as fp:
                fp.write(header.encode("utf-8", errors="ignore"))
                fp.write(snippet)
                fp.write(b"\n")
        except Exception:
            pass

    def _extract_first_json_document(self, text: str) -> str:
        if not text:
            return ""

        starts = [idx for idx in (text.find("{"), text.find("[")) if idx >= 0]
        if not starts:
            return ""
        start = min(starts)

        stack: list[str] = []
        in_string = False
        escaped = False
        opener_to_closer = {"{": "}", "[": "]"}
        closer_to_opener = {"}": "{", "]": "["}

        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                    continue
                if ch == "\\":
                    escaped = True
                    continue
                if ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
                continue
            if ch in opener_to_closer:
                stack.append(ch)
                continue
            if ch in closer_to_opener:
                if not stack:
                    return ""
                expected = closer_to_opener[ch]
                if stack[-1] != expected:
                    return ""
                stack.pop()
                if not stack:
                    return text[start : i + 1]

        return ""

    def _extract_all_json_documents(self, text: str) -> list[str]:
        if not text:
            return []

        decoder = json.JSONDecoder()
        out: list[str] = []
        idx = 0
        size = len(text)

        while idx < size:
            next_obj = text.find("{", idx)
            next_arr = text.find("[", idx)
            starts = [x for x in (next_obj, next_arr) if x >= 0]
            if not starts:
                break
            start = min(starts)

            try:
                _, consumed = decoder.raw_decode(text[start:])
            except Exception:
                idx = start + 1
                continue

            end = start + consumed
            if end > start:
                out.append(text[start:end])
            idx = max(end, start + 1)

        return out

    def _parse_multipart_payload(self, payload: bytes, content_type: str) -> list[tuple[dict[str, str], bytes]]:
        boundary_match = re.search(r'boundary="?([^";]+)"?', content_type, flags=re.IGNORECASE)
        if not boundary_match:
            return []

        boundary = boundary_match.group(1).encode("utf-8", errors="ignore")
        delimiter = b"--" + boundary
        segments = payload.split(delimiter)
        out: list[tuple[dict[str, str], bytes]] = []

        for segment in segments:
            part = segment.strip()
            if not part or part == b"--":
                continue
            if part.endswith(b"--"):
                part = part[:-2].strip()
            if b"\r\n\r\n" not in part:
                continue
            raw_headers, body = part.split(b"\r\n\r\n", 1)
            headers: dict[str, str] = {}
            for line in raw_headers.decode("utf-8", errors="ignore").split("\r\n"):
                if ":" not in line:
                    continue
                key, val = line.split(":", 1)
                headers[key.strip().lower()] = val.strip()
            body = body.rstrip(b"\r\n")
            out.append((headers, body))
        return out

    def _dedupe_temp_readings(self, readings: list[TempReading]) -> list[TempReading]:
        seen: set[tuple[str, str, str, float]] = set()
        out: list[TempReading] = []
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
            out.append(item)
        return out

    def _normalize_temp_unit(self, unit_raw: Any) -> str:
        if unit_raw is None:
            return "centigrade"
        text = str(unit_raw).strip().lower()
        if text in {"0", "centigrade", "celsius", "c"}:
            return "centigrade"
        if text in {"1", "fahrenheit", "f"}:
            return "fahrenheit"
        if text in {"2", "kelvin", "k"}:
            return "kelvin"
        return text or "centigrade"

    def _get_metadata_capabilities(self) -> str:
        urls = [
            f"{self._base_http}/ISAPI/Streaming/channels/{self.channel_id}/metadata/capabilities",
            f"{self._base_http}/ISAPI/Streaming/channels/{self.channel_id}/Metadata/capabilities",
        ]
        last_response = None
        for url in urls:
            resp = self._isapi_request("GET", url)
            last_response = resp
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            return resp.text

        if last_response is not None:
            last_response.raise_for_status()
        raise RuntimeError("Metadata capabilities endpoint not found on device.")

    def _try_enable_thermometry(self) -> None:
        url = f"{self._base_http}/ISAPI/Streaming/channels/{self.channel_id}/Metadata/thermometry"
        payload = (
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
            "<SingleMetadataCfg>"
            "<type>thermometry</type>"
            "<enable>true</enable>"
            "</SingleMetadataCfg>"
        )
        headers = {"Content-Type": "application/xml"}
        resp = self._isapi_request("PUT", url, data=payload.encode("utf-8"), headers=headers)
        if resp.status_code >= 400:
            fallback = f"{self._base_http}/ISAPI/Streaming/channels/{self.channel_id}/metadata/thermometry"
            resp = self._isapi_request(
                "PUT",
                fallback,
                data=payload.encode("utf-8"),
                headers=headers,
            )
        if resp.status_code in (401, 403, 404, 405):
            logging.warning(
                "Cannot enable thermometry via ISAPI (status=%s). Continuing with passive metadata read.",
                resp.status_code,
            )
            return
        resp.raise_for_status()

    def _subscribe_thermometry_or_default(self) -> str:
        url = f"{self._base_http}/ISAPI/Streaming/channels/{self.channel_id}/metadata/subscribeType?format=json"
        resp = self._isapi_request("POST", url, json={"type": ["thermometry"]})
        if resp.status_code in (400, 403, 404, 405):
            # Fallback for firmware variants without subscribeType API.
            candidates = self._default_metadata_rtsp_uris()
            uri = candidates[self._fallback_rtsp_index % len(candidates)]
            self._fallback_rtsp_index += 1
            logging.warning(
                "subscribeType unavailable/forbidden (status=%s). Fallback metadata RTSP: %s",
                resp.status_code,
                uri,
            )
            return uri
        resp.raise_for_status()
        body = resp.json()
        rtsp_uri = body.get("rtspURI")
        if not rtsp_uri:
            raise RuntimeError(f"Invalid subscribeType response: {body}")
        return rtsp_uri

    def _handle_auth_failure(self, message: str) -> float:
        self._consecutive_auth_failures += 1
        if self._consecutive_auth_failures >= self.max_auth_failures:
            logging.error("%s Entering cooldown for %.0f seconds to avoid camera lockout.", message, self.auth_lockout_sleep_sec)
            return self.auth_lockout_sleep_sec
        logging.error("%s", message)
        return max(20.0, self.reconnect_delay_sec)

    def _default_metadata_rtsp_uris(self) -> list[str]:
        user_q = quote(self.user, safe="")
        pass_q = quote(self.password, safe="")
        base = f"rtsp://{user_q}:{pass_q}@{self.ip}:{self.rtsp_port}"

        # Metadata API can use plain channel numbers (e.g., 2), while RTSP video uses 201/202.
        if self.channel_id >= 100:
            stream_ids = [self.channel_id]
        else:
            stream_ids = [self.channel_id * 100 + 1, self.channel_id * 100 + 2]

        uris = [f"{base}/Streaming/Channels/{sid}" for sid in stream_ids]
        uris.append(f"{base}/ISAPI/Streaming/channels/{self.channel_id}")
        return uris

    def _isapi_request(self, method: str, url: str, **kwargs):
        kwargs.setdefault("timeout", self.request_timeout_sec)
        resp = self._session.request(method, url, auth=self._digest_auth, **kwargs)
        if resp.status_code == 401:
            auth_header = resp.headers.get("WWW-Authenticate", "").lower()
            if "basic" in auth_header and "digest" not in auth_header:
                resp = self._session.request(method, url, auth=self._basic_auth, **kwargs)
        return resp

    def _read_metadata_stream(self, rtsp_uri: str) -> None:
        parsed = urlparse(rtsp_uri)
        host = parsed.hostname or self.ip
        port = parsed.port or self.rtsp_port

        cseq = 1
        with socket.create_connection((host, port), timeout=8) as sock:
            sock.settimeout(4)
            self._rtsp_digest = {}
            self._rtsp_nonce_counter = 0

            status, _, _ = self._rtsp_request(sock, "OPTIONS", rtsp_uri, cseq)
            self._ensure_rtsp_ok(status, "OPTIONS")
            cseq += 1

            status, _, body = self._rtsp_request(
                sock,
                "DESCRIBE",
                rtsp_uri,
                cseq,
                extra_headers={"Accept": "application/sdp"},
            )
            self._ensure_rtsp_ok(status, "DESCRIBE")
            sdp = body.decode("utf-8", errors="ignore")
            if "isapi.metadata" not in sdp.lower():
                raise RuntimeError(f"Metadata track not found in SDP for URI: {rtsp_uri}")
            cseq += 1

            setup_uri = self._resolve_track_uri(rtsp_uri, sdp)
            status, headers, _ = self._rtsp_request(
                sock,
                "SETUP",
                setup_uri,
                cseq,
                extra_headers={"Transport": "RTP/AVP/TCP;unicast;interleaved=0-1"},
            )
            self._ensure_rtsp_ok(status, "SETUP")
            session = headers.get("session", "").split(";")[0]
            cseq += 1

            extra = {"Session": session} if session else None
            status, _, _ = self._rtsp_request(sock, "PLAY", rtsp_uri, cseq, extra_headers=extra)
            self._ensure_rtsp_ok(status, "PLAY")

            buffer = b""
            idle_seconds = 0
            while not self._stop_event.is_set():
                try:
                    chunk = sock.recv(4096)
                except socket.timeout:
                    idle_seconds += 4
                    if idle_seconds >= 20:
                        logging.info("No metadata RTP packets yet on %s (waiting...)", rtsp_uri)
                        idle_seconds = 0
                    continue
                if not chunk:
                    raise RuntimeError("Metadata RTSP socket closed")
                idle_seconds = 0
                payload = self._extract_interleaved_rtp_payloads(chunk)
                if payload:
                    buffer += payload
                    buffer = self._consume_xml_messages(buffer)

    def _consume_xml_messages(self, buffer: bytes) -> bytes:
        while True:
            start = buffer.find(b"<?xml")
            if start < 0:
                if len(buffer) > 1024 * 1024:
                    return buffer[-65536:]
                return buffer

            end = buffer.find(b"</Metadata>", start)
            if end < 0:
                return buffer[start:]

            end += len(b"</Metadata>")
            xml_fragment = buffer[start:end]
            self._handle_xml_fragment(xml_fragment)
            buffer = buffer[end:]

    def _handle_xml_fragment(self, xml_fragment: bytes) -> None:
        readings = self._parse_thermometry(xml_fragment)
        if readings:
            with self._lock:
                self._latest_readings = readings

    def _append_raw_xml(self, xml_fragment: bytes) -> None:
        with self._raw_log_path.open("ab") as fp:
            fp.write(xml_fragment)
            fp.write(b"\n\n")

    def _parse_thermometry(self, xml_fragment: bytes) -> list[TempReading]:
        try:
            root = etree.fromstring(xml_fragment)
        except Exception:
            self._append_raw_xml(xml_fragment)
            logging.warning("failed to parse thermometry XML, raw snippet logged")
            return []

        sub_type = self._text_by_local_name(root, "subType")
        if sub_type.lower() != "thermometry":
            return []

        timestamp = self._text_by_local_name(root, "time")
        targets = self._children_by_local_name(root, "Target")

        out: list[TempReading] = []
        for target in targets:
            rule_id = self._text_by_local_name(target, "ruleID") or None
            region = self._parse_region_points(target)

            props = self._children_by_local_name(target, "Property")
            values: dict[str, str] = {}
            for prop in props:
                desc = self._text_by_local_name(prop, "description").strip()
                val = self._text_by_local_name(prop, "value").strip()
                if desc:
                    values[desc] = val

            if "tempValue" not in values:
                continue
            try:
                temp_value = float(values["tempValue"])
            except ValueError:
                continue

            temp_unit = values.get("tempUnit", "centigrade")
            temp_property = values.get("tempProperty", "")
            out.append(
                TempReading(
                    timestamp=timestamp,
                    sub_type=sub_type,
                    rule_id=rule_id,
                    temp_value=temp_value,
                    temp_unit=temp_unit,
                    temp_property=temp_property,
                    region=region,
                )
            )

        return out

    def _parse_region_points(self, target_node) -> list[RegionPoint]:
        points = self._children_by_local_name(target_node, "Point")
        result: list[RegionPoint] = []
        for point in points:
            x_txt = self._text_by_local_name(point, "x")
            y_txt = self._text_by_local_name(point, "y")
            try:
                result.append(RegionPoint(x=float(x_txt), y=float(y_txt)))
            except ValueError:
                continue
        return result

    def _children_by_local_name(self, node, local_name: str):
        return node.xpath(f".//*[local-name()='{local_name}']")

    def _text_by_local_name(self, node, local_name: str) -> str:
        items = node.xpath(f".//*[local-name()='{local_name}']")
        if not items:
            return ""
        value = items[0].text
        return value.strip() if value else ""

    def _resolve_track_uri(self, rtsp_uri: str, sdp: str) -> str:
        if "/trackID=" in rtsp_uri:
            return rtsp_uri

        parsed = urlparse(rtsp_uri)
        base = f"rtsp://{parsed.netloc}{parsed.path.rstrip('/')}"

        control_value = ""
        lines = [line.strip() for line in sdp.splitlines() if line.strip()]
        for idx, line in enumerate(lines):
            if line.startswith("a=rtpmap:") and "isapi.metadata" in line.lower():
                for j in range(idx + 1, min(idx + 8, len(lines))):
                    if lines[j].startswith("a=control:"):
                        control_value = lines[j].split("a=control:", 1)[1].strip()
                        break
                if control_value:
                    break

        if not control_value or control_value == "*":
            control_value = "trackID=3"

        if control_value.startswith("rtsp://"):
            return control_value
        if control_value.startswith("/"):
            return f"rtsp://{parsed.netloc}{control_value}"
        return f"{base}/{control_value}"

    def _rtsp_request(
        self,
        sock: socket.socket,
        method: str,
        uri: str,
        cseq: int,
        extra_headers: Optional[dict[str, str]] = None,
    ):
        headers = {
            "CSeq": str(cseq),
            "User-Agent": "hik-metadata-client/0.1",
        }
        if self._rtsp_digest:
            headers["Authorization"] = self._build_rtsp_digest_header(method, uri)
        if extra_headers:
            headers.update(extra_headers)

        lines = [f"{method} {uri} RTSP/1.0"] + [f"{k}: {v}" for k, v in headers.items()] + ["", ""]
        payload = "\r\n".join(lines).encode("utf-8")
        sock.sendall(payload)
        status, response_headers, body = self._read_rtsp_response(sock)

        if status == 401 and "www-authenticate" in response_headers:
            self._parse_rtsp_auth_challenge(response_headers["www-authenticate"])
            headers = {
                "CSeq": str(cseq),
                "User-Agent": "hik-metadata-client/0.1",
                "Authorization": self._build_rtsp_digest_header(method, uri),
            }
            if extra_headers:
                headers.update(extra_headers)
            lines = [f"{method} {uri} RTSP/1.0"] + [f"{k}: {v}" for k, v in headers.items()] + ["", ""]
            payload = "\r\n".join(lines).encode("utf-8")
            sock.sendall(payload)
            return self._read_rtsp_response(sock)

        return status, response_headers, body

    def _parse_rtsp_auth_challenge(self, header_value: str) -> None:
        challenge = header_value.strip()
        if challenge.lower().startswith("digest"):
            challenge = challenge[6:].strip()

        parsed_raw = parse_dict_header(challenge)
        parsed = {str(k).lower(): str(v) for k, v in parsed_raw.items()}
        if not parsed:
            raise RuntimeError(f"Unsupported RTSP auth header: {header_value}")

        self._rtsp_digest = {
            "realm": parsed.get("realm", ""),
            "nonce": parsed.get("nonce", ""),
            "opaque": parsed.get("opaque", ""),
            "qop": parsed.get("qop", ""),
            "algorithm": parsed.get("algorithm", "MD5"),
        }

    def _build_rtsp_digest_header(self, method: str, uri: str) -> str:
        realm = self._rtsp_digest.get("realm", "")
        nonce = self._rtsp_digest.get("nonce", "")
        opaque = self._rtsp_digest.get("opaque", "")
        qop_raw = self._rtsp_digest.get("qop", "")
        algorithm = self._rtsp_digest.get("algorithm", "MD5")
        request_uri = self._digest_request_uri(uri)
        method = method.upper()

        self._rtsp_nonce_counter += 1
        nc_value = f"{self._rtsp_nonce_counter:08x}"
        cnonce = secrets.token_hex(8)

        ha1 = hashlib.md5(f"{self.user}:{realm}:{self.password}".encode("utf-8")).hexdigest()
        if algorithm.upper() == "MD5-SESS":
            ha1 = hashlib.md5(f"{ha1}:{nonce}:{cnonce}".encode("utf-8")).hexdigest()

        ha2 = hashlib.md5(f"{method}:{request_uri}".encode("utf-8")).hexdigest()

        qop_token = ""
        if qop_raw:
            qop_parts = [part.strip() for part in qop_raw.split(",")]
            if "auth" in qop_parts:
                qop_token = "auth"

        if qop_token:
            response = hashlib.md5(
                f"{ha1}:{nonce}:{nc_value}:{cnonce}:{qop_token}:{ha2}".encode("utf-8")
            ).hexdigest()
        else:
            response = hashlib.md5(f"{ha1}:{nonce}:{ha2}".encode("utf-8")).hexdigest()

        parts = [
            'Digest username="%s"' % self.user,
            'realm="%s"' % realm,
            'nonce="%s"' % nonce,
            'uri="%s"' % request_uri,
            'response="%s"' % response,
        ]
        if algorithm:
            parts.append('algorithm="%s"' % algorithm)

        if opaque:
            parts.append('opaque="%s"' % opaque)
        if qop_token:
            parts.append('qop="%s"' % qop_token)
            parts.append("nc=%s" % nc_value)
            parts.append('cnonce="%s"' % cnonce)

        return ", ".join(parts)

    def _digest_request_uri(self, uri: str) -> str:
        parsed = urlparse(uri)
        path = parsed.path or "/"
        if parsed.query:
            path += f"?{parsed.query}"
        return path

    def _read_rtsp_response(self, sock: socket.socket):
        data = b""
        while True:
            while data.startswith(b"$"):
                while len(data) < 4:
                    part = sock.recv(4096)
                    if not part:
                        raise RuntimeError("RTSP connection closed")
                    data += part
                frame_len = int.from_bytes(data[2:4], "big")
                while len(data) < 4 + frame_len:
                    part = sock.recv(4096)
                    if not part:
                        raise RuntimeError("RTSP connection closed")
                    data += part
                data = data[4 + frame_len :]

            if b"\r\n\r\n" in data:
                break
            part = sock.recv(4096)
            if not part:
                raise RuntimeError("RTSP connection closed")
            data += part

        header_blob, rest = data.split(b"\r\n\r\n", 1)
        header_lines = header_blob.decode("utf-8", errors="ignore").split("\r\n")
        status_line = header_lines[0]
        try:
            status_code = int(status_line.split(" ")[1])
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Invalid RTSP status line: {status_line}") from exc

        headers: dict[str, str] = {}
        for line in header_lines[1:]:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            headers[key.strip().lower()] = value.strip()

        content_length = int(headers.get("content-length", "0") or "0")
        while len(rest) < content_length:
            rest += sock.recv(4096)

        body = rest[:content_length] if content_length > 0 else b""
        return status_code, headers, body

    def _ensure_rtsp_ok(self, status_code: int, method: str) -> None:
        if status_code < 200 or status_code >= 300:
            raise RuntimeError(f"RTSP {method} failed with status {status_code}")

    def _extract_interleaved_rtp_payloads(self, chunk: bytes) -> bytes:
        data = bytearray()
        i = 0
        n = len(chunk)
        while i < n:
            if chunk[i] != 0x24:  # '$' marks RTP over RTSP interleaved frame
                i += 1
                continue
            if i + 4 > n:
                break
            length = int.from_bytes(chunk[i + 2 : i + 4], "big")
            start = i + 4
            end = start + length
            if end > n:
                break

            rtp_packet = chunk[start:end]
            payload = self._extract_rtp_payload(rtp_packet)
            if payload:
                data.extend(payload)
            i = end
        return bytes(data)

    def _extract_rtp_payload(self, rtp_packet: bytes) -> bytes:
        if len(rtp_packet) < 12:
            return b""

        first = rtp_packet[0]
        csrc_count = first & 0x0F
        header_len = 12 + csrc_count * 4
        if len(rtp_packet) < header_len:
            return b""

        has_extension = bool(first & 0x10)
        if has_extension:
            if len(rtp_packet) < header_len + 4:
                return b""
            ext_len_words = int.from_bytes(rtp_packet[header_len + 2 : header_len + 4], "big")
            header_len += 4 + ext_len_words * 4
            if len(rtp_packet) < header_len:
                return b""

        payload = rtp_packet[header_len:]

        has_padding = bool(rtp_packet[0] & 0x20)
        if has_padding and payload:
            pad_len = payload[-1]
            if 0 < pad_len <= len(payload):
                payload = payload[:-pad_len]

        return payload
