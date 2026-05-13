from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


TEMP_COLUMNS = ("mean_temp_c", "min_temp_c", "max_temp_c")
FEATURE_COLUMNS = (
    "mean_temp_c",
    "min_temp_c",
    "max_temp_c",
    "range_temp_c",
    "delta_mean_temp",
    "delta_max_temp",
    "slope_mean_5s",
    "slope_mean_15s",
    "rolling_std_mean_15s",
    "rolling_range_mean_15s",
    "sin_time",
    "cos_time",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Przygotowuje dane sekwencyjne pod LSTM Autoencoder z sesji "
            "pomiarowych zapisanych przez mp_forehead_ui."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="Pomiary",
        help="Folder z sesjami CSV. Domyslnie: Pomiary",
    )
    parser.add_argument(
        "--output-dir",
        default="Dane_przygotowane\\lstm_autoencoder",
        help=(
            "Folder wyjsciowy na przygotowane dane. "
            "Domyslnie: Dane_przygotowane\\lstm_autoencoder"
        ),
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=60,
        help="Dlugosc okna sekwencji. Domyslnie: 60",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=5,
        help="Krok przesuwania okna. Domyslnie: 5",
    )
    parser.add_argument(
        "--skip-initial-sec",
        type=float,
        default=10.0,
        help="Ile sekund od poczatku kazdej sesji pominac. Domyslnie: 10",
    )
    parser.add_argument(
        "--max-missing-ratio",
        type=float,
        default=0.10,
        help=(
            "Maksymalny dopuszczalny udzial brakujacych probek temperatury "
            "w oknie przed interpolacja. Domyslnie: 0.10"
        ),
    )
    parser.add_argument(
        "--exclude-session-key",
        nargs="*",
        default=[],
        help=(
            "Lista session_key do wykluczenia z datasetu, np. "
            "M1_normalny_22.03.2026_12.23.25"
        ),
    )
    return parser.parse_args()


def _parse_float(value: str | None) -> float:
    if value is None:
        return float("nan")
    text = value.strip().replace(",", ".")
    if not text:
        return float("nan")
    return float(text)


def _safe_slug(value: str, fallback: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value.strip())
    cleaned = cleaned.strip("_")
    return cleaned or fallback


def _parse_timestamp(date_text: str, time_text: str) -> datetime:
    return datetime.strptime(f"{date_text} {time_text}", "%d.%m.%Y %H:%M:%S")


def _median_step_seconds(elapsed_sec: np.ndarray) -> float:
    if elapsed_sec.size < 2:
        return 0.0
    diffs = np.diff(elapsed_sec)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if diffs.size == 0:
        return 0.0
    return float(np.median(diffs))


def _interpolate_column(values: np.ndarray) -> tuple[np.ndarray | None, np.ndarray]:
    missing_mask = ~np.isfinite(values)
    if missing_mask.all():
        return None, missing_mask

    if not missing_mask.any():
        return values.copy(), missing_mask

    x = np.arange(values.size, dtype=np.float64)
    valid = ~missing_mask
    filled = values.copy()
    filled[missing_mask] = np.interp(x[missing_mask], x[valid], values[valid])
    return filled, missing_mask


def _compute_delta(values: np.ndarray) -> np.ndarray:
    return np.diff(values, prepend=values[0]).astype(np.float32)


def _compute_lag_slope(values: np.ndarray, lag: int) -> np.ndarray:
    slopes = np.zeros(values.shape[0], dtype=np.float32)
    if values.size <= 1:
        return slopes

    for idx in range(1, values.size):
        steps = min(idx, lag)
        slopes[idx] = float((values[idx] - values[idx - steps]) / steps)
    return slopes


def _compute_rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    rolling = np.zeros(values.shape[0], dtype=np.float32)
    for idx in range(values.size):
        start = max(0, idx - window + 1)
        window_values = values[start : idx + 1]
        rolling[idx] = float(np.std(window_values)) if window_values.size > 1 else 0.0
    return rolling


def _compute_rolling_range(values: np.ndarray, window: int) -> np.ndarray:
    rolling = np.zeros(values.shape[0], dtype=np.float32)
    for idx in range(values.size):
        start = max(0, idx - window + 1)
        window_values = values[start : idx + 1]
        rolling[idx] = float(window_values.max() - window_values.min()) if window_values.size > 0 else 0.0
    return rolling


def _load_session(
    csv_path: Path,
    skip_initial_sec: float,
    excluded_session_keys: set[str],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            timestamp = _parse_timestamp(raw_row["date"], raw_row["time"])
            rows.append(
                {
                    "sample_index": int(raw_row["sample_index"]),
                    "date": raw_row["date"],
                    "time": raw_row["time"],
                    "elapsed_sec": _parse_float(raw_row["elapsed_sec"]),
                    "subject_id": raw_row.get("subject_id", "").strip() or csv_path.parent.name,
                    "session_state": raw_row.get("session_state", "").strip() or "unknown",
                    "timestamp": timestamp,
                    "mean_temp_c": _parse_float(raw_row.get("mean_temp_c")),
                    "min_temp_c": _parse_float(raw_row.get("min_temp_c")),
                    "max_temp_c": _parse_float(raw_row.get("max_temp_c")),
                }
            )

    if not rows:
        raise ValueError(f"Plik {csv_path} nie zawiera zadnych wierszy danych.")

    subject_id = rows[0]["subject_id"]
    session_state = rows[0]["session_state"]
    session_key = csv_path.stem

    elapsed_all = np.asarray([row["elapsed_sec"] for row in rows], dtype=np.float64)
    valid_all_mask = np.isfinite(
        np.asarray(
            [[row[column] for column in TEMP_COLUMNS] for row in rows],
            dtype=np.float64,
        )
    ).all(axis=1)

    if session_key in excluded_session_keys:
        return {
            "source_file": str(csv_path),
            "session_key": session_key,
            "subject_id": subject_id,
            "session_state": session_state,
            "start_timestamp": rows[0]["timestamp"],
            "end_timestamp": rows[-1]["timestamp"],
            "num_rows_total": len(rows),
            "num_rows_after_skip": len(rows),
            "num_valid_rows_total": int(valid_all_mask.sum()),
            "num_valid_rows_after_skip": int(valid_all_mask.sum()),
            "sample_period_sec": _median_step_seconds(elapsed_all),
            "prepared_rows": [],
            "feature_matrix": None,
            "row_missing_mask": None,
            "sample_index": None,
            "elapsed_sec": None,
            "timestamps": None,
            "skipped_reason": "excluded_by_user",
        }

    filtered_rows = [row for row in rows if row["elapsed_sec"] >= skip_initial_sec]
    if not filtered_rows:
        return {
            "source_file": str(csv_path),
            "session_key": session_key,
            "subject_id": subject_id,
            "session_state": session_state,
            "start_timestamp": rows[0]["timestamp"],
            "end_timestamp": rows[-1]["timestamp"],
            "num_rows_total": len(rows),
            "num_rows_after_skip": 0,
            "num_valid_rows_total": int(valid_all_mask.sum()),
            "num_valid_rows_after_skip": 0,
            "sample_period_sec": _median_step_seconds(elapsed_all),
            "prepared_rows": [],
            "feature_matrix": None,
            "row_missing_mask": None,
            "sample_index": None,
            "elapsed_sec": None,
            "timestamps": None,
            "skipped_reason": "all_rows_removed_by_skip_initial_sec",
        }

    temp_matrix = np.asarray(
        [[row[column] for column in TEMP_COLUMNS] for row in filtered_rows],
        dtype=np.float64,
    )
    row_missing_mask = ~np.isfinite(temp_matrix).all(axis=1)

    filled_columns: list[np.ndarray] = []
    for column_idx in range(temp_matrix.shape[1]):
        filled, _ = _interpolate_column(temp_matrix[:, column_idx])
        if filled is None:
            return {
                "source_file": str(csv_path),
                "session_key": session_key,
                "subject_id": subject_id,
                "session_state": session_state,
                "start_timestamp": rows[0]["timestamp"],
                "end_timestamp": rows[-1]["timestamp"],
                "num_rows_total": len(rows),
                "num_rows_after_skip": len(filtered_rows),
                "num_valid_rows_total": int(valid_all_mask.sum()),
                "num_valid_rows_after_skip": int((~row_missing_mask).sum()),
                "sample_period_sec": _median_step_seconds(elapsed_all),
                "prepared_rows": [],
                "feature_matrix": None,
                "row_missing_mask": None,
                "sample_index": None,
                "elapsed_sec": None,
                "timestamps": None,
                "skipped_reason": f"no_valid_values_for_{TEMP_COLUMNS[column_idx]}",
            }
        filled_columns.append(filled)

    mean_temp, min_temp, max_temp = filled_columns
    range_temp = max_temp - min_temp
    delta_mean_temp = _compute_delta(mean_temp)
    delta_max_temp = _compute_delta(max_temp)
    slope_mean_5s = _compute_lag_slope(mean_temp, lag=5)
    slope_mean_15s = _compute_lag_slope(mean_temp, lag=15)
    rolling_std_mean_15s = _compute_rolling_std(mean_temp, window=15)
    rolling_range_mean_15s = _compute_rolling_range(mean_temp, window=15)

    timestamps = [row["timestamp"] for row in filtered_rows]
    seconds_of_day = np.asarray(
        [ts.hour * 3600 + ts.minute * 60 + ts.second for ts in timestamps],
        dtype=np.float64,
    )
    sin_time = np.sin((2.0 * math.pi * seconds_of_day) / 86400.0)
    cos_time = np.cos((2.0 * math.pi * seconds_of_day) / 86400.0)

    feature_matrix = np.column_stack(
        [
            mean_temp,
            min_temp,
            max_temp,
            range_temp,
            delta_mean_temp,
            delta_max_temp,
            slope_mean_5s,
            slope_mean_15s,
            rolling_std_mean_15s,
            rolling_range_mean_15s,
            sin_time,
            cos_time,
        ]
    ).astype(np.float32)

    prepared_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(filtered_rows):
        prepared_rows.append(
            {
                "source_file": str(csv_path),
                "session_key": session_key,
                "subject_id": subject_id,
                "session_state": session_state,
                "sample_index": row["sample_index"],
                "elapsed_sec": f"{row['elapsed_sec']:.3f}",
                "timestamp_iso": row["timestamp"].isoformat(),
                "mean_temp_c": f"{mean_temp[idx]:.3f}",
                "min_temp_c": f"{min_temp[idx]:.3f}",
                "max_temp_c": f"{max_temp[idx]:.3f}",
                "range_temp_c": f"{range_temp[idx]:.3f}",
                "delta_mean_temp": f"{delta_mean_temp[idx]:.6f}",
                "delta_max_temp": f"{delta_max_temp[idx]:.6f}",
                "slope_mean_5s": f"{slope_mean_5s[idx]:.6f}",
                "slope_mean_15s": f"{slope_mean_15s[idx]:.6f}",
                "rolling_std_mean_15s": f"{rolling_std_mean_15s[idx]:.6f}",
                "rolling_range_mean_15s": f"{rolling_range_mean_15s[idx]:.6f}",
                "sin_time": f"{sin_time[idx]:.8f}",
                "cos_time": f"{cos_time[idx]:.8f}",
                "row_had_missing_temp": "true" if row_missing_mask[idx] else "false",
            }
        )

    return {
        "source_file": str(csv_path),
        "session_key": session_key,
        "subject_id": subject_id,
        "session_state": session_state,
        "start_timestamp": rows[0]["timestamp"],
        "end_timestamp": rows[-1]["timestamp"],
        "num_rows_total": len(rows),
        "num_rows_after_skip": len(filtered_rows),
        "num_valid_rows_total": int(valid_all_mask.sum()),
        "num_valid_rows_after_skip": int((~row_missing_mask).sum()),
        "sample_period_sec": _median_step_seconds(elapsed_all),
        "prepared_rows": prepared_rows,
        "feature_matrix": feature_matrix,
        "row_missing_mask": row_missing_mask,
        "sample_index": np.asarray([row["sample_index"] for row in filtered_rows], dtype=np.int32),
        "elapsed_sec": np.asarray([row["elapsed_sec"] for row in filtered_rows], dtype=np.float32),
        "timestamps": timestamps,
        "skipped_reason": "",
    }


def _assign_subject_splits(sessions: list[dict[str, Any]]) -> None:
    usable_sessions = [session for session in sessions if not session["skipped_reason"]]
    normal_sessions = sorted(
        [session for session in usable_sessions if session["session_state"] == "normalny"],
        key=lambda item: item["start_timestamp"],
    )
    anomaly_sessions = sorted(
        [session for session in usable_sessions if session["session_state"] != "normalny"],
        key=lambda item: item["start_timestamp"],
    )

    for session in sessions:
        session["split"] = "excluded" if session["skipped_reason"] else "unassigned"

    total_normal = len(normal_sessions)
    if total_normal >= 6:
        train_sessions = normal_sessions[:-4]
        val_sessions = normal_sessions[-4:-2]
        test_sessions = normal_sessions[-2:]
    elif total_normal == 5:
        train_sessions = normal_sessions[:3]
        val_sessions = normal_sessions[3:4]
        test_sessions = normal_sessions[4:]
    elif total_normal == 4:
        train_sessions = normal_sessions[:2]
        val_sessions = normal_sessions[2:3]
        test_sessions = normal_sessions[3:]
    elif total_normal == 3:
        train_sessions = normal_sessions[:1]
        val_sessions = normal_sessions[1:2]
        test_sessions = normal_sessions[2:]
    elif total_normal == 2:
        train_sessions = normal_sessions[:1]
        val_sessions = []
        test_sessions = normal_sessions[1:]
    elif total_normal == 1:
        train_sessions = normal_sessions[:1]
        val_sessions = []
        test_sessions = []
    else:
        train_sessions = []
        val_sessions = []
        test_sessions = []

    for session in train_sessions:
        session["split"] = "train"
    for session in val_sessions:
        session["split"] = "val"
    for session in test_sessions:
        session["split"] = "test_normal"
    for session in anomaly_sessions:
        session["split"] = "test_anomaly"


def _build_subject_windows(
    sessions: list[dict[str, Any]],
    seq_len: int,
    stride: int,
    max_missing_ratio: float,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], list[dict[str, Any]]]:
    split_names = ("train", "val", "test_normal", "test_anomaly")
    raw_windows: dict[str, list[np.ndarray]] = {name: [] for name in split_names}
    prepared_rows: list[dict[str, Any]] = []
    window_index_rows: list[dict[str, Any]] = []

    for session in sorted(sessions, key=lambda item: item["start_timestamp"]):
        split = session["split"]
        for row in session["prepared_rows"]:
            prepared_rows.append({"split": split, **row})

        if split not in raw_windows:
            continue

        feature_matrix = session["feature_matrix"]
        if feature_matrix is None or feature_matrix.shape[0] < seq_len:
            continue

        sample_index = session["sample_index"]
        elapsed_sec = session["elapsed_sec"]
        timestamps = session["timestamps"]
        row_missing_mask = session["row_missing_mask"]

        for start in range(0, feature_matrix.shape[0] - seq_len + 1, stride):
            end = start + seq_len
            missing_ratio = float(row_missing_mask[start:end].mean())
            if missing_ratio > max_missing_ratio:
                continue

            array_index = len(raw_windows[split])
            raw_windows[split].append(feature_matrix[start:end])
            window_index_rows.append(
                {
                    "split": split,
                    "array_name": f"X_{split}",
                    "array_index": array_index,
                    "session_key": session["session_key"],
                    "source_file": session["source_file"],
                    "subject_id": session["subject_id"],
                    "session_state": session["session_state"],
                    "start_sample_index": int(sample_index[start]),
                    "end_sample_index": int(sample_index[end - 1]),
                    "start_elapsed_sec": f"{float(elapsed_sec[start]):.3f}",
                    "end_elapsed_sec": f"{float(elapsed_sec[end - 1]):.3f}",
                    "start_timestamp": timestamps[start].isoformat(),
                    "end_timestamp": timestamps[end - 1].isoformat(),
                    "missing_ratio_before_fill": f"{missing_ratio:.4f}",
                    "num_rows": seq_len,
                }
            )

    shaped_raw_windows: dict[str, np.ndarray] = {}
    feature_count = len(FEATURE_COLUMNS)
    for split, items in raw_windows.items():
        if items:
            shaped_raw_windows[split] = np.stack(items).astype(np.float32)
        else:
            shaped_raw_windows[split] = np.zeros((0, seq_len, feature_count), dtype=np.float32)
    return shaped_raw_windows, prepared_rows, window_index_rows


def _write_csv(rows: list[dict[str, Any]], target_path: Path, fieldnames: list[str]) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_subject_dataset(
    subject_id: str,
    sessions: list[dict[str, Any]],
    output_dir: Path,
    seq_len: int,
    stride: int,
    skip_initial_sec: float,
    max_missing_ratio: float,
) -> dict[str, Any]:
    _assign_subject_splits(sessions)
    raw_windows, prepared_rows, window_index_rows = _build_subject_windows(
        sessions=sessions,
        seq_len=seq_len,
        stride=stride,
        max_missing_ratio=max_missing_ratio,
    )

    train_raw = raw_windows["train"]
    if train_raw.shape[0] == 0:
        raise ValueError(
            f"Brak okien treningowych dla subject_id={subject_id}. "
            "Potrzebne sa co najmniej jedne dane normalne po preprocessingu."
        )

    flattened_train = train_raw.reshape(-1, train_raw.shape[-1])
    feature_mean = flattened_train.mean(axis=0)
    feature_std = flattened_train.std(axis=0)
    feature_std[feature_std < 1e-6] = 1.0

    normalized_windows: dict[str, np.ndarray] = {}
    for split, raw_array in raw_windows.items():
        normalized_windows[split] = ((raw_array - feature_mean) / feature_std).astype(np.float32)

    subject_dir = output_dir / _safe_slug(subject_id, "unknown_subject")
    subject_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        subject_dir / "dataset.npz",
        feature_names=np.asarray(FEATURE_COLUMNS),
        X_train=normalized_windows["train"],
        X_val=normalized_windows["val"],
        X_test_normal=normalized_windows["test_normal"],
        X_test_anomaly=normalized_windows["test_anomaly"],
        X_train_raw=raw_windows["train"],
        X_val_raw=raw_windows["val"],
        X_test_normal_raw=raw_windows["test_normal"],
        X_test_anomaly_raw=raw_windows["test_anomaly"],
    )

    normalization_payload = {
        "subject_id": subject_id,
        "feature_names": list(FEATURE_COLUMNS),
        "feature_mean": {
            feature_name: round(float(feature_mean[idx]), 8)
            for idx, feature_name in enumerate(FEATURE_COLUMNS)
        },
        "feature_std": {
            feature_name: round(float(feature_std[idx]), 8)
            for idx, feature_name in enumerate(FEATURE_COLUMNS)
        },
        "config": {
            "seq_len": seq_len,
            "stride": stride,
            "skip_initial_sec": skip_initial_sec,
            "max_missing_ratio": max_missing_ratio,
        },
    }
    (subject_dir / "normalization.json").write_text(
        json.dumps(normalization_payload, indent=2),
        encoding="utf-8",
    )

    session_summary_rows: list[dict[str, Any]] = []
    for session in sorted(sessions, key=lambda item: item["start_timestamp"]):
        session_summary_rows.append(
            {
                "subject_id": session["subject_id"],
                "session_key": session["session_key"],
                "session_state": session["session_state"],
                "split": session["split"],
                "source_file": session["source_file"],
                "start_timestamp": session["start_timestamp"].isoformat(),
                "end_timestamp": session["end_timestamp"].isoformat(),
                "num_rows_total": session["num_rows_total"],
                "num_rows_after_skip": session["num_rows_after_skip"],
                "num_valid_rows_total": session["num_valid_rows_total"],
                "num_valid_rows_after_skip": session["num_valid_rows_after_skip"],
                "sample_period_sec": f"{session['sample_period_sec']:.3f}",
                "skipped_reason": session["skipped_reason"],
            }
        )

    _write_csv(
        session_summary_rows,
        subject_dir / "session_summary.csv",
        [
            "subject_id",
            "session_key",
            "session_state",
            "split",
            "source_file",
            "start_timestamp",
            "end_timestamp",
            "num_rows_total",
            "num_rows_after_skip",
            "num_valid_rows_total",
            "num_valid_rows_after_skip",
            "sample_period_sec",
            "skipped_reason",
        ],
    )
    _write_csv(
        prepared_rows,
        subject_dir / "prepared_rows.csv",
        [
            "split",
            "source_file",
            "session_key",
            "subject_id",
            "session_state",
            "sample_index",
            "elapsed_sec",
            "timestamp_iso",
            "mean_temp_c",
            "min_temp_c",
            "max_temp_c",
            "range_temp_c",
            "delta_mean_temp",
            "delta_max_temp",
            "slope_mean_5s",
            "slope_mean_15s",
            "rolling_std_mean_15s",
            "rolling_range_mean_15s",
            "sin_time",
            "cos_time",
            "row_had_missing_temp",
        ],
    )
    _write_csv(
        window_index_rows,
        subject_dir / "window_index.csv",
        [
            "split",
            "array_name",
            "array_index",
            "session_key",
            "source_file",
            "subject_id",
            "session_state",
            "start_sample_index",
            "end_sample_index",
            "start_elapsed_sec",
            "end_elapsed_sec",
            "start_timestamp",
            "end_timestamp",
            "missing_ratio_before_fill",
            "num_rows",
        ],
    )

    split_summary = {
        "train": int(normalized_windows["train"].shape[0]),
        "val": int(normalized_windows["val"].shape[0]),
        "test_normal": int(normalized_windows["test_normal"].shape[0]),
        "test_anomaly": int(normalized_windows["test_anomaly"].shape[0]),
    }
    dataset_summary = {
        "subject_id": subject_id,
        "feature_names": list(FEATURE_COLUMNS),
        "config": {
            "seq_len": seq_len,
            "stride": stride,
            "skip_initial_sec": skip_initial_sec,
            "max_missing_ratio": max_missing_ratio,
        },
        "num_sessions": len(sessions),
        "num_prepared_rows": len(prepared_rows),
        "num_windows": split_summary,
        "note": (
            "Okna sa znormalizowane statystykami policzonymi tylko na zbiorze train. "
            "Surowe wersje okien znajduja sie w dataset.npz jako *_raw."
        ),
    }
    (subject_dir / "dataset_summary.json").write_text(
        json.dumps(dataset_summary, indent=2),
        encoding="utf-8",
    )

    readme_text = (
        "# Prepared LSTM dataset\n\n"
        "Ten folder zawiera dane przygotowane pod model sekwencyjny dla jednego pacjenta.\n\n"
        "Pliki:\n"
        "- dataset.npz: okna sekwencji gotowe do treningu i ewaluacji.\n"
        "- normalization.json: srednie i odchylenia uzyte do normalizacji.\n"
        "- session_summary.csv: podsumowanie sesji i przydzialu do splitow.\n"
        "- prepared_rows.csv: dane po interpolacji i engineeringu cech.\n"
        "- window_index.csv: mapowanie okien do sesji i timestampow.\n"
        "- dataset_summary.json: skrotowe statystyki zbioru.\n"
    )
    (subject_dir / "README.md").write_text(readme_text, encoding="utf-8")

    return {
        "subject_id": subject_id,
        "output_dir": str(subject_dir),
        "num_sessions": len(sessions),
        "num_prepared_rows": len(prepared_rows),
        "num_windows": split_summary,
    }


def main() -> None:
    args = _parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    excluded_session_keys = {item.strip() for item in args.exclude_session_key if item.strip()}

    if not input_dir.exists():
        raise FileNotFoundError(f"Nie znaleziono folderu wejsciowego: {input_dir}")

    csv_files = sorted(input_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Nie znaleziono plikow CSV w folderze: {input_dir}")

    sessions_by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for csv_path in csv_files:
        session = _load_session(
            csv_path,
            skip_initial_sec=args.skip_initial_sec,
            excluded_session_keys=excluded_session_keys,
        )
        sessions_by_subject[session["subject_id"]].append(session)

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_overview: list[dict[str, Any]] = []
    for subject_id, sessions in sorted(sessions_by_subject.items()):
        dataset_overview.append(
            _write_subject_dataset(
                subject_id=subject_id,
                sessions=sessions,
                output_dir=output_dir,
                seq_len=args.seq_len,
                stride=args.stride,
                skip_initial_sec=args.skip_initial_sec,
                max_missing_ratio=args.max_missing_ratio,
            )
        )

    overview_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "excluded_session_keys": sorted(excluded_session_keys),
        "subjects": dataset_overview,
    }
    (output_dir / "dataset_overview.json").write_text(
        json.dumps(overview_payload, indent=2),
        encoding="utf-8",
    )

    print(f"Prepared dataset saved to: {output_dir}")
    print(json.dumps(overview_payload, indent=2))


if __name__ == "__main__":
    main()
