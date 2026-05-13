from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rysuje porownanie przebiegu mean_temp_c dla dwoch sesji CSV."
        )
    )
    parser.add_argument("--normal-csv", required=True, help="Sciezka do sesji normalnej CSV.")
    parser.add_argument("--anomaly-csv", required=True, help="Sciezka do sesji anomalnej CSV.")
    parser.add_argument(
        "--output-dir",
        default="Wykresy",
        help="Folder wyjsciowy. Domyslnie: Wykresy",
    )
    parser.add_argument(
        "--output-name",
        default="mean_temp_session_comparison",
        help="Bazowa nazwa pliku bez rozszerzenia.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=30,
        help="Okno wygladzania w probkach. Domyslnie: 30",
    )
    parser.add_argument(
        "--match-shorter-duration",
        action="store_true",
        help="Przytnij obie sesje do czasu krotszej z nich.",
    )
    return parser.parse_args()


def _load_series(csv_path: Path) -> dict[str, Any]:
    elapsed: list[float] = []
    mean_temp: list[float] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            elapsed_text = (row.get("elapsed_sec") or "").strip().replace(",", ".")
            mean_text = (row.get("mean_temp_c") or "").strip().replace(",", ".")
            if not elapsed_text or not mean_text:
                continue
            elapsed.append(float(elapsed_text))
            mean_temp.append(float(mean_text))

    elapsed_arr = np.asarray(elapsed, dtype=np.float64)
    mean_arr = np.asarray(mean_temp, dtype=np.float64)
    return {
        "path": str(csv_path),
        "session_key": csv_path.stem,
        "elapsed_sec": elapsed_arr,
        "mean_temp_c": mean_arr,
        "mean": float(np.mean(mean_arr)) if mean_arr.size else None,
        "std": float(np.std(mean_arr)) if mean_arr.size else None,
        "min": float(np.min(mean_arr)) if mean_arr.size else None,
        "max": float(np.max(mean_arr)) if mean_arr.size else None,
        "count": int(mean_arr.size),
    }


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0:
        return values
    if window <= 1:
        return values.copy()
    kernel = np.ones(window, dtype=np.float64) / float(window)
    padded = np.pad(values, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: values.size]


def _truncate_to_duration(series: dict[str, Any], max_elapsed: float) -> dict[str, Any]:
    mask = series["elapsed_sec"] <= max_elapsed
    return {
        **series,
        "elapsed_sec": series["elapsed_sec"][mask],
        "mean_temp_c": series["mean_temp_c"][mask],
        "count": int(np.sum(mask)),
        "mean": float(np.mean(series["mean_temp_c"][mask])) if np.any(mask) else None,
        "std": float(np.std(series["mean_temp_c"][mask])) if np.any(mask) else None,
        "min": float(np.min(series["mean_temp_c"][mask])) if np.any(mask) else None,
        "max": float(np.max(series["mean_temp_c"][mask])) if np.any(mask) else None,
    }


def _save(fig: plt.Figure, png_path: Path, svg_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    normal = _load_series(Path(args.normal_csv))
    anomaly = _load_series(Path(args.anomaly_csv))

    if args.match_shorter_duration:
        max_elapsed = float(min(normal["elapsed_sec"][-1], anomaly["elapsed_sec"][-1]))
        normal = _truncate_to_duration(normal, max_elapsed)
        anomaly = _truncate_to_duration(anomaly, max_elapsed)

    normal_smooth = _rolling_mean(normal["mean_temp_c"], args.rolling_window)
    anomaly_smooth = _rolling_mean(anomaly["mean_temp_c"], args.rolling_window)

    fig, ax = plt.subplots(figsize=(11.5, 6.8), facecolor="#f7f7f5")
    ax.set_facecolor("#fcfcfb")

    ax.plot(
        normal["elapsed_sec"],
        normal["mean_temp_c"],
        color="#2563eb",
        alpha=0.18,
        linewidth=1.0,
    )
    ax.plot(
        anomaly["elapsed_sec"],
        anomaly["mean_temp_c"],
        color="#dc2626",
        alpha=0.18,
        linewidth=1.0,
    )
    ax.plot(
        normal["elapsed_sec"],
        normal_smooth,
        color="#1d4ed8",
        linewidth=2.8,
        label=f"Normalna: {normal['session_key']} (średnia={normal['mean']:.3f}°C)",
    )
    ax.plot(
        anomaly["elapsed_sec"],
        anomaly_smooth,
        color="#b91c1c",
        linewidth=2.8,
        label=f"Anomalna: {anomaly['session_key']} (średnia={anomaly['mean']:.3f}°C)",
    )
    ax.axhline(normal["mean"], color="#1d4ed8", linestyle="--", linewidth=1.6, alpha=0.8)
    ax.axhline(anomaly["mean"], color="#b91c1c", linestyle="--", linewidth=1.6, alpha=0.8)

    ax.set_title("Porównanie przebiegu mean_temp_c w sesji normalnej i anomalnej", fontsize=18, fontweight="bold")
    ax.set_xlabel("Czas od początku sesji [s]", fontsize=13)
    ax.set_ylabel("mean_temp_c [°C]", fontsize=13)
    ax.grid(True, alpha=0.18)
    ax.legend(frameon=True, facecolor="white", framealpha=0.92, loc="best")

    note = (
        f"Wygładzanie: {args.rolling_window} próbek | "
        f"Przycięcie do wspólnego czasu: {'tak' if args.match_shorter_duration else 'nie'}"
    )
    fig.text(0.01, 0.01, note, fontsize=10, color="#374151")

    png_path = output_dir / f"{args.output_name}.png"
    svg_path = output_dir / f"{args.output_name}.svg"
    _save(fig, png_path, svg_path)

    summary = {
        "normal": {
            "session_key": normal["session_key"],
            "path": normal["path"],
            "count": normal["count"],
            "mean": normal["mean"],
            "std": normal["std"],
            "min": normal["min"],
            "max": normal["max"],
        },
        "anomaly": {
            "session_key": anomaly["session_key"],
            "path": anomaly["path"],
            "count": anomaly["count"],
            "mean": anomaly["mean"],
            "std": anomaly["std"],
            "min": anomaly["min"],
            "max": anomaly["max"],
        },
        "rolling_window": args.rolling_window,
        "match_shorter_duration": args.match_shorter_duration,
        "output_png": str(png_path),
        "output_svg": str(svg_path),
    }

    summary_path = output_dir / f"{args.output_name}_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"Saved plot: {png_path}")
    print(f"Saved plot: {svg_path}")
    print(f"Saved summary: {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
