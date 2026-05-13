from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.train_lstm_autoencoder import LSTMAutoencoder


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Buduje figure z trzema przebiegami anomaly score: sesja anomalna M1, "
            "sesja K1 liczona modelem M1 oraz sesja normalna M1."
        )
    )
    parser.add_argument("--run-dir", required=True, help="Folder runu M1.")
    parser.add_argument("--m1-anomaly-session", required=True, help="session_key sesji anomalnej M1.")
    parser.add_argument("--m1-normal-session", required=True, help="session_key sesji normalnej M1.")
    parser.add_argument("--k1-dataset-dir", required=True, help="Folder datasetu K1 zgodnego z runem.")
    parser.add_argument("--k1-session-key", required=True, help="session_key sesji K1 do narysowania.")
    parser.add_argument("--output-dir", default="Wykresy", help="Folder wyjsciowy.")
    parser.add_argument(
        "--output-name",
        default="three_session_timeline_comparison",
        help="Bazowa nazwa pliku bez rozszerzenia.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_run_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with (run_dir / "scores_by_window.csv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            start_dt = datetime.fromisoformat(row["start_timestamp"])
            rows.append(
                {
                    "session_key": row["session_key"],
                    "split": row["split"],
                    "session_state": row["session_state"],
                    "start_dt": start_dt,
                    "array_index": int(row["array_index"]),
                    "reconstruction_error": float(row["reconstruction_error"]),
                    "is_anomaly_bool": row["is_anomaly"].strip().lower() == "true",
                }
            )
    return rows


def _load_model(run_dir: Path, metrics: dict[str, Any]) -> LSTMAutoencoder:
    cfg = metrics["training_config"]
    model = LSTMAutoencoder(
        input_size=len(metrics["feature_names"]),
        hidden_size=cfg["hidden_size"],
        latent_size=cfg["latent_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    )
    model.load_state_dict(torch.load(run_dir / "model_best.pt", map_location="cpu"))
    model.eval()
    return model


def _score_k1_session(
    run_dir: Path,
    metrics: dict[str, Any],
    k1_dataset_dir: Path,
    session_key: str,
) -> list[dict[str, Any]]:
    m1_dataset_dir = Path(metrics["dataset_dir"])
    normalization = _load_json(m1_dataset_dir / "normalization.json")
    feature_names = metrics["feature_names"]
    mean = np.asarray([normalization["feature_mean"][f] for f in feature_names], dtype=np.float32)
    std = np.asarray([normalization["feature_std"][f] for f in feature_names], dtype=np.float32)
    std = np.where(np.abs(std) < 1e-8, 1.0, std)
    threshold = float(metrics["threshold"]["reconstruction_error_threshold"])

    with (k1_dataset_dir / "window_index.csv").open("r", encoding="utf-8", newline="") as handle:
        index_rows = list(csv.DictReader(handle))
    selected_rows = [row for row in index_rows if row["session_key"] == session_key]
    if not selected_rows:
        raise SystemExit(f"Nie znaleziono session_key={session_key} w {k1_dataset_dir / 'window_index.csv'}")
    data = np.load(k1_dataset_dir / "dataset.npz", allow_pickle=True)

    model = _load_model(run_dir, metrics)
    batch_size = int(metrics["training_config"]["batch_size"])
    selected_raw_arrays: list[np.ndarray] = []
    for row in selected_rows:
        array_name = row["array_name"]
        array_index = int(row["array_index"])
        raw_array = data[f"{array_name}_raw"][array_index].astype(np.float32)
        selected_raw_arrays.append(raw_array)
    selected_arrays_raw = np.stack(selected_raw_arrays).astype(np.float32)
    selected_arrays = ((selected_arrays_raw - mean[None, None, :]) / std[None, None, :]).astype(np.float32)
    errors: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, selected_arrays.shape[0], batch_size):
            batch_np = selected_arrays[start : start + batch_size]
            batch = torch.from_numpy(batch_np)
            reconstruction = model(batch)
            batch_errors = torch.mean((reconstruction - batch) ** 2, dim=(1, 2)).cpu().numpy()
            errors.append(batch_errors)
    error_values = np.concatenate(errors, axis=0)

    output_rows: list[dict[str, Any]] = []
    for idx, (row, error) in enumerate(zip(selected_rows, error_values)):
        output_rows.append(
            {
                "session_key": row["session_key"],
                "split": "other_subject",
                "session_state": row["session_state"],
                "start_dt": datetime.fromisoformat(row["start_timestamp"]),
                "array_index": idx,
                "reconstruction_error": float(error),
                "is_anomaly_bool": float(error) > threshold,
            }
        )
    return output_rows


def _smooth(values: np.ndarray) -> np.ndarray:
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64)
    kernel /= kernel.sum()
    return np.convolve(values, kernel, mode="same")


def _plot_axis(
    ax: plt.Axes,
    session_rows: list[dict[str, Any]],
    threshold: float,
    title: str,
    smooth_color: str,
) -> dict[str, Any]:
    session_rows = sorted(session_rows, key=lambda item: item["array_index"])
    first_dt = session_rows[0]["start_dt"]
    elapsed_min = np.asarray(
        [(row["start_dt"] - first_dt).total_seconds() / 60.0 for row in session_rows],
        dtype=np.float32,
    )
    errors = np.asarray([row["reconstruction_error"] for row in session_rows], dtype=np.float32)
    anomaly_mask = np.asarray([row["is_anomaly_bool"] for row in session_rows], dtype=bool)
    smooth_errors = _smooth(errors)

    ax.plot(elapsed_min, errors, color="#94a3b8", linewidth=1.2, alpha=0.9)
    ax.plot(elapsed_min, smooth_errors, color=smooth_color, linewidth=3.0)
    ax.axhline(threshold, color="#111827", linestyle=":", linewidth=2.2)
    if anomaly_mask.any():
        ax.scatter(
            elapsed_min[anomaly_mask],
            errors[anomaly_mask],
            color="#dc2626",
            s=34,
            alpha=0.95,
            zorder=5,
        )
        ax.fill_between(
            elapsed_min,
            threshold,
            errors,
            where=errors > threshold,
            color="#fecaca",
            alpha=0.45,
            interpolate=True,
        )
    ax.set_ylabel("Error")
    ax.set_title(title, fontsize=18)
    ax.grid(True, alpha=0.18)
    ax.set_facecolor("#fcfcfb")

    return {
        "session_key": session_rows[0]["session_key"],
        "mean_error": float(errors.mean()),
        "p95_error": float(np.quantile(errors, 0.95)),
        "above_threshold_ratio": float(anomaly_mask.mean()),
        "duration_min": float(elapsed_min[-1]) if elapsed_min.size else 0.0,
    }


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = _load_json(run_dir / "metrics.json")
    threshold = float(metrics["threshold"]["reconstruction_error_threshold"])
    run_rows = _load_run_rows(run_dir)

    m1_anomaly_rows = [row for row in run_rows if row["session_key"] == args.m1_anomaly_session]
    m1_normal_rows = [row for row in run_rows if row["session_key"] == args.m1_normal_session]
    if not m1_anomaly_rows:
        raise SystemExit(f"Nie znaleziono sesji {args.m1_anomaly_session} w scores_by_window.csv")
    if not m1_normal_rows:
        raise SystemExit(f"Nie znaleziono sesji {args.m1_normal_session} w scores_by_window.csv")

    k1_rows = _score_k1_session(
        run_dir=run_dir,
        metrics=metrics,
        k1_dataset_dir=Path(args.k1_dataset_dir),
        session_key=args.k1_session_key,
    )

    fig, axes = plt.subplots(3, 1, figsize=(16, 15), facecolor="#f7f7f5")

    summaries = {
        "m1_anomaly": _plot_axis(
            axes[0],
            m1_anomaly_rows,
            threshold,
            f"{args.m1_anomaly_session} (test_anomaly)",
            "#dc2626",
        ),
        "k1_other": _plot_axis(
            axes[1],
            k1_rows,
            threshold,
            f"{args.k1_session_key} (other_subject)",
            "#b45309",
        ),
        "m1_normal": _plot_axis(
            axes[2],
            m1_normal_rows,
            threshold,
            f"{args.m1_normal_session} (test_normal)",
            "#0f766e",
        ),
    }

    axes[2].set_xlabel("Czas od początku sesji [min]")
    fig.suptitle("Porównanie przebiegu anomaly score dla sesji M1 i K1", fontsize=24, fontweight="bold", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    png_path = output_dir / f"{args.output_name}.png"
    svg_path = output_dir / f"{args.output_name}.svg"
    fig.savefig(png_path, dpi=240, facecolor=fig.get_facecolor())
    fig.savefig(svg_path, facecolor=fig.get_facecolor())
    plt.close(fig)

    summary = {
        "run_dir": str(run_dir),
        "threshold": threshold,
        "sessions": summaries,
        "output_png": str(png_path),
        "output_svg": str(svg_path),
    }
    summary_path = output_dir / f"{args.output_name}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved plot: {png_path}")
    print(f"Saved plot: {svg_path}")
    print(f"Saved summary: {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
