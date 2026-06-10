from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Brak pakietu 'torch'. Zainstaluj zaleznosci ML poleceniem: "
        "pip install -r requirements-ml.txt"
    ) from exc

from src.train_lstm_autoencoder import LSTMAutoencoder, _compute_reconstruction_errors


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scores a selected session with an already trained LSTM Autoencoder."
    )
    parser.add_argument("--run-dir", required=True, help="Folder runu z model_best.pt i metrics.json.")
    parser.add_argument("--dataset-dir", required=True, help="Folder datasetu z dataset.npz i window_index.csv.")
    parser.add_argument("--session-key", required=True, help="Session key do oceny.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Folder wyjsciowy. Domyslnie: <run-dir>\\figures\\session_scores\\<session-key>",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size do scoringu.")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Urzadzenie do scoringu. Domyslnie: auto.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Wybrano CUDA, ale torch nie widzi GPU.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_window_index(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _session_windows(
    dataset: Any,
    window_rows: list[dict[str, str]],
    session_key: str,
) -> tuple[np.ndarray, list[dict[str, str]]]:
    arrays = {
        "train": dataset["X_train"],
        "val": dataset["X_val"],
        "test_normal": dataset["X_test_normal"],
        "test_anomaly": dataset["X_test_anomaly"],
    }
    selected_rows = [
        row for row in window_rows if row.get("session_key", "") == session_key
    ]
    selected_rows.sort(key=lambda row: (row["split"], int(row["array_index"])))

    windows: list[np.ndarray] = []
    for row in selected_rows:
        split = row["split"]
        array_index = int(row["array_index"])
        if split not in arrays:
            continue
        windows.append(arrays[split][array_index])

    if not windows:
        return np.zeros((0, 0, 0), dtype=np.float32), []
    return np.stack(windows).astype(np.float32), selected_rows


def _summarize(errors: np.ndarray, threshold: float) -> dict[str, Any]:
    if errors.size == 0:
        return {
            "count": 0,
            "mean_error": None,
            "median_error": None,
            "p95_error": None,
            "max_error": None,
            "above_threshold_count": 0,
            "above_threshold_ratio": None,
        }
    above = errors > threshold
    return {
        "count": int(errors.size),
        "mean_error": round(float(errors.mean()), 8),
        "median_error": round(float(np.median(errors)), 8),
        "p95_error": round(float(np.quantile(errors, 0.95)), 8),
        "max_error": round(float(errors.max()), 8),
        "above_threshold_count": int(above.sum()),
        "above_threshold_ratio": round(float(above.mean()), 8),
    }


def _write_scores(
    rows: list[dict[str, str]],
    errors: np.ndarray,
    threshold: float,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "array_index",
                "session_key",
                "source_file",
                "subject_id",
                "session_state",
                "start_timestamp",
                "end_timestamp",
                "reconstruction_error",
                "is_anomaly",
            ],
        )
        writer.writeheader()
        for row, error in zip(rows, errors.tolist(), strict=True):
            writer.writerow(
                {
                    "split": row.get("split", ""),
                    "array_index": row.get("array_index", ""),
                    "session_key": row.get("session_key", ""),
                    "source_file": row.get("source_file", ""),
                    "subject_id": row.get("subject_id", ""),
                    "session_state": row.get("session_state", ""),
                    "start_timestamp": row.get("start_timestamp", ""),
                    "end_timestamp": row.get("end_timestamp", ""),
                    "reconstruction_error": f"{float(error):.8f}",
                    "is_anomaly": "true" if float(error) > threshold else "false",
                }
            )


def _plot_scores(
    rows: list[dict[str, str]],
    errors: np.ndarray,
    threshold: float,
    output_dir: Path,
    session_key: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return

    x = np.arange(errors.size)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(x, errors, color="#2563eb", linewidth=2.0)
    ax.axhline(threshold, color="#111827", linestyle=":", linewidth=2.0, label="threshold")
    ax.fill_between(
        x,
        threshold,
        errors,
        where=errors > threshold,
        color="#dc2626",
        alpha=0.25,
    )
    ax.set_title(f"LSTM anomaly score - {session_key}")
    ax.set_xlabel("Window index in session")
    ax.set_ylabel("Reconstruction error")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "session_scores.png", dpi=180)
    fig.savefig(output_dir / "session_scores.svg")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir)
    dataset_dir = Path(args.dataset_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else run_dir / "figures" / "session_scores" / args.session_key
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = _load_json(run_dir / "metrics.json")
    threshold = float(metrics["threshold"]["reconstruction_error_threshold"])
    training_config = metrics["training_config"]

    loaded = np.load(dataset_dir / "dataset.npz")
    window_rows = _load_window_index(dataset_dir / "window_index.csv")
    windows, selected_rows = _session_windows(loaded, window_rows, args.session_key)
    if windows.shape[0] == 0:
        raise SystemExit(f"Nie znaleziono okien dla sesji: {args.session_key}")

    device = _resolve_device(args.device)
    model = LSTMAutoencoder(
        input_size=int(windows.shape[2]),
        hidden_size=int(training_config["hidden_size"]),
        latent_size=int(training_config["latent_size"]),
        num_layers=int(training_config["num_layers"]),
        dropout=float(training_config["dropout"]),
    ).to(device)
    model.load_state_dict(
        torch.load(run_dir / "model_best.pt", map_location=device, weights_only=True)
    )

    errors = _compute_reconstruction_errors(
        model,
        windows,
        device=device,
        batch_size=args.batch_size,
    )
    summary = {
        "session_key": args.session_key,
        "run_dir": str(run_dir),
        "dataset_dir": str(dataset_dir),
        "threshold": threshold,
        "split": selected_rows[0].get("split", ""),
        "session_state": selected_rows[0].get("session_state", ""),
        "start_timestamp": selected_rows[0].get("start_timestamp", ""),
        "end_timestamp": selected_rows[-1].get("end_timestamp", ""),
        "score_summary": _summarize(errors, threshold),
    }

    _write_scores(selected_rows, errors, threshold, output_dir / "scores_by_window.csv")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    _plot_scores(selected_rows, errors, threshold, output_dir, args.session_key)
    print(f"Saved session scoring to: {output_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
