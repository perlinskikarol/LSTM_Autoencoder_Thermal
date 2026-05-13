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
import torch

from src.train_lstm_autoencoder import LSTMAutoencoder, _compute_reconstruction_errors


GROUP_CONFIG = {
    "m1_normal": {
        "label": "M1 normal",
        "color": "#0f766e",
    },
    "m1_prysznic": {
        "label": "M1 prysznic",
        "color": "#dc2626",
    },
    "k1_normal": {
        "label": "K1 normal",
        "color": "#2563eb",
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Porownuje rozklady bledu rekonstrukcji dla M1 normal, M1 prysznic "
            "oraz K1 liczonych tym samym modelem M1."
        )
    )
    parser.add_argument(
        "--run-dir",
        default="runs\\lstm_autoencoder\\M1\\with_shower_20260424",
        help="Folder runu z wytrenowanym modelem M1. Domyslnie: with_shower_20260424",
    )
    parser.add_argument(
        "--other-dataset-dir",
        default="Dane_przygotowane\\lstm_autoencoder\\K1",
        help="Folder datasetu drugiej osoby. Domyslnie: K1",
    )
    parser.add_argument(
        "--output-dir",
        default="Wykresy",
        help="Folder wyjsciowy na wykresy. Domyslnie: Wykresy",
    )
    parser.add_argument(
        "--output-name",
        default="reconstruction_error_comparison_M1_vs_K1",
        help="Bazowa nazwa plikow wyjsciowych bez rozszerzenia.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_scores_by_split(path: Path) -> dict[str, np.ndarray]:
    grouped: dict[str, list[float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            split = row["split"]
            grouped.setdefault(split, []).append(float(row["reconstruction_error"]))
    return {
        split: np.asarray(values, dtype=np.float32)
        for split, values in grouped.items()
    }


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


def _load_other_subject_errors(
    run_dir: Path,
    metrics: dict[str, Any],
    other_dataset_dir: Path,
) -> np.ndarray:
    m1_dataset_dir = Path(metrics["dataset_dir"])
    normalization = _load_json(m1_dataset_dir / "normalization.json")
    data_other = np.load(other_dataset_dir / "dataset.npz")
    raw_arrays = [
        data_other["X_train_raw"].astype(np.float32),
        data_other["X_test_normal_raw"].astype(np.float32),
        data_other["X_test_anomaly_raw"].astype(np.float32),
    ]
    raw_arrays = [array for array in raw_arrays if array.shape[0] > 0]
    if not raw_arrays:
        return np.zeros((0,), dtype=np.float32)

    x_other_raw = np.concatenate(raw_arrays, axis=0)
    feature_names = metrics["feature_names"]
    mean = np.asarray(
        [normalization["feature_mean"][feature] for feature in feature_names],
        dtype=np.float32,
    )
    std = np.asarray(
        [normalization["feature_std"][feature] for feature in feature_names],
        dtype=np.float32,
    )
    x_other = ((x_other_raw - mean) / std).astype(np.float32)

    model = _load_model(run_dir, metrics)
    batch_size = int(metrics["training_config"]["batch_size"])
    return _compute_reconstruction_errors(
        model,
        x_other,
        device=torch.device("cpu"),
        batch_size=batch_size,
    )


def _summarize(errors: np.ndarray, threshold: float) -> dict[str, Any]:
    if errors.size == 0:
        return {
            "count": 0,
            "mean_error": None,
            "median_error": None,
            "p95_error": None,
            "above_threshold_ratio": None,
            "threshold": threshold,
        }
    return {
        "count": int(errors.size),
        "mean_error": round(float(errors.mean()), 8),
        "median_error": round(float(np.median(errors)), 8),
        "p95_error": round(float(np.quantile(errors, 0.95)), 8),
        "above_threshold_ratio": round(float((errors > threshold).mean()), 8),
        "threshold": round(float(threshold), 8),
    }


def _smoothed_hist_line(errors: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hist, edges = np.histogram(errors, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64)
    kernel /= kernel.sum()
    smooth = np.convolve(hist, kernel, mode="same")
    return centers, smooth


def _build_plot(
    grouped_errors: dict[str, np.ndarray],
    threshold: float,
    output_png: Path,
    output_svg: Path,
) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    all_errors = np.concatenate([values for values in grouped_errors.values() if values.size > 0])
    x_min = 0.0
    x_max = float(np.quantile(all_errors, 0.995) * 1.12)
    bins = np.linspace(x_min, max(x_max, threshold * 1.1, 0.6), 80)

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 18,
            "axes.labelsize": 13,
            "legend.fontsize": 11,
        }
    )

    fig = plt.figure(figsize=(15.5, 8.5), facecolor="#f7f7f5")
    grid = fig.add_gridspec(2, 2, height_ratios=[3.2, 1.2], width_ratios=[3.0, 1.4])
    ax_density = fig.add_subplot(grid[:, 0])
    ax_box = fig.add_subplot(grid[0, 1])
    ax_text = fig.add_subplot(grid[1, 1])

    for key, errors in grouped_errors.items():
        cfg = GROUP_CONFIG[key]
        if errors.size == 0:
            continue
        centers, smooth = _smoothed_hist_line(errors, bins)
        ax_density.fill_between(
            centers,
            smooth,
            color=cfg["color"],
            alpha=0.16,
        )
        ax_density.plot(
            centers,
            smooth,
            color=cfg["color"],
            linewidth=3.0,
            label=f"{cfg['label']} (n={errors.size})",
        )
        ax_density.axvline(
            float(np.median(errors)),
            color=cfg["color"],
            linestyle="--",
            linewidth=1.2,
            alpha=0.65,
        )

    ax_density.axvline(
        threshold,
        color="#111827",
        linestyle=":",
        linewidth=2.6,
        label=f"Prog anomalii = {threshold:.3f}",
    )
    ax_density.set_title("Porownanie bledu rekonstrukcji")
    ax_density.set_xlabel("Blad rekonstrukcji (MSE na oknie 60 s)")
    ax_density.set_ylabel("Gestosc")
    ax_density.grid(True, alpha=0.18)
    ax_density.legend(loc="upper right", frameon=True)
    ax_density.set_facecolor("#fcfcfb")

    box_data = [grouped_errors[key] for key in ("m1_normal", "m1_prysznic", "k1_normal")]
    box_labels = [GROUP_CONFIG[key]["label"] for key in ("m1_normal", "m1_prysznic", "k1_normal")]
    box = ax_box.boxplot(
        box_data,
        vert=True,
        patch_artist=True,
        widths=0.55,
        medianprops={"color": "#111827", "linewidth": 2.0},
        whiskerprops={"color": "#374151", "linewidth": 1.4},
        capprops={"color": "#374151", "linewidth": 1.4},
        boxprops={"color": "#374151", "linewidth": 1.4},
        flierprops={
            "marker": "o",
            "markersize": 2.8,
            "alpha": 0.22,
            "markeredgecolor": "#6b7280",
            "markerfacecolor": "#9ca3af",
        },
    )
    for patch, key in zip(box["boxes"], ("m1_normal", "m1_prysznic", "k1_normal"), strict=True):
        patch.set_facecolor(GROUP_CONFIG[key]["color"])
        patch.set_alpha(0.50)
    ax_box.axhline(threshold, color="#111827", linestyle=":", linewidth=2.0)
    ax_box.set_xticklabels(box_labels, rotation=12)
    ax_box.set_ylabel("Blad rekonstrukcji")
    ax_box.set_title("Rozrzut bledu")
    ax_box.grid(True, axis="y", alpha=0.18)
    ax_box.set_facecolor("#fcfcfb")

    ax_text.axis("off")
    summaries = []
    for key in ("m1_normal", "m1_prysznic", "k1_normal"):
        cfg = GROUP_CONFIG[key]
        stats = _summarize(grouped_errors[key], threshold)
        summaries.append(
            (
                f"{cfg['label']}\n"
                f"n={stats['count']} | mean={stats['mean_error']:.3f} | "
                f"median={stats['median_error']:.3f} | "
                f"> prog={(stats['above_threshold_ratio'] * 100.0):.1f}%"
            )
        )
    ax_text.text(
        0.0,
        1.0,
        "Podsumowanie\n\n" + "\n\n".join(summaries),
        va="top",
        ha="left",
        fontsize=11.5,
        family="monospace",
        color="#111827",
    )

    fig.suptitle(
        "Model M1: M1 normal vs M1 prysznic vs K1 normal",
        fontsize=20,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0.01, 0.02, 0.99, 0.95])
    fig.savefig(output_png, dpi=240, facecolor=fig.get_facecolor())
    fig.savefig(output_svg, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir)
    other_dataset_dir = Path(args.other_dataset_dir)
    output_dir = Path(args.output_dir)

    metrics = _load_json(run_dir / "metrics.json")
    threshold = float(metrics["threshold"]["reconstruction_error_threshold"])
    split_scores = _load_scores_by_split(run_dir / "scores_by_window.csv")
    k1_errors = _load_other_subject_errors(run_dir, metrics, other_dataset_dir)

    grouped_errors = {
        "m1_normal": split_scores.get("test_normal", np.zeros((0,), dtype=np.float32)),
        "m1_prysznic": split_scores.get("test_anomaly", np.zeros((0,), dtype=np.float32)),
        "k1_normal": k1_errors,
    }

    output_png = output_dir / f"{args.output_name}.png"
    output_svg = output_dir / f"{args.output_name}.svg"
    _build_plot(grouped_errors, threshold, output_png, output_svg)

    summary_payload = {
        key: _summarize(values, threshold)
        for key, values in grouped_errors.items()
    }
    summary_path = output_dir / f"{args.output_name}_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print(f"Saved plot: {output_png}")
    print(f"Saved plot: {output_svg}")
    print(f"Saved summary: {summary_path}")
    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()
