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
            "Buduje porownawczy wykres ROC dla dwoch runow LSTM Autoencoder "
            "na podstawie scores_by_window.csv."
        )
    )
    parser.add_argument(
        "--run-a-dir",
        required=True,
        help="Folder pierwszego runu z metrics.json i scores_by_window.csv.",
    )
    parser.add_argument(
        "--run-a-label",
        required=True,
        help="Etykieta pierwszego runu na legendzie, np. 60 s.",
    )
    parser.add_argument(
        "--run-b-dir",
        required=True,
        help="Folder drugiego runu z metrics.json i scores_by_window.csv.",
    )
    parser.add_argument(
        "--run-b-label",
        required=True,
        help="Etykieta drugiego runu na legendzie, np. 10 min.",
    )
    parser.add_argument(
        "--output-dir",
        default="Wykresy",
        help="Folder wyjsciowy. Domyslnie: Wykresy",
    )
    parser.add_argument(
        "--output-name",
        default="roc_auc_comparison",
        help="Bazowa nazwa plikow wyjsciowych bez rozszerzenia.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_scores(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    csv_path = run_dir / "scores_by_window.csv"
    scores: list[float] = []
    labels: list[int] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            split = row["split"]
            if split == "test_normal":
                scores.append(float(row["reconstruction_error"]))
                labels.append(0)
            elif split == "test_anomaly":
                scores.append(float(row["reconstruction_error"]))
                labels.append(1)
    return np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.int32)


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _binary_metrics_at_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    preds = scores > threshold
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    fpr = _safe_div(fp, fp + tn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "fpr": fpr,
        "f1": f1,
    }


def _compute_roc(scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    if scores.size == 0 or len(np.unique(labels)) < 2:
        return {
            "thresholds": np.zeros((0,), dtype=np.float64),
            "roc_fpr": np.zeros((0,), dtype=np.float64),
            "roc_tpr": np.zeros((0,), dtype=np.float64),
            "roc_auc": None,
        }

    thresholds = np.unique(scores)[::-1]
    roc_fpr: list[float] = [0.0]
    roc_tpr: list[float] = [0.0]

    for threshold in thresholds:
        metrics = _binary_metrics_at_threshold(scores, labels, float(threshold))
        roc_fpr.append(metrics["fpr"])
        roc_tpr.append(metrics["recall"])

    roc_fpr.append(1.0)
    roc_tpr.append(1.0)

    roc_fpr_arr = np.asarray(roc_fpr, dtype=np.float64)
    roc_tpr_arr = np.asarray(roc_tpr, dtype=np.float64)
    order = np.argsort(roc_fpr_arr)
    roc_auc = float(np.trapezoid(roc_tpr_arr[order], roc_fpr_arr[order]))

    return {
        "thresholds": thresholds,
        "roc_fpr": roc_fpr_arr,
        "roc_tpr": roc_tpr_arr,
        "roc_auc": roc_auc,
    }


def _save(fig: plt.Figure, png_path: Path, svg_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


def _plot_comparison(
    roc_a: dict[str, Any],
    roc_b: dict[str, Any],
    label_a: str,
    label_b: str,
    output_dir: Path,
    output_name: str,
) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(9.5, 7.0), facecolor="#f7f7f5")
    ax.set_facecolor("#fcfcfb")

    ax.plot(
        roc_a["roc_fpr"],
        roc_a["roc_tpr"],
        color="#0f766e",
        linewidth=3.0,
        label=f"{label_a} (AUC={roc_a['roc_auc']:.3f})",
    )
    ax.plot(
        roc_b["roc_fpr"],
        roc_b["roc_tpr"],
        color="#b45309",
        linewidth=3.0,
        label=f"{label_b} (AUC={roc_b['roc_auc']:.3f})",
    )
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="#6b7280", linewidth=1.8, label="losowy klasyfikator")

    ax.set_title("Porownanie krzywych ROC", fontsize=18, fontweight="bold")
    ax.set_xlabel("False positive rate", fontsize=13)
    ax.set_ylabel("True positive rate / Recall", fontsize=13)
    ax.grid(True, alpha=0.18)
    ax.legend(frameon=True, facecolor="white", framealpha=0.92)

    png_path = output_dir / f"{output_name}.png"
    svg_path = output_dir / f"{output_name}.svg"
    _save(fig, png_path, svg_path)
    return png_path, svg_path


def main() -> None:
    args = _parse_args()
    run_a_dir = Path(args.run_a_dir)
    run_b_dir = Path(args.run_b_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_a = _load_json(run_a_dir / "metrics.json")
    metrics_b = _load_json(run_b_dir / "metrics.json")
    scores_a, labels_a = _load_scores(run_a_dir)
    scores_b, labels_b = _load_scores(run_b_dir)
    roc_a = _compute_roc(scores_a, labels_a)
    roc_b = _compute_roc(scores_b, labels_b)
    threshold_a = float(metrics_a["threshold"]["reconstruction_error_threshold"])
    threshold_b = float(metrics_b["threshold"]["reconstruction_error_threshold"])
    binary_a = _binary_metrics_at_threshold(scores_a, labels_a, threshold_a)
    binary_b = _binary_metrics_at_threshold(scores_b, labels_b, threshold_b)

    png_path, svg_path = _plot_comparison(
        roc_a=roc_a,
        roc_b=roc_b,
        label_a=args.run_a_label,
        label_b=args.run_b_label,
        output_dir=output_dir,
        output_name=args.output_name,
    )

    summary = {
        "run_a": {
            "label": args.run_a_label,
            "run_dir": str(run_a_dir),
            "roc_auc": roc_a["roc_auc"],
            "average_precision": metrics_a.get("average_precision"),
            "threshold": threshold_a,
            "recall": binary_a["recall"],
            "specificity": binary_a["specificity"],
            "precision": binary_a["precision"],
            "f1": binary_a["f1"],
        },
        "run_b": {
            "label": args.run_b_label,
            "run_dir": str(run_b_dir),
            "roc_auc": roc_b["roc_auc"],
            "average_precision": metrics_b.get("average_precision"),
            "threshold": threshold_b,
            "recall": binary_b["recall"],
            "specificity": binary_b["specificity"],
            "precision": binary_b["precision"],
            "f1": binary_b["f1"],
        },
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
