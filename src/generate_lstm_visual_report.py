from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


SPLIT_COLORS = {
    "train": "#94a3b8",
    "val": "#2563eb",
    "test_normal": "#0f766e",
    "test_anomaly": "#dc2626",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generuje raport wizualny dla wynikow LSTM Autoencoder: krzywe uczenia, "
            "ROC/PR, rozklady bledu, confusion matrix, analize per sesja i timeline testow."
        )
    )
    parser.add_argument(
        "--run-dir",
        default="runs\\lstm_autoencoder\\M1\\dynamics_200ep_pat10_20260429",
        help="Folder runu z metrics.json, history.csv i scores_by_window.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Folder wyjsciowy na raport. Domyslnie: <run-dir>\\figures"
        ),
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_history(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "epoch": int(row["epoch"]),
                    "train_loss": float(row["train_loss"]),
                    "val_loss": float(row["val_loss"]),
                }
            )
    return rows


def _load_scores(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            start_ts = row["start_timestamp"]
            end_ts = row["end_timestamp"]
            rows.append(
                {
                    **row,
                    "array_index": int(row["array_index"]),
                    "reconstruction_error": float(row["reconstruction_error"]),
                    "is_anomaly_bool": row["is_anomaly"].strip().lower() == "true",
                    "start_dt": datetime.fromisoformat(start_ts) if start_ts else None,
                    "end_dt": datetime.fromisoformat(end_ts) if end_ts else None,
                }
            )
    return rows


def _group_errors_by_split(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["split"]].append(row["reconstruction_error"])
    return {
        split: np.asarray(values, dtype=np.float32)
        for split, values in grouped.items()
    }


def _compute_binary_eval_rows(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    scores: list[float] = []
    labels: list[int] = []
    for row in rows:
        if row["split"] == "test_normal":
            scores.append(row["reconstruction_error"])
            labels.append(0)
        elif row["split"] == "test_anomaly":
            scores.append(row["reconstruction_error"])
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


def _compute_roc_pr(scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    if scores.size == 0 or len(np.unique(labels)) < 2:
        return {
            "thresholds": np.zeros((0,), dtype=np.float64),
            "roc_fpr": np.zeros((0,), dtype=np.float64),
            "roc_tpr": np.zeros((0,), dtype=np.float64),
            "pr_precision": np.zeros((0,), dtype=np.float64),
            "pr_recall": np.zeros((0,), dtype=np.float64),
            "roc_auc": None,
            "average_precision": None,
        }

    thresholds = np.unique(scores)[::-1]
    roc_fpr: list[float] = [0.0]
    roc_tpr: list[float] = [0.0]
    pr_precision: list[float] = [1.0]
    pr_recall: list[float] = [0.0]

    for threshold in thresholds:
        metrics = _binary_metrics_at_threshold(scores, labels, float(threshold))
        roc_fpr.append(metrics["fpr"])
        roc_tpr.append(metrics["recall"])
        pr_precision.append(metrics["precision"])
        pr_recall.append(metrics["recall"])

    roc_fpr.append(1.0)
    roc_tpr.append(1.0)
    pr_precision.append(float(np.mean(labels)))
    pr_recall.append(1.0)

    roc_fpr_arr = np.asarray(roc_fpr, dtype=np.float64)
    roc_tpr_arr = np.asarray(roc_tpr, dtype=np.float64)
    pr_precision_arr = np.asarray(pr_precision, dtype=np.float64)
    pr_recall_arr = np.asarray(pr_recall, dtype=np.float64)

    order = np.argsort(roc_fpr_arr)
    roc_auc = float(np.trapezoid(roc_tpr_arr[order], roc_fpr_arr[order]))

    pr_order = np.argsort(pr_recall_arr)
    average_precision = float(np.trapezoid(pr_precision_arr[pr_order], pr_recall_arr[pr_order]))

    return {
        "thresholds": thresholds,
        "roc_fpr": roc_fpr_arr,
        "roc_tpr": roc_tpr_arr,
        "pr_precision": pr_precision_arr,
        "pr_recall": pr_recall_arr,
        "roc_auc": roc_auc,
        "average_precision": average_precision,
    }


def _smooth_hist(errors: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hist, edges = np.histogram(errors, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64)
    kernel /= kernel.sum()
    smooth = np.convolve(hist, kernel, mode="same")
    return centers, smooth


def _summarize_sessions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["session_key"]].append(row)

    output: list[dict[str, Any]] = []
    for session_key, session_rows in grouped.items():
        session_rows = sorted(session_rows, key=lambda item: item["array_index"])
        errors = np.asarray([row["reconstruction_error"] for row in session_rows], dtype=np.float64)
        ratio = float(np.mean([row["is_anomaly_bool"] for row in session_rows]))
        output.append(
            {
                "session_key": session_key,
                "split": session_rows[0]["split"],
                "session_state": session_rows[0]["session_state"],
                "subject_id": session_rows[0]["subject_id"],
                "count": int(errors.size),
                "start_timestamp": session_rows[0]["start_timestamp"],
                "end_timestamp": session_rows[-1]["end_timestamp"],
                "mean_error": float(errors.mean()),
                "median_error": float(np.median(errors)),
                "p95_error": float(np.quantile(errors, 0.95)),
                "max_error": float(errors.max()),
                "above_threshold_ratio": ratio,
                "above_threshold_count": int(np.sum([row["is_anomaly_bool"] for row in session_rows])),
            }
        )
    split_order = {"train": 0, "val": 1, "test_normal": 2, "test_anomaly": 3}
    return sorted(
        output,
        key=lambda item: (split_order.get(item["split"], 99), item["start_timestamp"], item["session_key"]),
    )


def _write_session_summary(rows: list[dict[str, Any]], target_path: Path) -> None:
    with target_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "session_key",
                "split",
                "session_state",
                "subject_id",
                "count",
                "start_timestamp",
                "end_timestamp",
                "mean_error",
                "median_error",
                "p95_error",
                "max_error",
                "above_threshold_ratio",
                "above_threshold_count",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown_report(
    metrics: dict[str, Any],
    session_rows: list[dict[str, Any]],
    binary_metrics: dict[str, float] | None,
    roc_pr: dict[str, Any],
    output_path: Path,
) -> None:
    lines = [
        "# LSTM Autoencoder Visual Report",
        "",
        f"- Subject: `{metrics['subject_id']}`",
        f"- Run dir: `{metrics['run_dir']}`",
        f"- Best epoch: `{metrics['best_epoch']}`",
        f"- Best val loss: `{metrics['best_val_loss']}`",
        f"- Threshold: `{metrics['threshold']['reconstruction_error_threshold']}`",
        "",
        "## Split metrics",
        "",
        "| Split | Count | Mean | Median | P95 | Max | Above threshold |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split_name, split_data in metrics["split_metrics"].items():
        lines.append(
            "| "
            + split_name
            + f" | {split_data['count']} | {split_data['mean_error']} | {split_data['median_error']} | "
            + f"{split_data['p95_error']} | {split_data['max_error']} | {split_data['above_threshold_ratio']} |"
        )

    lines.extend(["", "## Binary evaluation (test_normal vs test_anomaly)", ""])
    if binary_metrics is None or roc_pr["roc_auc"] is None:
        lines.append("Brak danych do ROC/PR.")
    else:
        lines.append(f"- ROC AUC: `{roc_pr['roc_auc']:.4f}`")
        lines.append(f"- Average precision: `{roc_pr['average_precision']:.4f}`")
        lines.append(f"- Precision @ threshold: `{binary_metrics['precision']:.4f}`")
        lines.append(f"- Recall @ threshold: `{binary_metrics['recall']:.4f}`")
        lines.append(f"- Specificity @ threshold: `{binary_metrics['specificity']:.4f}`")
        lines.append(f"- F1 @ threshold: `{binary_metrics['f1']:.4f}`")

    test_session_rows = [row for row in session_rows if row["split"] in ("test_normal", "test_anomaly")]
    if test_session_rows:
        lines.extend(["", "## Test sessions", "", "| Session | Split | State | Mean | P95 | Above threshold |", "|---|---|---|---:|---:|---:|"])
        for row in test_session_rows:
            lines.append(
                f"| {row['session_key']} | {row['split']} | {row['session_state']} | "
                f"{row['mean_error']:.4f} | {row['p95_error']:.4f} | {row['above_threshold_ratio']:.4f} |"
            )

    lines.extend(
        [
            "",
            "## Figures",
            "",
            "| File prefix | What it shows |",
            "|---|---|",
            "| `01_loss_curve` | Train/validation loss and best epoch |",
            "| `02_error_distribution` | Reconstruction-error distributions for train/val/test splits |",
            "| `03_boxplot_splits` | Split-level error boxplots with the operating threshold |",
            "| `04_roc_pr_curves` | ROC and Precision-Recall curves |",
            "| `05_confusion_matrix` | Count-based confusion matrix at the operating threshold |",
            "| `06_threshold_sweep` | Precision/recall/specificity/F1 by score quantile threshold |",
            "| `07_session_ranking` | Test-session detection ratios and P95 errors |",
            "| `08_test_timelines` | Reconstruction-error timelines for test sessions |",
            "| `09_error_ecdf` | Empirical CDFs of reconstruction error by split |",
            "| `10_test_overlap_zoom` | Zoomed test_normal vs test_anomaly score overlap |",
            "| `11_metrics_by_raw_threshold` | Metrics as a function of raw reconstruction-error threshold |",
            "| `12_confusion_matrix_normalized` | Percentage-normalized confusion matrix |",
            "| `13_detection_gain` | Ranked-score gain/lift style view |",
            "| `14_session_error_boxplots` | Error boxplots for each test session |",
            "| `15_session_detection_heatmap` | Window-level threshold hits per test session |",
            "| `16_session_quantile_profile` | Error quantile profiles per test session |",
        ]
    )

    lines.extend(
        [
            "",
            "## Tables",
            "",
            "- `threshold_sweep.csv`: quantile-threshold metrics.",
            "- `raw_threshold_metrics.csv`: raw-threshold metrics with TP/FP/TN/FN.",
            "- `session_summary.csv`: split/session-level score summary.",
            "- `top_100_test_windows_by_error.csv`: highest-error test windows.",
            "- `top_100_missed_anomaly_windows.csv`: anomaly windows below threshold, sorted by score.",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "figure.titlesize": 18,
        }
    )


def _save(fig: plt.Figure, png_path: Path, svg_path: Path) -> None:
    fig.savefig(png_path, dpi=240, facecolor=fig.get_facecolor(), bbox_inches="tight")
    fig.savefig(svg_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _plot_loss(history_rows: list[dict[str, Any]], output_dir: Path) -> None:
    epochs = [row["epoch"] for row in history_rows]
    train_loss = [row["train_loss"] for row in history_rows]
    val_loss = [row["val_loss"] for row in history_rows]

    fig, ax = plt.subplots(figsize=(10.5, 6.2), facecolor="#f7f7f5")
    ax.plot(epochs, train_loss, color="#0f766e", linewidth=2.6, label="train_loss")
    ax.plot(epochs, val_loss, color="#dc2626", linewidth=2.6, label="val_loss")
    best_idx = int(np.argmin(val_loss))
    ax.scatter([epochs[best_idx]], [val_loss[best_idx]], color="#111827", s=60, zorder=5, label="best val")
    ax.set_xlabel("Epoka")
    ax.set_ylabel("MSE")
    ax.set_title("Krzywa uczenia")
    ax.grid(True, alpha=0.2)
    ax.legend()
    ax.set_facecolor("#fcfcfb")
    _save(fig, output_dir / "01_loss_curve.png", output_dir / "01_loss_curve.svg")


def _plot_error_distribution(
    errors_by_split: dict[str, np.ndarray],
    threshold: float,
    output_dir: Path,
) -> None:
    selected_splits = ["train", "val", "test_normal", "test_anomaly"]
    available = [split for split in selected_splits if split in errors_by_split and errors_by_split[split].size > 0]
    all_values = np.concatenate([errors_by_split[split] for split in available])
    x_max = float(np.quantile(all_values, 0.995) * 1.10)
    bins = np.linspace(0.0, max(x_max, threshold * 1.05), 120)

    fig, ax = plt.subplots(figsize=(12.5, 7.2), facecolor="#f7f7f5")
    for split in available:
        centers, smooth = _smooth_hist(errors_by_split[split], bins)
        ax.fill_between(centers, smooth, color=SPLIT_COLORS[split], alpha=0.14)
        ax.plot(centers, smooth, color=SPLIT_COLORS[split], linewidth=2.6, label=split)
    ax.axvline(threshold, color="#111827", linestyle=":", linewidth=2.3, label=f"threshold={threshold:.3f}")
    ax.set_xlabel("Blad rekonstrukcji")
    ax.set_ylabel("Gestosc")
    ax.set_title("Rozklad bledu rekonstrukcji")
    ax.grid(True, alpha=0.18)
    ax.legend()
    ax.set_facecolor("#fcfcfb")
    _save(fig, output_dir / "02_error_distribution.png", output_dir / "02_error_distribution.svg")


def _plot_boxplot(
    errors_by_split: dict[str, np.ndarray],
    threshold: float,
    output_dir: Path,
) -> None:
    selected_splits = [split for split in ("train", "val", "test_normal", "test_anomaly") if split in errors_by_split and errors_by_split[split].size > 0]
    data = [errors_by_split[split] for split in selected_splits]

    fig, ax = plt.subplots(figsize=(10.5, 6.5), facecolor="#f7f7f5")
    box = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.6,
        medianprops={"color": "#111827", "linewidth": 2.0},
        whiskerprops={"color": "#374151", "linewidth": 1.4},
        capprops={"color": "#374151", "linewidth": 1.4},
        boxprops={"color": "#374151", "linewidth": 1.4},
        flierprops={"marker": "o", "markersize": 2.5, "alpha": 0.18},
    )
    for patch, split in zip(box["boxes"], selected_splits, strict=True):
        patch.set_facecolor(SPLIT_COLORS[split])
        patch.set_alpha(0.45)
    ax.axhline(threshold, color="#111827", linestyle=":", linewidth=2.0, label="threshold")
    ax.set_xticklabels(selected_splits)
    ax.set_ylabel("Blad rekonstrukcji")
    ax.set_title("Boxplot bledu rekonstrukcji")
    ax.grid(True, axis="y", alpha=0.18)
    ax.legend()
    ax.set_facecolor("#fcfcfb")
    _save(fig, output_dir / "03_boxplot_splits.png", output_dir / "03_boxplot_splits.svg")


def _plot_roc_pr(
    roc_pr: dict[str, Any],
    output_dir: Path,
) -> None:
    if roc_pr["roc_auc"] is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), facecolor="#f7f7f5")

    axes[0].plot(roc_pr["roc_fpr"], roc_pr["roc_tpr"], color="#2563eb", linewidth=2.8)
    axes[0].plot([0, 1], [0, 1], color="#9ca3af", linestyle="--", linewidth=1.5)
    axes[0].set_title(f"ROC curve (AUC={roc_pr['roc_auc']:.3f})")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].grid(True, alpha=0.18)
    axes[0].set_facecolor("#fcfcfb")

    axes[1].plot(roc_pr["pr_recall"], roc_pr["pr_precision"], color="#dc2626", linewidth=2.8)
    axes[1].set_title(f"Precision-Recall (AP={roc_pr['average_precision']:.3f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].grid(True, alpha=0.18)
    axes[1].set_facecolor("#fcfcfb")

    _save(fig, output_dir / "04_roc_pr_curves.png", output_dir / "04_roc_pr_curves.svg")


def _plot_confusion_matrix(
    binary_metrics: dict[str, float] | None,
    output_dir: Path,
) -> None:
    if binary_metrics is None:
        return
    matrix = np.asarray(
        [
            [binary_metrics["tn"], binary_metrics["fp"]],
            [binary_metrics["fn"], binary_metrics["tp"]],
        ],
        dtype=np.float64,
    )

    fig, ax = plt.subplots(figsize=(6.2, 5.8), facecolor="#f7f7f5")
    im = ax.imshow(matrix, cmap="Blues")
    for (row, col), value in np.ndenumerate(matrix):
        ax.text(col, row, int(value), ha="center", va="center", color="#111827", fontsize=14, fontweight="bold")
    ax.set_xticks([0, 1], ["Pred normal", "Pred anomaly"])
    ax.set_yticks([0, 1], ["True normal", "True anomaly"])
    ax.set_title("Confusion matrix")
    fig.colorbar(im, ax=ax, shrink=0.88)
    _save(fig, output_dir / "05_confusion_matrix.png", output_dir / "05_confusion_matrix.svg")


def _plot_threshold_sweep(
    scores: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
) -> list[dict[str, float]]:
    if scores.size == 0 or len(np.unique(labels)) < 2:
        return []

    quantiles = np.linspace(0.90, 0.995, 20)
    rows: list[dict[str, float]] = []
    for quantile in quantiles:
        threshold = float(np.quantile(scores, quantile))
        metrics = _binary_metrics_at_threshold(scores, labels, threshold)
        rows.append(
            {
                "quantile": float(quantile),
                "threshold": threshold,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "specificity": metrics["specificity"],
                "f1": metrics["f1"],
                "fpr": metrics["fpr"],
            }
        )

    fig, ax = plt.subplots(figsize=(11.5, 6.4), facecolor="#f7f7f5")
    ax.plot([row["quantile"] for row in rows], [row["recall"] for row in rows], label="recall", color="#dc2626", linewidth=2.5)
    ax.plot([row["quantile"] for row in rows], [row["precision"] for row in rows], label="precision", color="#2563eb", linewidth=2.5)
    ax.plot([row["quantile"] for row in rows], [row["specificity"] for row in rows], label="specificity", color="#0f766e", linewidth=2.5)
    ax.plot([row["quantile"] for row in rows], [row["f1"] for row in rows], label="f1", color="#7c3aed", linewidth=2.5)
    ax.set_xlabel("Threshold quantile")
    ax.set_ylabel("Metric value")
    ax.set_title("Threshold sensitivity analysis")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.18)
    ax.legend(ncol=2)
    ax.set_facecolor("#fcfcfb")
    _save(fig, output_dir / "06_threshold_sweep.png", output_dir / "06_threshold_sweep.svg")
    return rows


def _plot_session_ranking(
    session_rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    selected = [row for row in session_rows if row["split"] in ("test_normal", "test_anomaly")]
    if not selected:
        return

    x = np.arange(len(selected))
    ratio_pct = np.asarray([row["above_threshold_ratio"] * 100.0 for row in selected], dtype=np.float64)
    p95 = np.asarray([row["p95_error"] for row in selected], dtype=np.float64)
    colors = [SPLIT_COLORS[row["split"]] for row in selected]
    labels = [f"{row['session_state']}\n{row['session_key'].split('_')[2]}" if len(row["session_key"].split("_")) >= 3 else row["session_key"] for row in selected]

    fig = plt.figure(figsize=(14.5, 8.0), facecolor="#f7f7f5")
    grid = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.20)
    ax_top = fig.add_subplot(grid[0, 0])
    ax_bottom = fig.add_subplot(grid[1, 0])

    ax_top.bar(x, ratio_pct, color=colors, alpha=0.82, edgecolor="#1f2937", linewidth=0.5)
    ax_top.set_ylabel("Okna > prog [%]")
    ax_top.set_title("Porownanie sesji testowych")
    ax_top.grid(True, axis="y", alpha=0.18)
    ax_top.set_facecolor("#fcfcfb")

    ax_bottom.bar(x, p95, color=colors, alpha=0.55, edgecolor="#1f2937", linewidth=0.5)
    ax_bottom.set_xticks(x, labels)
    ax_bottom.set_ylabel("P95 bledu")
    ax_bottom.grid(True, axis="y", alpha=0.18)
    ax_bottom.set_facecolor("#fcfcfb")

    _save(fig, output_dir / "07_session_ranking.png", output_dir / "07_session_ranking.svg")


def _plot_test_timelines(
    rows: list[dict[str, Any]],
    threshold: float,
    output_dir: Path,
) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["split"] in ("test_normal", "test_anomaly"):
            grouped[row["session_key"]].append(row)
    if not grouped:
        return

    session_keys = sorted(
        grouped.keys(),
        key=lambda key: grouped[key][0]["start_timestamp"],
    )
    n = len(session_keys)
    fig, axes = plt.subplots(n, 1, figsize=(15.5, 3.4 * n), facecolor="#f7f7f5", squeeze=False)
    axes_flat = axes.flatten()
    for ax, session_key in zip(axes_flat, session_keys, strict=True):
        session_rows = sorted(grouped[session_key], key=lambda item: item["array_index"])
        first_dt = session_rows[0]["start_dt"]
        x_min = np.asarray(
            [(row["start_dt"] - first_dt).total_seconds() / 60.0 for row in session_rows],
            dtype=np.float64,
        )
        errors = np.asarray([row["reconstruction_error"] for row in session_rows], dtype=np.float64)
        mask = np.asarray([row["is_anomaly_bool"] for row in session_rows], dtype=bool)
        split = session_rows[0]["split"]
        color = SPLIT_COLORS[split]

        kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64)
        kernel /= kernel.sum()
        smooth = np.convolve(errors, kernel, mode="same")

        ax.plot(x_min, errors, color="#94a3b8", linewidth=1.0, alpha=0.8)
        ax.plot(x_min, smooth, color=color, linewidth=2.6)
        ax.axhline(threshold, color="#111827", linestyle=":", linewidth=2.0)
        if np.any(mask):
            ax.scatter(x_min[mask], errors[mask], color="#dc2626", s=22, zorder=5)
        ax.set_title(f"{session_key} ({split})")
        ax.set_ylabel("Error")
        ax.grid(True, alpha=0.16)
        ax.set_facecolor("#fcfcfb")
    axes_flat[-1].set_xlabel("Czas od poczatku sesji [min]")
    _save(fig, output_dir / "08_test_timelines.png", output_dir / "08_test_timelines.svg")


def _plot_error_ecdf(
    errors_by_split: dict[str, np.ndarray],
    threshold: float,
    output_dir: Path,
) -> None:
    selected_splits = [
        split
        for split in ("train", "val", "test_normal", "test_anomaly")
        if split in errors_by_split and errors_by_split[split].size > 0
    ]
    if not selected_splits:
        return

    fig, ax = plt.subplots(figsize=(11.5, 6.4), facecolor="#f7f7f5")
    for split in selected_splits:
        values = np.sort(errors_by_split[split].astype(np.float64))
        y = np.arange(1, values.size + 1, dtype=np.float64) / values.size
        ax.plot(values, y, color=SPLIT_COLORS[split], linewidth=2.5, label=split)

    ax.axvline(threshold, color="#111827", linestyle=":", linewidth=2.2, label="threshold")
    ax.set_xlabel("Blad rekonstrukcji")
    ax.set_ylabel("ECDF")
    ax.set_title("Dystrybuanta empiryczna bledu")
    ax.grid(True, alpha=0.18)
    ax.legend()
    ax.set_facecolor("#fcfcfb")
    _save(fig, output_dir / "09_error_ecdf.png", output_dir / "09_error_ecdf.svg")


def _plot_test_overlap_zoom(
    errors_by_split: dict[str, np.ndarray],
    threshold: float,
    output_dir: Path,
) -> None:
    normal = errors_by_split.get("test_normal", np.zeros((0,), dtype=np.float32))
    anomaly = errors_by_split.get("test_anomaly", np.zeros((0,), dtype=np.float32))
    if normal.size == 0 or anomaly.size == 0:
        return

    combined = np.concatenate([normal, anomaly]).astype(np.float64)
    x_max = max(float(np.quantile(combined, 0.98)) * 1.15, threshold * 1.10)
    bins = np.linspace(0.0, x_max, 80)

    fig, ax = plt.subplots(figsize=(11.5, 6.4), facecolor="#f7f7f5")
    for split, values in (("test_normal", normal), ("test_anomaly", anomaly)):
        clipped = np.asarray(values, dtype=np.float64)
        clipped = clipped[clipped <= x_max]
        centers, smooth = _smooth_hist(clipped, bins)
        ax.fill_between(centers, smooth, color=SPLIT_COLORS[split], alpha=0.20)
        ax.plot(centers, smooth, color=SPLIT_COLORS[split], linewidth=2.8, label=split)

    for quantile, linestyle in ((0.50, "--"), (0.95, "-.")):
        ax.axvline(
            float(np.quantile(normal, quantile)),
            color=SPLIT_COLORS["test_normal"],
            linestyle=linestyle,
            linewidth=1.5,
            alpha=0.80,
        )
        ax.axvline(
            float(np.quantile(anomaly, quantile)),
            color=SPLIT_COLORS["test_anomaly"],
            linestyle=linestyle,
            linewidth=1.5,
            alpha=0.80,
        )
    ax.axvline(threshold, color="#111827", linestyle=":", linewidth=2.2, label="threshold")
    ax.set_xlabel("Blad rekonstrukcji")
    ax.set_ylabel("Gestosc")
    ax.set_title("Nakladanie rozkladow test_normal i test_anomaly")
    ax.grid(True, alpha=0.18)
    ax.legend()
    ax.set_facecolor("#fcfcfb")
    _save(fig, output_dir / "10_test_overlap_zoom.png", output_dir / "10_test_overlap_zoom.svg")


def _plot_raw_threshold_curves(
    scores: np.ndarray,
    labels: np.ndarray,
    operating_threshold: float,
    output_dir: Path,
) -> list[dict[str, float]]:
    if scores.size == 0 or len(np.unique(labels)) < 2:
        return []

    quantiles = np.linspace(0.0, 1.0, 240)
    thresholds = np.quantile(scores, quantiles)
    thresholds = np.unique(np.concatenate([thresholds, np.asarray([operating_threshold])]))

    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        metrics = _binary_metrics_at_threshold(scores, labels, float(threshold))
        rows.append(
            {
                "threshold": float(threshold),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "specificity": metrics["specificity"],
                "fpr": metrics["fpr"],
                "f1": metrics["f1"],
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "tn": metrics["tn"],
                "fn": metrics["fn"],
            }
        )

    fig, ax = plt.subplots(figsize=(12.0, 6.6), facecolor="#f7f7f5")
    for metric, color in (
        ("precision", "#2563eb"),
        ("recall", "#dc2626"),
        ("specificity", "#0f766e"),
        ("f1", "#7c3aed"),
        ("fpr", "#f59e0b"),
    ):
        ax.plot([row["threshold"] for row in rows], [row[metric] for row in rows], color=color, linewidth=2.4, label=metric)

    ax.axvline(operating_threshold, color="#111827", linestyle=":", linewidth=2.2, label="operating threshold")
    ax.set_xlabel("Prog bledu rekonstrukcji")
    ax.set_ylabel("Wartosc metryki")
    ax.set_title("Metryki klasyfikacji wzgledem progu")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.18)
    ax.legend(ncol=3)
    ax.set_facecolor("#fcfcfb")
    _save(fig, output_dir / "11_metrics_by_raw_threshold.png", output_dir / "11_metrics_by_raw_threshold.svg")
    return rows


def _plot_normalized_confusion_matrix(
    binary_metrics: dict[str, float] | None,
    output_dir: Path,
) -> None:
    if binary_metrics is None:
        return

    tn = binary_metrics["tn"]
    fp = binary_metrics["fp"]
    fn = binary_metrics["fn"]
    tp = binary_metrics["tp"]
    matrix = np.asarray(
        [
            [_safe_div(tn, tn + fp), _safe_div(fp, tn + fp)],
            [_safe_div(fn, fn + tp), _safe_div(tp, fn + tp)],
        ],
        dtype=np.float64,
    )

    fig, ax = plt.subplots(figsize=(6.2, 5.8), facecolor="#f7f7f5")
    im = ax.imshow(matrix, cmap="Greens", vmin=0.0, vmax=1.0)
    counts = np.asarray([[tn, fp], [fn, tp]], dtype=np.int64)
    for (row, col), value in np.ndenumerate(matrix):
        ax.text(
            col,
            row,
            f"{value * 100:.1f}%\n(n={counts[row, col]})",
            ha="center",
            va="center",
            color="#111827",
            fontsize=12,
            fontweight="bold",
        )
    ax.set_xticks([0, 1], ["Pred normal", "Pred anomaly"])
    ax.set_yticks([0, 1], ["True normal", "True anomaly"])
    ax.set_title("Normalized confusion matrix")
    fig.colorbar(im, ax=ax, shrink=0.88)
    _save(fig, output_dir / "12_confusion_matrix_normalized.png", output_dir / "12_confusion_matrix_normalized.svg")


def _plot_detection_gain(
    scores: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
) -> None:
    if scores.size == 0 or len(np.unique(labels)) < 2:
        return

    order = np.argsort(scores)[::-1]
    sorted_labels = labels[order]
    sample_fraction = np.arange(1, sorted_labels.size + 1, dtype=np.float64) / sorted_labels.size
    positives = float(np.sum(labels == 1))
    negatives = float(np.sum(labels == 0))
    cumulative_recall = np.cumsum(sorted_labels == 1) / positives
    cumulative_fpr = np.cumsum(sorted_labels == 0) / negatives

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), facecolor="#f7f7f5")
    axes[0].plot(sample_fraction, cumulative_recall, color="#dc2626", linewidth=2.8, label="anomaly recall")
    axes[0].plot(sample_fraction, sample_fraction, color="#9ca3af", linestyle="--", linewidth=1.5, label="random")
    axes[0].set_xlabel("Odsetek okien sprawdzanych od najwyzszego bledu")
    axes[0].set_ylabel("Skumulowany recall anomalii")
    axes[0].set_title("Cumulative gain")
    axes[0].grid(True, alpha=0.18)
    axes[0].legend()
    axes[0].set_facecolor("#fcfcfb")

    axes[1].plot(sample_fraction, cumulative_recall, color="#dc2626", linewidth=2.8, label="recall")
    axes[1].plot(sample_fraction, cumulative_fpr, color="#f59e0b", linewidth=2.8, label="FPR consumed")
    axes[1].set_xlabel("Odsetek okien sprawdzanych od najwyzszego bledu")
    axes[1].set_ylabel("Odsetek klasy")
    axes[1].set_title("Recall vs false positives in ranked scores")
    axes[1].grid(True, alpha=0.18)
    axes[1].legend()
    axes[1].set_facecolor("#fcfcfb")

    _save(fig, output_dir / "13_detection_gain.png", output_dir / "13_detection_gain.svg")


def _plot_session_error_boxplots(
    rows: list[dict[str, Any]],
    threshold: float,
    output_dir: Path,
) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["split"] in ("test_normal", "test_anomaly"):
            grouped[row["session_key"]].append(row)
    if not grouped:
        return

    session_keys = sorted(grouped.keys(), key=lambda key: grouped[key][0]["start_timestamp"])
    data = [
        np.asarray([row["reconstruction_error"] for row in grouped[key]], dtype=np.float64)
        for key in session_keys
    ]
    labels = [
        f"{grouped[key][0]['session_state']}\n{key.split('_')[2] if len(key.split('_')) >= 3 else key}"
        for key in session_keys
    ]

    fig, ax = plt.subplots(figsize=(14.5, 6.8), facecolor="#f7f7f5")
    box = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.55,
        medianprops={"color": "#111827", "linewidth": 2.0},
        whiskerprops={"color": "#374151", "linewidth": 1.3},
        capprops={"color": "#374151", "linewidth": 1.3},
        boxprops={"color": "#374151", "linewidth": 1.3},
        flierprops={"marker": "o", "markersize": 2.4, "alpha": 0.20},
    )
    for patch, key in zip(box["boxes"], session_keys, strict=True):
        patch.set_facecolor(SPLIT_COLORS[grouped[key][0]["split"]])
        patch.set_alpha(0.45)
    ax.axhline(threshold, color="#111827", linestyle=":", linewidth=2.1, label="threshold")
    ax.set_xticklabels(labels)
    ax.set_ylabel("Blad rekonstrukcji")
    ax.set_title("Boxplot bledu dla sesji testowych")
    ax.grid(True, axis="y", alpha=0.18)
    ax.legend()
    ax.set_facecolor("#fcfcfb")
    _save(fig, output_dir / "14_session_error_boxplots.png", output_dir / "14_session_error_boxplots.svg")


def _plot_session_detection_heatmap(
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["split"] in ("test_normal", "test_anomaly"):
            grouped[row["session_key"]].append(row)
    if not grouped:
        return

    session_keys = sorted(grouped.keys(), key=lambda key: grouped[key][0]["start_timestamp"])
    max_len = max(len(grouped[key]) for key in session_keys)
    matrix = np.full((len(session_keys), max_len), np.nan, dtype=np.float64)
    labels: list[str] = []
    for row_idx, key in enumerate(session_keys):
        session_rows = sorted(grouped[key], key=lambda item: item["array_index"])
        flags = np.asarray([1.0 if row["is_anomaly_bool"] else 0.0 for row in session_rows], dtype=np.float64)
        matrix[row_idx, : flags.size] = flags
        ratio = float(np.nanmean(flags))
        labels.append(f"{session_rows[0]['session_state']} | {key.split('_')[2]} | {ratio * 100:.1f}%")

    cmap = plt.get_cmap("Reds").copy()
    cmap.set_bad(color="#e5e7eb")
    fig, ax = plt.subplots(figsize=(15.5, 0.95 * len(session_keys) + 3.0), facecolor="#f7f7f5")
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_yticks(np.arange(len(session_keys)), labels)
    ax.set_xlabel("Indeks okna w sesji")
    ax.set_title("Mapa wykryc progowych w sesjach testowych")
    fig.colorbar(im, ax=ax, shrink=0.82, ticks=[0, 1], label="Okno > prog")
    ax.set_facecolor("#fcfcfb")
    _save(fig, output_dir / "15_session_detection_heatmap.png", output_dir / "15_session_detection_heatmap.svg")


def _plot_session_quantile_profile(
    rows: list[dict[str, Any]],
    threshold: float,
    output_dir: Path,
) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["split"] in ("test_normal", "test_anomaly"):
            grouped[row["session_key"]].append(row)
    if not grouped:
        return

    session_keys = sorted(grouped.keys(), key=lambda key: grouped[key][0]["start_timestamp"])
    labels = [
        f"{grouped[key][0]['session_state']}\n{key.split('_')[2] if len(key.split('_')) >= 3 else key}"
        for key in session_keys
    ]
    quantiles = [0.50, 0.75, 0.90, 0.95]
    values = np.asarray(
        [
            np.quantile(
                np.asarray([row["reconstruction_error"] for row in grouped[key]], dtype=np.float64),
                quantiles,
            )
            for key in session_keys
        ],
        dtype=np.float64,
    )

    x = np.arange(len(session_keys))
    fig, ax = plt.subplots(figsize=(14.5, 6.8), facecolor="#f7f7f5")
    for idx, quantile in enumerate(quantiles):
        ax.plot(x, values[:, idx], marker="o", linewidth=2.2, label=f"Q{int(quantile * 100)}")
    ax.axhline(threshold, color="#111827", linestyle=":", linewidth=2.1, label="threshold")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Blad rekonstrukcji")
    ax.set_title("Profil kwantyli bledu per sesja")
    ax.grid(True, axis="y", alpha=0.18)
    ax.legend(ncol=5)
    ax.set_facecolor("#fcfcfb")
    _save(fig, output_dir / "16_session_quantile_profile.png", output_dir / "16_session_quantile_profile.svg")


def _write_ranked_window_tables(
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    test_rows = [row for row in rows if row["split"] in ("test_normal", "test_anomaly")]
    top_rows = sorted(test_rows, key=lambda row: row["reconstruction_error"], reverse=True)[:100]
    false_negative_rows = [
        row
        for row in test_rows
        if row["split"] == "test_anomaly" and not row["is_anomaly_bool"]
    ]
    false_negative_rows = sorted(false_negative_rows, key=lambda row: row["reconstruction_error"], reverse=True)[:100]

    fieldnames = [
        "split",
        "session_key",
        "session_state",
        "array_index",
        "start_timestamp",
        "end_timestamp",
        "reconstruction_error",
        "is_anomaly",
    ]
    for filename, selected_rows in (
        ("top_100_test_windows_by_error.csv", top_rows),
        ("top_100_missed_anomaly_windows.csv", false_negative_rows),
    ):
        with (output_dir / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in selected_rows:
                writer.writerow({field: row[field] for field in fieldnames})


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Nie znaleziono run_dir: {run_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    _set_style()

    metrics = _load_json(run_dir / "metrics.json")
    history_rows = _load_history(run_dir / "history.csv")
    score_rows = _load_scores(run_dir / "scores_by_window.csv")
    errors_by_split = _group_errors_by_split(score_rows)
    session_rows = _summarize_sessions(score_rows)
    threshold = float(metrics["threshold"]["reconstruction_error_threshold"])
    scores, labels = _compute_binary_eval_rows(score_rows)
    roc_pr = _compute_roc_pr(scores, labels)
    binary_metrics = _binary_metrics_at_threshold(scores, labels, threshold) if scores.size > 0 and len(np.unique(labels)) == 2 else None

    _plot_loss(history_rows, output_dir)
    _plot_error_distribution(errors_by_split, threshold, output_dir)
    _plot_boxplot(errors_by_split, threshold, output_dir)
    _plot_roc_pr(roc_pr, output_dir)
    _plot_confusion_matrix(binary_metrics, output_dir)
    threshold_rows = _plot_threshold_sweep(scores, labels, output_dir)
    _plot_session_ranking(session_rows, output_dir)
    _plot_test_timelines(score_rows, threshold, output_dir)
    _plot_error_ecdf(errors_by_split, threshold, output_dir)
    _plot_test_overlap_zoom(errors_by_split, threshold, output_dir)
    raw_threshold_rows = _plot_raw_threshold_curves(scores, labels, threshold, output_dir)
    _plot_normalized_confusion_matrix(binary_metrics, output_dir)
    _plot_detection_gain(scores, labels, output_dir)
    _plot_session_error_boxplots(score_rows, threshold, output_dir)
    _plot_session_detection_heatmap(score_rows, output_dir)
    _plot_session_quantile_profile(score_rows, threshold, output_dir)
    _write_ranked_window_tables(score_rows, output_dir)

    _write_session_summary(session_rows, output_dir / "session_summary.csv")
    if threshold_rows:
        with (output_dir / "threshold_sweep.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["quantile", "threshold", "precision", "recall", "specificity", "f1", "fpr"],
            )
            writer.writeheader()
            writer.writerows(threshold_rows)
    if raw_threshold_rows:
        with (output_dir / "raw_threshold_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "threshold",
                    "precision",
                    "recall",
                    "specificity",
                    "fpr",
                    "f1",
                    "tp",
                    "fp",
                    "tn",
                    "fn",
                ],
            )
            writer.writeheader()
            writer.writerows(raw_threshold_rows)

    summary_payload = {
        "run_dir": str(run_dir),
        "subject_id": metrics["subject_id"],
        "best_epoch": metrics["best_epoch"],
        "best_val_loss": metrics["best_val_loss"],
        "threshold": metrics["threshold"],
        "split_metrics": metrics["split_metrics"],
        "binary_window_metrics": binary_metrics,
        "roc_auc": roc_pr["roc_auc"],
        "average_precision": roc_pr["average_precision"],
    }
    (output_dir / "report_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    _write_markdown_report(metrics, session_rows, binary_metrics, roc_pr, output_dir / "REPORT.md")

    print(f"Saved visual report to: {output_dir}")
    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()
