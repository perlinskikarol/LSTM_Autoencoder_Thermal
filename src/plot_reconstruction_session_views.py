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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generuje dwa widoki dla wynikow LSTM Autoencoder: porownanie per sesja "
            "oraz anomaly score w czasie dla wybranej sesji."
        )
    )
    parser.add_argument(
        "--run-dir",
        default="runs\\lstm_autoencoder\\M1\\with_shower_20260424",
        help="Folder runu z metrics.json i scores_by_window.csv.",
    )
    parser.add_argument(
        "--session-key",
        default="M1_prysznic_16.04.2026_15.55.46",
        help="Session key do wykresu anomaly score w czasie.",
    )
    parser.add_argument(
        "--output-dir",
        default="Wykresy",
        help="Folder wyjsciowy na wykresy. Domyslnie: Wykresy",
    )
    parser.add_argument(
        "--output-prefix",
        default="with_shower_20260424",
        help="Prefiks nazw plikow wyjsciowych.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _load_scores(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    **row,
                    "array_index": int(row["array_index"]),
                    "reconstruction_error": float(row["reconstruction_error"]),
                    "is_anomaly_bool": row["is_anomaly"].strip().lower() == "true",
                    "start_dt": _parse_timestamp(row["start_timestamp"]) if row["start_timestamp"] else None,
                    "end_dt": _parse_timestamp(row["end_timestamp"]) if row["end_timestamp"] else None,
                }
            )
    return rows


def _session_split_rank(split: str) -> int:
    order = {"train": 0, "val": 1, "test_normal": 2, "test_anomaly": 3}
    return order.get(split, 99)


def _summarize_sessions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["session_key"]].append(row)

    summary: list[dict[str, Any]] = []
    for session_key, session_rows in grouped.items():
        session_rows = sorted(session_rows, key=lambda item: item["array_index"])
        errors = np.asarray([item["reconstruction_error"] for item in session_rows], dtype=np.float32)
        anomaly_ratio = float(np.mean([item["is_anomaly_bool"] for item in session_rows]))
        summary.append(
            {
                "session_key": session_key,
                "split": session_rows[0]["split"],
                "subject_id": session_rows[0]["subject_id"],
                "session_state": session_rows[0]["session_state"],
                "source_file": session_rows[0]["source_file"],
                "start_timestamp": session_rows[0]["start_timestamp"],
                "end_timestamp": session_rows[-1]["end_timestamp"],
                "count": int(errors.size),
                "mean_error": float(errors.mean()),
                "median_error": float(np.median(errors)),
                "p95_error": float(np.quantile(errors, 0.95)),
                "max_error": float(errors.max()),
                "above_threshold_ratio": anomaly_ratio,
                "above_threshold_count": int(round(anomaly_ratio * errors.size)),
            }
        )

    return sorted(
        summary,
        key=lambda item: (_session_split_rank(item["split"]), item["start_timestamp"], item["session_key"]),
    )


def _write_session_summary(summary_rows: list[dict[str, Any]], target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "session_key",
                "split",
                "subject_id",
                "session_state",
                "source_file",
                "start_timestamp",
                "end_timestamp",
                "count",
                "mean_error",
                "median_error",
                "p95_error",
                "max_error",
                "above_threshold_ratio",
                "above_threshold_count",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)


def _color_for_row(row: dict[str, Any]) -> str:
    if row["split"] == "test_anomaly":
        return "#dc2626"
    if row["split"] == "test_normal":
        return "#0f766e"
    if row["split"] == "val":
        return "#2563eb"
    return "#9ca3af"


def _short_label(session_key: str, split: str) -> str:
    pieces = session_key.split("_")
    if len(pieces) >= 4:
        state = pieces[1]
        date = pieces[2]
        time = pieces[3]
        return f"{state}\n{date}\n{time[:5]}"
    return f"{split}\n{session_key[-12:]}"


def _plot_per_session(
    summary_rows: list[dict[str, Any]],
    threshold: float,
    output_png: Path,
    output_svg: Path,
) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    labels = [_short_label(row["session_key"], row["split"]) for row in summary_rows]
    x = np.arange(len(summary_rows))
    colors = [_color_for_row(row) for row in summary_rows]
    anomaly_pct = np.asarray([row["above_threshold_ratio"] * 100.0 for row in summary_rows], dtype=np.float32)
    mean_error = np.asarray([row["mean_error"] for row in summary_rows], dtype=np.float32)
    p95_error = np.asarray([row["p95_error"] for row in summary_rows], dtype=np.float32)

    fig = plt.figure(figsize=(18, 10), facecolor="#f7f7f5", constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=[1.05, 1.0], hspace=0.22)
    ax_top = fig.add_subplot(grid[0, 0])
    ax_bottom = fig.add_subplot(grid[1, 0])

    bars = ax_top.bar(x, anomaly_pct, color=colors, alpha=0.86, edgecolor="#1f2937", linewidth=0.5)
    ax_top.set_ylabel("Okna powyzej progu [%]")
    ax_top.set_title("LSTM Autoencoder - porownanie rekonstrukcji per sesja")
    ax_top.grid(True, axis="y", alpha=0.18)
    ax_top.set_facecolor("#fcfcfb")
    for idx, (bar, row) in enumerate(zip(bars, summary_rows, strict=True)):
        if anomaly_pct[idx] >= 3.0:
            ax_top.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 1.0,
                f"{anomaly_pct[idx]:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#111827",
                rotation=90,
            )

    ax_bottom.plot(x, mean_error, color="#0f766e", linewidth=2.8, marker="o", label="Sredni blad")
    ax_bottom.plot(x, p95_error, color="#7c3aed", linewidth=2.8, marker="o", label="P95 bledu")
    ax_bottom.axhline(
        threshold,
        color="#111827",
        linestyle=":",
        linewidth=2.2,
        label=f"Prog anomalii = {threshold:.3f}",
    )
    for idx, row in enumerate(summary_rows):
        ax_bottom.scatter(
            idx,
            mean_error[idx],
            s=55,
            color=_color_for_row(row),
            edgecolors="#111827",
            linewidths=0.4,
            zorder=4,
        )
    ax_bottom.set_xticks(x, labels)
    ax_bottom.set_ylabel("Blad rekonstrukcji")
    ax_bottom.grid(True, axis="y", alpha=0.18)
    ax_bottom.set_facecolor("#fcfcfb")
    ax_bottom.legend(loc="upper right")

    legend_handles = [
        plt.Line2D([0], [0], color="#9ca3af", lw=8, label="train"),
        plt.Line2D([0], [0], color="#2563eb", lw=8, label="val"),
        plt.Line2D([0], [0], color="#0f766e", lw=8, label="test_normal"),
        plt.Line2D([0], [0], color="#dc2626", lw=8, label="test_anomaly"),
    ]
    ax_top.legend(handles=legend_handles, ncol=4, loc="upper right", frameon=True)

    fig.suptitle(
        "Porownanie sesji: udzial anomalii oraz poziom bledu",
        fontsize=20,
        fontweight="bold",
        y=0.98,
    )
    fig.savefig(output_png, dpi=240, facecolor=fig.get_facecolor())
    fig.savefig(output_svg, facecolor=fig.get_facecolor())
    plt.close(fig)


def _plot_session_timeline(
    session_rows: list[dict[str, Any]],
    threshold: float,
    output_png: Path,
    output_svg: Path,
) -> dict[str, Any]:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    session_rows = sorted(session_rows, key=lambda item: item["array_index"])
    first_dt = session_rows[0]["start_dt"]
    elapsed_min = np.asarray(
        [(row["start_dt"] - first_dt).total_seconds() / 60.0 for row in session_rows],
        dtype=np.float32,
    )
    errors = np.asarray([row["reconstruction_error"] for row in session_rows], dtype=np.float32)
    anomaly_mask = np.asarray([row["is_anomaly_bool"] for row in session_rows], dtype=bool)

    smooth_kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64)
    smooth_kernel /= smooth_kernel.sum()
    smooth_errors = np.convolve(errors, smooth_kernel, mode="same")

    fig = plt.figure(figsize=(16, 8), facecolor="#f7f7f5")
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(
        elapsed_min,
        errors,
        color="#94a3b8",
        linewidth=1.2,
        alpha=0.9,
        label="Blad okna 60 s",
    )
    ax.plot(
        elapsed_min,
        smooth_errors,
        color="#0f766e",
        linewidth=3.0,
        label="Wygładzony anomaly score",
    )
    ax.axhline(
        threshold,
        color="#dc2626",
        linestyle=":",
        linewidth=2.6,
        label=f"Prog anomalii = {threshold:.3f}",
    )
    if anomaly_mask.any():
        ax.scatter(
            elapsed_min[anomaly_mask],
            errors[anomaly_mask],
            color="#dc2626",
            s=34,
            alpha=0.95,
            label="Okna oznaczone jako anomalne",
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
    ax.set_xlabel("Czas od poczatku sesji [min]")
    ax.set_ylabel("Blad rekonstrukcji")
    ax.set_title(f"Anomaly score w czasie\n{session_rows[0]['session_key']}")
    ax.grid(True, alpha=0.18)
    ax.legend(loc="upper right", frameon=True)
    ax.set_facecolor("#fcfcfb")

    fig.tight_layout()
    fig.savefig(output_png, dpi=240, facecolor=fig.get_facecolor())
    fig.savefig(output_svg, facecolor=fig.get_facecolor())
    plt.close(fig)

    return {
        "session_key": session_rows[0]["session_key"],
        "split": session_rows[0]["split"],
        "session_state": session_rows[0]["session_state"],
        "count": int(errors.size),
        "duration_min": round(float(elapsed_min[-1]) if elapsed_min.size else 0.0, 4),
        "mean_error": round(float(errors.mean()), 8),
        "median_error": round(float(np.median(errors)), 8),
        "p95_error": round(float(np.quantile(errors, 0.95)), 8),
        "above_threshold_ratio": round(float(anomaly_mask.mean()), 8),
        "above_threshold_count": int(anomaly_mask.sum()),
        "threshold": round(float(threshold), 8),
    }


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    scores_path = run_dir / "scores_by_window.csv"
    metrics_path = run_dir / "metrics.json"

    rows = _load_scores(scores_path)
    metrics = _load_json(metrics_path)
    threshold = float(metrics["threshold"]["reconstruction_error_threshold"])
    session_summary = _summarize_sessions(rows)

    per_session_csv = output_dir / f"{args.output_prefix}_session_summary.csv"
    _write_session_summary(session_summary, per_session_csv)

    per_session_png = output_dir / f"{args.output_prefix}_per_session.png"
    per_session_svg = output_dir / f"{args.output_prefix}_per_session.svg"
    _plot_per_session(session_summary, threshold, per_session_png, per_session_svg)

    selected_session_rows = [row for row in rows if row["session_key"] == args.session_key]
    if not selected_session_rows:
        raise SystemExit(
            f"Nie znaleziono session_key={args.session_key} w {scores_path}"
        )

    timeline_png = output_dir / f"{args.output_prefix}_{args.session_key}_timeline.png"
    timeline_svg = output_dir / f"{args.output_prefix}_{args.session_key}_timeline.svg"
    timeline_summary = _plot_session_timeline(
        selected_session_rows,
        threshold,
        timeline_png,
        timeline_svg,
    )
    timeline_summary_path = output_dir / f"{args.output_prefix}_{args.session_key}_timeline_summary.json"
    timeline_summary_path.write_text(json.dumps(timeline_summary, indent=2), encoding="utf-8")

    print(f"Saved per-session plot: {per_session_png}")
    print(f"Saved per-session plot: {per_session_svg}")
    print(f"Saved session summary: {per_session_csv}")
    print(f"Saved timeline plot: {timeline_png}")
    print(f"Saved timeline plot: {timeline_svg}")
    print(f"Saved timeline summary: {timeline_summary_path}")
    print(json.dumps(timeline_summary, indent=2))


if __name__ == "__main__":
    main()
