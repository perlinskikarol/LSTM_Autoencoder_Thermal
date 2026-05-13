from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


SERIES_CONFIG = {
    "mean": {
        "column": "mean_temp_c",
        "label": "Temperatura srednia",
        "color": "#0f766e",
        "linewidth": 3.4,
    },
    "min": {
        "column": "min_temp_c",
        "label": "Temperatura minimalna",
        "color": "#2563eb",
        "linewidth": 2.4,
    },
    "max": {
        "column": "max_temp_c",
        "label": "Temperatura maksymalna",
        "color": "#dc2626",
        "linewidth": 2.4,
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generuje wykres temperatury w czasie z CSV zapisanego przez mp_forehead_ui."
    )
    parser.add_argument(
        "input_csv",
        help="Sciezka do pliku CSV z pomiarami.",
    )
    parser.add_argument(
        "--series",
        nargs="+",
        choices=sorted(SERIES_CONFIG.keys()),
        default=["mean"],
        help="Ktore serie temperatury narysowac. Domyslnie: mean",
    )
    parser.add_argument(
        "--from-sec",
        type=float,
        default=None,
        help="Poczatek zakresu czasu na osi X [s].",
    )
    parser.add_argument(
        "--to-sec",
        type=float,
        default=None,
        help="Koniec zakresu czasu na osi X [s].",
    )
    parser.add_argument(
        "--output-dir",
        default="Wykresy",
        help="Folder wyjsciowy na wykresy. Domyslnie: Wykresy",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["png", "svg"],
        default=["png", "svg"],
        help="Formaty plikow wyjsciowych. Domyslnie: png svg",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Rozdzielczosc dla PNG. Domyslnie: 220",
    )
    parser.add_argument(
        "--title",
        default="",
        help="Wlasny tytul wykresu. Jesli pusty, zostanie wygenerowany automatycznie.",
    )
    parser.add_argument(
        "--show-range-band",
        action="store_true",
        help="Dodatkowo wypelnij obszar miedzy min i max, jesli obie serie sa wybrane.",
    )
    return parser.parse_args()


def _load_rows(csv_path: Path) -> tuple[dict[str, str], list[dict[str, str]]]:
    with csv_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV has no data rows: {csv_path}")
    return rows[0], rows


def _to_float(value: str) -> float:
    text = (value or "").strip()
    if not text:
        return float("nan")
    return float(text)


def _filter_rows(
    rows: Iterable[dict[str, str]],
    from_sec: float | None,
    to_sec: float | None,
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in rows:
        elapsed = _to_float(row.get("elapsed_sec", ""))
        if np.isnan(elapsed):
            continue
        if from_sec is not None and elapsed < from_sec:
            continue
        if to_sec is not None and elapsed > to_sec:
            continue
        out.append(row)
    if not out:
        raise ValueError("No samples left after applying the selected time range.")
    return out


def _build_title(first_row: dict[str, str], from_sec: float | None, to_sec: float | None) -> str:
    subject_id = first_row.get("subject_id", "") or "-"
    session_state = first_row.get("session_state", "") or "-"
    date_text = first_row.get("date", "") or "-"
    time_text = first_row.get("time", "") or "-"
    title = f"Profil temperatury w czasie | {subject_id} | stan: {session_state} | start: {date_text} {time_text}"
    if from_sec is not None or to_sec is not None:
        from_label = f"{from_sec:.0f}" if from_sec is not None else "start"
        to_label = f"{to_sec:.0f}" if to_sec is not None else "koniec"
        title += f" | zakres: {from_label}-{to_label}s"
    return title


def _collect_series(rows: list[dict[str, str]], series_names: list[str]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    elapsed = np.asarray([_to_float(row.get("elapsed_sec", "")) for row in rows], dtype=float)
    series_data: dict[str, np.ndarray] = {}
    for name in series_names:
        column = SERIES_CONFIG[name]["column"]
        series_data[name] = np.asarray([_to_float(row.get(column, "")) for row in rows], dtype=float)
    return elapsed, series_data


def _render_plot(
    *,
    x: np.ndarray,
    series_names: list[str],
    series_data: dict[str, np.ndarray],
    title: str,
    output_png: Path | None,
    output_svg: Path | None,
    dpi: int,
    show_range_band: bool,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(16, 9), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fcfcfc")

    if show_range_band and "min" in series_names and "max" in series_names:
        min_arr = series_data["min"]
        max_arr = series_data["max"]
        valid_band = np.isfinite(min_arr) & np.isfinite(max_arr)
        if np.any(valid_band):
            ax.fill_between(
                x,
                min_arr,
                max_arr,
                where=valid_band,
                color="#f6c177",
                alpha=0.16,
                label="Zakres min-max",
                interpolate=True,
            )

    for name in series_names:
        cfg = SERIES_CONFIG[name]
        y = series_data[name]
        valid = np.isfinite(y)
        if not np.any(valid):
            continue
        ax.plot(
            x[valid],
            y[valid],
            color=cfg["color"],
            linewidth=cfg["linewidth"],
            label=cfg["label"],
        )

    ax.set_title(title, fontsize=22, fontweight="bold", pad=18)
    ax.set_xlabel("Czas od startu badania [s]", fontsize=16, fontweight="bold")
    ax.set_ylabel("Temperatura [deg C]", fontsize=16, fontweight="bold")
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(True, which="major", color="#d9d9d9", linewidth=0.8, alpha=0.9)
    ax.grid(True, which="minor", color="#efefef", linewidth=0.5, alpha=0.8)
    ax.minorticks_on()

    legend = ax.legend(loc="upper right", fontsize=13, frameon=True)
    if legend is not None:
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("#cccccc")
        legend.get_frame().set_alpha(0.95)

    finite_arrays = [series_data[name][np.isfinite(series_data[name])] for name in series_names]
    finite_arrays = [arr for arr in finite_arrays if arr.size > 0]
    if finite_arrays:
        finite_all = np.concatenate(finite_arrays)
        y_min = float(np.min(finite_all))
        y_max = float(np.max(finite_all))
        pad = max(0.08, (y_max - y_min) * 0.12)
        ax.set_ylim(y_min - pad, y_max + pad)

    ax.margins(x=0.01)
    plt.tight_layout()

    if output_png is not None:
        fig.savefig(output_png, bbox_inches="tight")
    if output_svg is not None:
        fig.savefig(output_svg, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    csv_path = Path(args.input_csv).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    first_row, rows = _load_rows(csv_path)
    rows = _filter_rows(rows, from_sec=args.from_sec, to_sec=args.to_sec)
    x, series_data = _collect_series(rows, args.series)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = "_".join(args.series)
    stem = f"{csv_path.stem}_wykres_{suffix}"
    output_png = output_dir / f"{stem}.png" if "png" in args.formats else None
    output_svg = output_dir / f"{stem}.svg" if "svg" in args.formats else None

    title = args.title.strip() or _build_title(first_row, args.from_sec, args.to_sec)
    _render_plot(
        x=x,
        series_names=args.series,
        series_data=series_data,
        title=title,
        output_png=output_png,
        output_svg=output_svg,
        dpi=max(72, int(args.dpi)),
        show_range_band=args.show_range_band,
    )

    if output_png is not None:
        print(output_png)
    if output_svg is not None:
        print(output_svg)


if __name__ == "__main__":
    main()
