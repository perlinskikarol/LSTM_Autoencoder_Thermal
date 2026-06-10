from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Brak pakietu 'torch'. Zainstaluj zaleznosci ML poleceniem: "
        "pip install -r requirements-ml.txt"
    ) from exc


class LSTMAutoencoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        latent_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.to_latent = nn.Linear(hidden_size, latent_size)
        self.from_latent = nn.Linear(latent_size, hidden_size)
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.output_projection = nn.Linear(hidden_size, input_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded_sequence, _ = self.encoder(inputs)
        context = encoded_sequence[:, -1, :]
        latent = self.to_latent(context)
        repeated_context = self.from_latent(latent).unsqueeze(1).repeat(
            1, inputs.size(1), 1
        )
        decoded_sequence, _ = self.decoder(repeated_context)
        return self.output_projection(decoded_sequence)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Szkic treningu LSTM Autoencoder dla danych przygotowanych przez "
            "src.prepare_lstm_dataset."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        default="Dane_przygotowane\\lstm_autoencoder\\M1",
        help="Folder z dataset.npz i plikami metadanych. Domyslnie: M1",
    )
    parser.add_argument(
        "--output-dir",
        default="runs\\lstm_autoencoder",
        help="Folder bazowy na wyniki treningu. Domyslnie: runs\\lstm_autoencoder",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Opcjonalna nazwa uruchomienia. Domyslnie generowana automatycznie.",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Liczba epok. Domyslnie: 200")
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Rozmiar batcha. Domyslnie: 64"
    )
    parser.add_argument(
        "--hidden-size", type=int, default=64, help="Rozmiar ukryty LSTM. Domyslnie: 64"
    )
    parser.add_argument(
        "--latent-size", type=int, default=32, help="Rozmiar bottleneck. Domyslnie: 32"
    )
    parser.add_argument(
        "--num-layers", type=int, default=1, help="Liczba warstw LSTM. Domyslnie: 1"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.0, help="Dropout LSTM. Domyslnie: 0.0"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate. Domyslnie: 1e-3")
    parser.add_argument(
        "--weight-decay", type=float, default=1e-5, help="Weight decay. Domyslnie: 1e-5"
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience. Domyslnie: 10"
    )
    parser.add_argument(
        "--threshold-quantile",
        type=float,
        default=0.99,
        help="Percentyl bledu rekonstrukcji na val do progu anomalii. Domyslnie: 0.99",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed losowosci. Domyslnie: 42"
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Urządzenie do treningu. Domyslnie: auto",
    )
    parser.add_argument(
        "--max-train-windows",
        type=int,
        default=None,
        help="Opcjonalny limit liczby okien train do szybszych testow.",
    )
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Wybrano CUDA, ale torch nie widzi GPU.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_dataset(dataset_dir: Path, max_train_windows: int | None) -> dict[str, Any]:
    dataset_path = dataset_dir / "dataset.npz"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku dataset.npz: {dataset_path}")

    loaded = np.load(dataset_path)
    data = {
        "feature_names": [str(item) for item in loaded["feature_names"].tolist()],
        "X_train": loaded["X_train"].astype(np.float32),
        "X_val": loaded["X_val"].astype(np.float32),
        "X_test_normal": loaded["X_test_normal"].astype(np.float32),
        "X_test_anomaly": loaded["X_test_anomaly"].astype(np.float32),
    }

    if max_train_windows is not None and max_train_windows > 0:
        data["X_train"] = data["X_train"][:max_train_windows]

    if data["X_train"].shape[0] == 0:
        raise ValueError("Zbior train jest pusty.")

    if data["X_val"].shape[0] == 0 and data["X_train"].shape[0] >= 10:
        holdout = max(1, int(round(data["X_train"].shape[0] * 0.1)))
        data["X_val"] = data["X_train"][-holdout:].copy()
        data["X_train"] = data["X_train"][:-holdout].copy()

    data["dataset_summary"] = (
        _load_json(dataset_dir / "dataset_summary.json")
        if (dataset_dir / "dataset_summary.json").exists()
        else {}
    )
    data["window_index_rows"] = _load_window_index(dataset_dir / "window_index.csv")
    return data


def _load_window_index(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _make_loader(array: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensor = torch.from_numpy(array)
    dataset = TensorDataset(tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_samples = 0

    for (batch_inputs,) in loader:
        batch_inputs = batch_inputs.to(device)
        if is_train:
            optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_inputs)
        if is_train:
            loss.backward()
            optimizer.step()

        batch_size = batch_inputs.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


def _compute_reconstruction_errors(
    model: nn.Module,
    array: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if array.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)

    loader = _make_loader(array, batch_size=batch_size, shuffle=False)
    model.eval()
    all_errors: list[np.ndarray] = []
    with torch.no_grad():
        for (batch_inputs,) in loader:
            batch_inputs = batch_inputs.to(device)
            outputs = model(batch_inputs)
            errors = ((outputs - batch_inputs) ** 2).mean(dim=(1, 2))
            all_errors.append(errors.cpu().numpy().astype(np.float32))
    return np.concatenate(all_errors, axis=0)


def _save_history_csv(history_rows: list[dict[str, Any]], target_path: Path) -> None:
    with target_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(history_rows)


def _maybe_save_loss_plot(history_rows: list[dict[str, Any]], target_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return

    epochs = [row["epoch"] for row in history_rows]
    train_loss = [row["train_loss"] for row in history_rows]
    val_loss = [row["val_loss"] for row in history_rows]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label="train_loss", color="#0f766e", linewidth=2.5)
    plt.plot(epochs, val_loss, label="val_loss", color="#dc2626", linewidth=2.5)
    plt.xlabel("Epoka")
    plt.ylabel("MSE")
    plt.title("LSTM Autoencoder - przebieg uczenia")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(target_path, dpi=180)
    plt.close()


def _summarize_errors(errors: np.ndarray, threshold: float) -> dict[str, Any]:
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

    above = int((errors > threshold).sum())
    return {
        "count": int(errors.size),
        "mean_error": round(float(errors.mean()), 8),
        "median_error": round(float(np.median(errors)), 8),
        "p95_error": round(float(np.quantile(errors, 0.95)), 8),
        "max_error": round(float(errors.max()), 8),
        "above_threshold_count": above,
        "above_threshold_ratio": round(float(above / errors.size), 8),
    }


def _build_scores_rows(
    window_index_rows: list[dict[str, str]],
    split_errors: dict[str, np.ndarray],
    threshold: float,
) -> list[dict[str, Any]]:
    rows_by_split: dict[str, list[dict[str, str]]] = {}
    for row in window_index_rows:
        rows_by_split.setdefault(row["split"], []).append(row)

    output_rows: list[dict[str, Any]] = []
    for split, errors in split_errors.items():
        split_rows = sorted(
            rows_by_split.get(split, []),
            key=lambda item: int(item["array_index"]),
        )
        if len(split_rows) != int(errors.shape[0]):
            for idx, error in enumerate(errors.tolist()):
                output_rows.append(
                    {
                        "split": split,
                        "array_index": idx,
                        "session_key": "",
                        "source_file": "",
                        "subject_id": "",
                        "session_state": "",
                        "start_timestamp": "",
                        "end_timestamp": "",
                        "reconstruction_error": f"{float(error):.8f}",
                        "is_anomaly": "true" if float(error) > threshold else "false",
                    }
                )
            continue

        for row, error in zip(split_rows, errors.tolist(), strict=True):
            output_rows.append(
                {
                    "split": split,
                    "array_index": row["array_index"],
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
    return output_rows


def main() -> None:
    args = _parse_args()
    _set_seed(args.seed)
    device = _resolve_device(args.device)

    dataset_dir = Path(args.dataset_dir)
    output_root = Path(args.output_dir)
    dataset = _load_dataset(dataset_dir, max_train_windows=args.max_train_windows)

    subject_id = (
        dataset["dataset_summary"].get("subject_id")
        or dataset_dir.name
        or "unknown_subject"
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{subject_id}_{timestamp}"
    run_dir = output_root / subject_id / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    x_train = dataset["X_train"]
    x_val = dataset["X_val"]
    x_test_normal = dataset["X_test_normal"]
    x_test_anomaly = dataset["X_test_anomaly"]
    input_size = x_train.shape[2]

    model = LSTMAutoencoder(
        input_size=input_size,
        hidden_size=args.hidden_size,
        latent_size=args.latent_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.MSELoss()

    train_loader = _make_loader(x_train, batch_size=args.batch_size, shuffle=True)
    val_loader = _make_loader(x_val, batch_size=args.batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history_rows: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = _run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = _run_epoch(model, val_loader, criterion, None, device)
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": round(float(train_loss), 8),
                "val_loss": round(float(val_loss), 8),
            }
        )
        print(
            f"[epoch {epoch:03d}] train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f}"
        )

        if val_loss < best_val_loss - 1e-7:
            best_val_loss = float(val_loss)
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), run_dir / "model_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    model.load_state_dict(
        torch.load(run_dir / "model_best.pt", map_location=device, weights_only=True)
    )

    val_errors = _compute_reconstruction_errors(
        model, x_val, device=device, batch_size=args.batch_size
    )
    threshold = float(np.quantile(val_errors, args.threshold_quantile))

    split_errors = {
        "train": _compute_reconstruction_errors(
            model, x_train, device=device, batch_size=args.batch_size
        ),
        "val": val_errors,
        "test_normal": _compute_reconstruction_errors(
            model, x_test_normal, device=device, batch_size=args.batch_size
        ),
        "test_anomaly": _compute_reconstruction_errors(
            model, x_test_anomaly, device=device, batch_size=args.batch_size
        ),
    }

    scores_rows = _build_scores_rows(
        window_index_rows=dataset["window_index_rows"],
        split_errors=split_errors,
        threshold=threshold,
    )
    with (run_dir / "scores_by_window.csv").open("w", encoding="utf-8", newline="") as handle:
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
        writer.writerows(scores_rows)

    metrics_payload = {
        "subject_id": subject_id,
        "dataset_dir": str(dataset_dir),
        "run_dir": str(run_dir),
        "device": str(device),
        "feature_names": dataset["feature_names"],
        "dataset_shapes": {
            "X_train": list(x_train.shape),
            "X_val": list(x_val.shape),
            "X_test_normal": list(x_test_normal.shape),
            "X_test_anomaly": list(x_test_anomaly.shape),
        },
        "training_config": {
            "epochs_requested": args.epochs,
            "batch_size": args.batch_size,
            "hidden_size": args.hidden_size,
            "latent_size": args.latent_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "threshold_quantile": args.threshold_quantile,
            "seed": args.seed,
        },
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 8),
        "threshold": {
            "quantile": args.threshold_quantile,
            "reconstruction_error_threshold": round(threshold, 8),
        },
        "split_metrics": {
            split_name: _summarize_errors(errors, threshold)
            for split_name, errors in split_errors.items()
        },
    }

    if split_errors["test_normal"].size > 0 and split_errors["test_anomaly"].size > 0:
        normal_fpr = float((split_errors["test_normal"] > threshold).mean())
        anomaly_tpr = float((split_errors["test_anomaly"] > threshold).mean())
        metrics_payload["detection_preview"] = {
            "normal_false_positive_rate": round(normal_fpr, 8),
            "anomaly_true_positive_rate": round(anomaly_tpr, 8),
        }

    (run_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2),
        encoding="utf-8",
    )
    _save_history_csv(history_rows, run_dir / "history.csv")
    _maybe_save_loss_plot(history_rows, run_dir / "loss_curve.png")

    print(f"Training artifacts saved to: {run_dir}")
    print(json.dumps(metrics_payload, indent=2))


if __name__ == "__main__":
    main()
