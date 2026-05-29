from __future__ import annotations

import argparse
import json
from datetime import datetime
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

from src.train_lstm_autoencoder import LSTMAutoencoder


DEFAULT_DATASET_DIR = Path("Dane_przygotowane") / "lstm_autoencoder" / "M1"
DEFAULT_OUTPUT_DIR = Path("runs") / "netron"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Eksportuje LSTM Autoencoder do plikow wygodnych do podgladu w Netron."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Folder runu z metrics.json i opcjonalnie model_best.pt. "
            "Gdy podany, skrypt odtwarza konfiguracje modelu z metryk."
        ),
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=(
            "Folder z dataset.npz uzywany do wyznaczenia dlugosci sekwencji "
            "i liczby cech, gdy nie podano --run-dir. Domyslnie: "
            f"{DEFAULT_DATASET_DIR}"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Opcjonalny plik state_dict .pt. Domyslnie: model_best.pt z --run-dir, "
            "jesli istnieje. Bez checkpointu eksportowana jest sama architektura "
            "z losowymi wagami."
        ),
    )
    parser.add_argument(
        "--skip-checkpoint",
        action="store_true",
        help="Nie laduj wag nawet wtedy, gdy checkpoint jest dostepny.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Folder wynikowy. Domyslnie: <run-dir>/netron albo runs/netron, "
            "gdy --run-dir nie jest podany."
        ),
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Bazowa nazwa plikow wynikowych. Domyslnie generowana automatycznie.",
    )
    parser.add_argument(
        "--format",
        choices=["torchscript", "onnx", "both"],
        default="both",
        help=(
            "Format modelu dla Netron. TorchScript nie wymaga dodatkowych paczek; "
            "ONNX wymaga pakietu onnx. Domyslnie: both."
        ),
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="Wersja ONNX opset. Domyslnie: 17.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Rozmiar przykladowego batcha uzytego przy eksporcie. Domyslnie: 1.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Dlugosc sekwencji. Nadpisuje wartosc z metrics.json/dataset.npz.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=None,
        help="Liczba cech wejsciowych. Nadpisuje wartosc z metrics.json/dataset.npz.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=None,
        help="Rozmiar ukryty LSTM. Nadpisuje wartosc z metrics.json/checkpointu.",
    )
    parser.add_argument(
        "--latent-size",
        type=int,
        default=None,
        help="Rozmiar bottleneck. Nadpisuje wartosc z metrics.json/checkpointu.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Liczba warstw LSTM. Nadpisuje wartosc z metrics.json/checkpointu.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Dropout LSTM. Nadpisuje wartosc z metrics.json.",
    )
    parser.add_argument(
        "--feature-names",
        default=None,
        help="Opcjonalna lista cech po przecinku. Nadpisuje metryki/dataset.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_dataset_metadata(dataset_dir: Path) -> dict[str, Any]:
    dataset_path = dataset_dir / "dataset.npz"
    if not dataset_path.exists():
        return {}

    loaded = np.load(dataset_path)
    x_train = loaded["X_train"]
    return {
        "dataset_dir": str(dataset_dir),
        "seq_len": int(x_train.shape[1]),
        "input_size": int(x_train.shape[2]),
        "feature_names": [str(item) for item in loaded["feature_names"].tolist()],
        "dataset_shapes": {
            key: [int(dim) for dim in loaded[key].shape]
            for key in (
                "X_train",
                "X_val",
                "X_test_normal",
                "X_test_anomaly",
            )
            if key in loaded
        },
    }


def _load_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")

    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(f"Nieobslugiwany format checkpointu: {path}")
    return payload


def _infer_config_from_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, int]:
    inferred: dict[str, int] = {}
    encoder_input = state_dict.get("encoder.weight_ih_l0")
    encoder_hidden = state_dict.get("encoder.weight_hh_l0")
    latent_weight = state_dict.get("to_latent.weight")

    if encoder_input is not None:
        inferred["input_size"] = int(encoder_input.shape[1])
    if encoder_hidden is not None:
        inferred["hidden_size"] = int(encoder_hidden.shape[1])
    if latent_weight is not None:
        inferred["latent_size"] = int(latent_weight.shape[0])

    layer_indexes = []
    prefix = "encoder.weight_ih_l"
    for key in state_dict:
        if key.startswith(prefix):
            suffix = key[len(prefix) :]
            if suffix.isdigit():
                layer_indexes.append(int(suffix))
    if layer_indexes:
        inferred["num_layers"] = max(layer_indexes) + 1

    return inferred


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _resolve_model_context(args: argparse.Namespace) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    dataset_meta: dict[str, Any] = {}
    state_dict: dict[str, torch.Tensor] | None = None
    checkpoint_path: Path | None = None

    if args.run_dir is not None:
        metrics_path = args.run_dir / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Nie znaleziono metrics.json: {metrics_path}")
        metrics = _read_json(metrics_path)

    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
    elif args.run_dir is not None and (args.run_dir / "model_best.pt").exists():
        checkpoint_path = args.run_dir / "model_best.pt"

    if checkpoint_path is not None and not args.skip_checkpoint:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Nie znaleziono checkpointu: {checkpoint_path}")
        state_dict = _load_checkpoint(checkpoint_path)

    if args.dataset_dir is not None:
        dataset_meta = _load_dataset_metadata(args.dataset_dir)

    inferred = _infer_config_from_state_dict(state_dict) if state_dict else {}
    training_cfg = metrics.get("training_config", {})
    dataset_shapes = metrics.get("dataset_shapes") or dataset_meta.get("dataset_shapes") or {}
    x_train_shape = dataset_shapes.get("X_train", [])

    feature_names = (
        [item.strip() for item in args.feature_names.split(",") if item.strip()]
        if args.feature_names
        else metrics.get("feature_names") or dataset_meta.get("feature_names")
    )

    seq_len = _first_present(
        args.seq_len,
        int(x_train_shape[1]) if len(x_train_shape) >= 2 else None,
        dataset_meta.get("seq_len"),
    )
    input_size = _first_present(
        args.input_size,
        int(x_train_shape[2]) if len(x_train_shape) >= 3 else None,
        dataset_meta.get("input_size"),
        inferred.get("input_size"),
        len(feature_names) if feature_names else None,
    )
    hidden_size = _first_present(
        args.hidden_size,
        training_cfg.get("hidden_size"),
        inferred.get("hidden_size"),
        64,
    )
    latent_size = _first_present(
        args.latent_size,
        training_cfg.get("latent_size"),
        inferred.get("latent_size"),
        32,
    )
    num_layers = _first_present(
        args.num_layers,
        training_cfg.get("num_layers"),
        inferred.get("num_layers"),
        1,
    )
    dropout = _first_present(args.dropout, training_cfg.get("dropout"), 0.0)

    missing = [
        name
        for name, value in {
            "seq_len": seq_len,
            "input_size": input_size,
            "hidden_size": hidden_size,
            "latent_size": latent_size,
            "num_layers": num_layers,
            "dropout": dropout,
        }.items()
        if value is None
    ]
    if missing:
        raise ValueError(
            "Nie udalo sie ustalic parametrow modelu: "
            + ", ".join(missing)
            + ". Podaj je argumentami CLI albo uzyj --run-dir/--dataset-dir."
        )

    if not feature_names:
        feature_names = [f"feature_{idx}" for idx in range(int(input_size))]
    if len(feature_names) != int(input_size):
        raise ValueError(
            "Liczba feature_names nie zgadza sie z input_size: "
            f"{len(feature_names)} != {input_size}"
        )

    return {
        "metrics": metrics,
        "dataset_meta": dataset_meta,
        "state_dict": state_dict,
        "checkpoint_path": checkpoint_path if state_dict is not None else None,
        "seq_len": int(seq_len),
        "input_size": int(input_size),
        "hidden_size": int(hidden_size),
        "latent_size": int(latent_size),
        "num_layers": int(num_layers),
        "dropout": float(dropout),
        "feature_names": list(feature_names),
        "dataset_shapes": dataset_shapes,
    }


def _build_model(context: dict[str, Any]) -> LSTMAutoencoder:
    model = LSTMAutoencoder(
        input_size=context["input_size"],
        hidden_size=context["hidden_size"],
        latent_size=context["latent_size"],
        num_layers=context["num_layers"],
        dropout=context["dropout"],
    )
    state_dict = context["state_dict"]
    if state_dict is not None:
        model.load_state_dict(state_dict)
    model.eval()
    return model


def _make_output_base(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        output_dir = args.output_dir
    elif args.run_dir is not None:
        output_dir = args.run_dir / "netron"
    else:
        output_dir = DEFAULT_OUTPUT_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.name:
        stem = args.name
    elif args.run_dir is not None:
        stem = args.run_dir.name
    else:
        stem = "lstm_autoencoder_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / stem


def _export_torchscript(
    model: LSTMAutoencoder,
    dummy_input: torch.Tensor,
    output_path: Path,
) -> None:
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy_input, strict=False)
        traced.save(str(output_path))


def _export_onnx(
    model: LSTMAutoencoder,
    dummy_input: torch.Tensor,
    output_path: Path,
    opset: int,
) -> None:
    try:
        import onnx  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Eksport ONNX wymaga pakietu 'onnx'. Zainstaluj go poleceniem: "
            "pip install -r requirements-ml.txt"
        ) from exc

    with torch.no_grad():
        export_kwargs: dict[str, Any] = {
            "export_params": True,
            "opset_version": opset,
            "do_constant_folding": True,
            "input_names": ["input_window"],
            "output_names": ["reconstructed_window"],
            "dynamic_axes": {
                "input_window": {0: "batch", 1: "sequence"},
                "reconstructed_window": {0: "batch", 1: "sequence"},
            },
        }
        if hasattr(torch.onnx.export, "__code__") and (
            "dynamo" in torch.onnx.export.__code__.co_varnames
        ):
            export_kwargs["dynamo"] = False
        torch.onnx.export(model, dummy_input, output_path, **export_kwargs)


def _parameter_counts(model: LSTMAutoencoder) -> dict[str, Any]:
    modules = {
        name: int(sum(parameter.numel() for parameter in module.parameters()))
        for name, module in model.named_children()
    }
    total = int(sum(parameter.numel() for parameter in model.parameters()))
    trainable = int(
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    )
    return {
        "total": total,
        "trainable": trainable,
        "modules": modules,
    }


def _architecture_payload(
    args: argparse.Namespace,
    context: dict[str, Any],
    model: LSTMAutoencoder,
    exported_files: dict[str, str],
) -> dict[str, Any]:
    seq_len = context["seq_len"]
    input_size = context["input_size"]
    hidden_size = context["hidden_size"]
    latent_size = context["latent_size"]
    num_layers = context["num_layers"]
    dropout = context["dropout"]

    return {
        "note": (
            "Netron otwiera plik .pt TorchScript albo .onnx. Ten JSON jest "
            "metadanymi opisujacymi architekture, nie samodzielnym modelem Netron."
        ),
        "generated_by": "src.export_lstm_autoencoder_netron",
        "source_run_dir": str(args.run_dir) if args.run_dir else None,
        "checkpoint": (
            str(context["checkpoint_path"]) if context["checkpoint_path"] else None
        ),
        "dataset_dir": (
            context["metrics"].get("dataset_dir")
            or context["dataset_meta"].get("dataset_dir")
            or str(args.dataset_dir)
        ),
        "netron_files": exported_files,
        "model": {
            "class_name": "LSTMAutoencoder",
            "input_shape": ["batch", seq_len, input_size],
            "output_shape": ["batch", seq_len, input_size],
            "hidden_size": hidden_size,
            "latent_size": latent_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "recurrent_dropout": dropout if num_layers > 1 else 0.0,
            "parameter_counts": _parameter_counts(model),
        },
        "input": {
            "name": "input_window",
            "shape": ["batch", seq_len, input_size],
            "feature_names": context["feature_names"],
        },
        "output": {
            "name": "reconstructed_window",
            "shape": ["batch", seq_len, input_size],
        },
        "graph_flow": [
            {
                "name": "encoder",
                "type": "torch.nn.LSTM",
                "input": ["batch", seq_len, input_size],
                "output": ["batch", seq_len, hidden_size],
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "batch_first": True,
            },
            {
                "name": "context",
                "type": "select_last_timestep",
                "input": ["batch", seq_len, hidden_size],
                "output": ["batch", hidden_size],
            },
            {
                "name": "to_latent",
                "type": "torch.nn.Linear",
                "input": ["batch", hidden_size],
                "output": ["batch", latent_size],
            },
            {
                "name": "from_latent",
                "type": "torch.nn.Linear",
                "input": ["batch", latent_size],
                "output": ["batch", hidden_size],
            },
            {
                "name": "repeat_context",
                "type": "unsqueeze_repeat",
                "input": ["batch", hidden_size],
                "output": ["batch", seq_len, hidden_size],
            },
            {
                "name": "decoder",
                "type": "torch.nn.LSTM",
                "input": ["batch", seq_len, hidden_size],
                "output": ["batch", seq_len, hidden_size],
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "batch_first": True,
            },
            {
                "name": "output_projection",
                "type": "torch.nn.Linear",
                "input": ["batch", seq_len, hidden_size],
                "output": ["batch", seq_len, input_size],
            },
        ],
        "dataset_shapes": context["dataset_shapes"],
        "training_config": context["metrics"].get("training_config"),
        "threshold": context["metrics"].get("threshold"),
    }


def main() -> None:
    args = _parse_args()
    context = _resolve_model_context(args)
    model = _build_model(context)
    dummy_input = torch.zeros(
        args.batch_size,
        context["seq_len"],
        context["input_size"],
        dtype=torch.float32,
    )
    output_base = _make_output_base(args)
    exported_files: dict[str, str] = {}

    if args.format in {"torchscript", "both"}:
        torchscript_path = output_base.with_suffix(".torchscript.pt")
        _export_torchscript(model, dummy_input, torchscript_path)
        exported_files["torchscript"] = str(torchscript_path)

    if args.format in {"onnx", "both"}:
        onnx_path = output_base.with_suffix(".onnx")
        try:
            _export_onnx(model, dummy_input, onnx_path, args.opset)
        except ModuleNotFoundError:
            if args.format == "onnx":
                raise
            print("Pominieto ONNX: zainstaluj pakiet 'onnx', aby utworzyc .onnx.")
        else:
            exported_files["onnx"] = str(onnx_path)

    metadata_path = output_base.with_suffix(".metadata.json")
    exported_files = {**exported_files, "metadata": str(metadata_path)}
    metadata_payload = _architecture_payload(args, context, model, exported_files)
    metadata_path.write_text(
        json.dumps(metadata_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Wygenerowano pliki:")
    for kind, path in exported_files.items():
        print(f"- {kind}: {path}")


if __name__ == "__main__":
    main()
