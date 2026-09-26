"""CLI de inferencia: evalua un modelo YA entrenado sobre el test, sin reentrenar.

Sirve para un checkpoint descargado (p. ej. el ``best/`` que baja un notebook, o su .zip
descomprimido): reconstruye las predicciones y las metricas por tercil sin volver a pasar
por el fine-tuning. Reusa el mismo protocolo de evaluacion que el benchmark, asi las
metricas son comparables con las de una corrida completa.

    python -m vit_for_101_food_app.modeling.predict --model swin --model-dir models/swin/full/best
"""

from pathlib import Path

from loguru import logger
import numpy as np
import torch
import typer

from vit_for_101_food_app.config import FOOD101_META_DIR, MODELS, TRAIN_VAL_SPLIT
from vit_for_101_food_app.modeling import benchmark, evaluation
from vit_for_101_food_app.preprocessing import loaders, splits

app = typer.Typer()


def collect_logits(model, dataloader, device: torch.device) -> np.ndarray:
    """Corre el modelo sobre el dataloader y devuelve los logits en orden."""
    model.eval().to(device)
    salida = []
    with torch.no_grad():
        for batch in dataloader:
            logits = model(pixel_values=batch["pixel_values"].to(device)).logits
            salida.append(logits.float().cpu().numpy())
    return np.concatenate(salida)


@app.command()
def main(
    model: str = typer.Option(..., "--model", "-m", help=f"clave del registry: {sorted(MODELS)}"),
    model_dir: Path = typer.Option(..., help="carpeta del modelo fine-tuneado (from_pretrained)"),
    results_dir: Path = typer.Option(None, help="donde escribir los CSV (default: reports/results/<model>)"),
    num_workers: int = typer.Option(None, help="workers del DataLoader (default: min(4, nproc))"),
):
    if model not in MODELS:
        raise typer.BadParameter(f"{model!r} no esta en config.MODELS; disponibles: {sorted(MODELS)}")

    from transformers import AutoModelForImageClassification

    id2label, _ = benchmark.verify_artifacts()
    results_dir = results_dir or benchmark.default_dirs(model)[1]
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = AutoModelForImageClassification.from_pretrained(str(model_dir))

    dls = loaders.build_dataloaders(
        model,
        source="cache",
        num_workers=num_workers if num_workers is not None else 0,
        splits_to_load=("test",),
    )
    logits = collect_logits(net, dls["test"], device)

    frame = splits.load_split("test", csv_path=TRAIN_VAL_SPLIT, meta_dir=FOOD101_META_DIR)
    preds = evaluation.predictions_frame(frame, logits, id2label)
    preds.to_csv(Path(results_dir) / "predictions_test.csv", index=False)

    subset = evaluation.load_benchmark_subset()
    benchmark_tabla = evaluation.benchmark_metrics(preds, subset)
    benchmark_tabla.to_csv(Path(results_dir) / "benchmark_por_tercil.csv")

    metrics = evaluation.split_metrics(preds)
    logger.success(
        f"{model}: test accuracy {metrics['accuracy']:.4f} | F1 macro {metrics['f1_macro']:.4f} "
        f"-> {results_dir}"
    )


if __name__ == "__main__":
    app()
