"""CLI headless de fine-tuning: corre el benchmark completo para un modelo del registry.

Es el mismo pipeline que los notebooks (seccion 0.3 en adelante), pero sin celdas: pensado para
una corrida desatendida en Kaggle, en una VM, o en Colab con la sesion en segundo plano.
Toda la logica vive en ``benchmark.run_benchmark``; aca solo se parsea la linea de comando.

    python -m vit_for_101_food_app.modeling.train --model swin
    python -m vit_for_101_food_app.modeling.train --model mobilevit --resume
"""

from pathlib import Path

from loguru import logger
import typer

from vit_for_101_food_app.config import MODELS
from vit_for_101_food_app.modeling import benchmark

app = typer.Typer()


@app.command()
def main(
    model: str = typer.Option(..., "--model", "-m", help=f"clave del registry: {sorted(MODELS)}"),
    resume: bool = typer.Option(False, help="retomar desde el ultimo checkpoint de output-dir"),
    output_dir: Path = typer.Option(None, help="checkpoints (default: models/<model>/full)"),
    results_dir: Path = typer.Option(None, help="metricas y CSV (default: reports/results/<model>)"),
    num_workers: int = typer.Option(None, help="workers del DataLoader (default: min(4, nproc))"),
):
    if model not in MODELS:
        raise typer.BadParameter(f"{model!r} no esta en config.MODELS; disponibles: {sorted(MODELS)}")

    metrics = benchmark.run_benchmark(
        model,
        resume=resume,
        output_dir=output_dir,
        results_dir=results_dir,
        num_workers=num_workers,
    )
    logger.success(
        f"{model}: F1 macro val {metrics['best_val_f1_macro']:.4f} | "
        f"test accuracy {metrics['test']['accuracy']:.4f}"
    )


if __name__ == "__main__":
    app()
