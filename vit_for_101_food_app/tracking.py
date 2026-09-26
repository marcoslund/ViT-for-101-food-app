"""Registro en MLflow de los resultados del benchmark a partir de ``reports/results``.

python -m vit_for_101_food_app.tracking
mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import hashlib
import json
import math
import os
from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from vit_for_101_food_app.config import FIGURES_DIR, PROJ_ROOT, REPORTS_DIR

app = typer.Typer()

RESULTS_DIR = REPORTS_DIR / "results"
EXPERIMENT = "food101-benchmark"
TRACKING_URI = f"sqlite:///{(PROJ_ROOT / 'mlflow.db').as_posix()}"
ARTIFACT_ROOT = PROJ_ROOT / "mlruns"


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}/{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out


def split_values(metrics: dict) -> tuple[dict, dict, dict]:
    """Separa un ``metrics.json`` en params (receta), metricas numericas y tags."""
    params, numeric, tags = {}, {}, {}
    for key, value in flatten(metrics).items():
        if key.startswith("receta/"):
            params[key.removeprefix("receta/")] = value
        elif isinstance(value, (bool, str)):
            tags[key] = str(value)
        elif isinstance(value, (int, float)) and math.isfinite(value):
            numeric[key] = float(value)
    return params, numeric, tags


def history_metrics(history: pd.DataFrame) -> list[tuple[str, float, int]]:
    """Convierte ``training_history.csv`` en ``(clave, valor, paso)``, con los tags de TensorBoard."""
    rows = []
    for _, row in history.iterrows():
        step = int(row["step"])
        for col, value in row.items():
            if col in ("step", "epoch", "split") or pd.isna(value):
                continue
            key = f"eval/{col[5:]}" if col.startswith("eval_") else f"train/{col}"
            rows.append((key, float(value), step))
    return rows


def source_files(model_key: str) -> list[Path]:
    results = RESULTS_DIR / model_key
    files = sorted(p for p in results.iterdir() if p.is_file())
    files += sorted(FIGURES_DIR.glob(f"curvas-{model_key}-*.png"))
    return files


def fingerprint(files: list[Path]) -> str:
    h = hashlib.sha256()
    for p in files:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def log_model(client, experiment_id: str, model_key: str, force: bool = False) -> str | None:
    """Crea el run de ``model_key``; si ya existe con los mismos archivos, no hace nada."""
    from mlflow.entities import Metric, Param

    files = source_files(model_key)
    huella = fingerprint(files)
    previos = client.search_runs([experiment_id], filter_string=f"tags.model_key = '{model_key}'")
    if not force and any(r.data.tags.get("fuente_sha") == huella for r in previos):
        logger.info(f"{model_key}: sin cambios, se mantiene el run existente")
        return None
    for r in previos:
        client.delete_run(r.info.run_id)

    metrics = json.loads((RESULTS_DIR / model_key / "metrics.json").read_text(encoding="utf-8"))
    params, numeric, tags = split_values(metrics)
    tags.update({"model_key": model_key, "fuente_sha": huella})

    run = client.create_run(experiment_id, run_name=model_key, tags=tags)
    run_id = run.info.run_id
    batch = [Metric(k, v, 0, 0) for k, v in numeric.items()]
    history_path = RESULTS_DIR / model_key / "training_history.csv"
    if history_path.exists():
        batch += [Metric(k, v, 0, s) for k, v, s in history_metrics(pd.read_csv(history_path))]
    client.log_batch(run_id, params=[Param(k, str(v)) for k, v in params.items()])
    for i in range(0, len(batch), 1000):
        client.log_batch(run_id, metrics=batch[i : i + 1000])
    for p in files:
        client.log_artifact(
            run_id, str(p), artifact_path="figuras" if p.suffix == ".png" else None
        )
    client.set_terminated(run_id)
    logger.success(f"{model_key}: run {run_id} ({len(batch)} metricas, {len(files)} archivos)")
    return run_id


@app.command()
def main(
    model: list[str] = typer.Option(
        None, "--model", "-m", help="default: todos los de reports/results"
    ),
    force: bool = typer.Option(False, help="recrear los runs aunque no haya cambios"),
):
    import mlflow
    from mlflow.tracking import MlflowClient

    uri = os.environ.get("MLFLOW_TRACKING_URI", TRACKING_URI)
    mlflow.set_tracking_uri(uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        ARTIFACT_ROOT.mkdir(exist_ok=True)
        experiment_id = client.create_experiment(EXPERIMENT, ARTIFACT_ROOT.as_uri())
    else:
        experiment_id = experiment.experiment_id
    client.set_experiment_tag(experiment_id, "mlflow.experimentKind", "custom_model_development")

    modelos = model or sorted(p.parent.name for p in RESULTS_DIR.glob("*/metrics.json"))
    for m in modelos:
        log_model(client, experiment_id, m, force=force)
    logger.info(f"para verlos: mlflow ui --backend-store-uri {uri}")


if __name__ == "__main__":
    app()
