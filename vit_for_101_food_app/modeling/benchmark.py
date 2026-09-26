"""Orquestacion del benchmark: une preprocessing, training y evaluation en un pipeline.

Los notebooks (Colab/Kaggle, celda por celda) y el CLI headless (``train.py``) llaman a
las MISMAS funciones de aca, asi que una corrida desatendida y una interactiva miden
exactamente lo mismo. Todo es generico sobre ``model_key`` (una clave de ``config.MODELS``):
sumar una arquitectura al benchmark no toca este archivo.

La regla del proyecto sigue valiendo: nada de resoluciones ni recetas escritas a mano.
La geometria sale del ``AutoImageProcessor`` (via ``training.resolve_batch_plan``) y la
receta de optimizacion de ``training.TrainingRecipe``, la misma para todos los modelos.
"""

import json
from pathlib import Path

from loguru import logger
import pandas as pd

from vit_for_101_food_app import config, plots
from vit_for_101_food_app.modeling import evaluation, training
from vit_for_101_food_app.preprocessing import cache, loaders, splits

REQUIRED_ARTIFACTS = (
    config.TRAIN_VAL_SPLIT,
    config.TRAIN_VAL_MANIFEST,
    config.LABEL_MAP,
    config.CACHE_DIR / "cache_manifest.json",
)


def verify_artifacts(
    label_map_path: Path = config.LABEL_MAP,
    cache_dir: Path = config.CACHE_DIR,
    short_side: int = config.CACHE_SHORT_SIDE,
    expected_num_labels: int | None = 101,
) -> tuple[dict[int, str], dict[str, int]]:
    """Falla temprano si el preprocessing no esta listo; devuelve ``(id2label, label2id)``.

    Comprueba que existan los artefactos versionados y que el cache coincida con la
    resolucion configurada. Sin esto, un split o un cache desactualizado harian que el
    modelo entrene sobre datos distintos a los de otra corrida, sin que nada falle.
    """
    faltan = [Path(p) for p in REQUIRED_ARTIFACTS if not Path(p).exists()]
    if faltan:
        raise FileNotFoundError(
            "Faltan artefactos del preprocessing:\n"
            + "\n".join(f" - {p}" for p in faltan)
            + "\n\nEjecuta primero `make preprocess` (o corre la seccion 0.3 en Colab)."
        )
    if not cache.cache_is_valid(cache_dir=cache_dir, short_side=short_side):
        raise RuntimeError(
            f"El cache en {cache_dir} existe pero no coincide con CACHE_SHORT_SIDE="
            f"{short_side}. Regeneralo con `make cache` (o corre la seccion 0.3 en Colab)."
        )

    id2label, label2id = splits.load_label_map(label_map_path)
    if expected_num_labels is not None and len(label2id) != expected_num_labels:
        raise ValueError(
            f"el label_map tiene {len(label2id)} clases, se esperaban {expected_num_labels}"
        )
    return id2label, label2id


def assemble_metrics(
    model_key: str,
    recipe: training.TrainingRecipe,
    trainer,
    elapsed_train: float,
    test_metrics: dict,
    benchmark: pd.DataFrame,
    costo: dict,
    latencia: dict,
) -> dict:
    """Arma el dict de metricas que se serializa a ``metrics.json``.

    Es el registro comparable entre modelos: receta, tiempo, F1 de validacion, metricas
    de test, tabla por tercil, costo arquitectonico, latencia y versiones del entorno.
    """
    return {
        "model_key": model_key,
        "checkpoint": config.MODELS[model_key],
        "receta": {
            **training.recipe_as_dict(recipe),
            "epochs_entrenadas": trainer.state.epoch,
        },
        "tiempo_entrenamiento_s": elapsed_train,
        "best_val_f1_macro": trainer.state.best_metric,
        "test": test_metrics,
        "benchmark_por_tercil": benchmark.to_dict(orient="index"),
        "costo": costo,
        "latencia": latencia,
        "entorno": training.software_versions(),
    }


def save_metrics(metrics: dict, results_dir: Path) -> Path:
    """Escribe ``metrics.json`` en ``results_dir`` y devuelve su ruta."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    destino = results_dir / "metrics.json"
    destino.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return destino


def default_dirs(model_key: str) -> tuple[Path, Path]:
    """Rutas por defecto de checkpoints y resultados para ``model_key``.

    En Colab con Drive los notebooks las sobreescriben apuntando a Drive; el CLI headless
    (Kaggle, local) usa estas.
    """
    return (
        config.MODELS_DIR / model_key / "full",
        config.REPORTS_DIR / "results" / model_key,
    )


def run_benchmark(
    model_key: str,
    resume: bool = False,
    output_dir: Path | None = None,
    results_dir: Path | None = None,
    num_workers: int | None = None,
    train_policy: str = "standard",
    source: str = "cache",
) -> dict:
    """Pipeline completo y headless: entrena, evalua y guarda todos los artefactos.

    Es la version desatendida de los notebooks (misma secuencia, mismas funciones), pensada
    para Kaggle o una corrida larga de Colab sin supervisar. Escribe checkpoints en
    ``output_dir``, y en ``results_dir`` los CSV de predicciones, reporte por clase,
    confusiones, tabla por tercil y ``metrics.json``. Devuelve el dict de metricas.
    """
    id2label, label2id = verify_artifacts()
    if output_dir is None or results_dir is None:
        default_output, default_results = default_dirs(model_key)
        output_dir = output_dir or default_output
        results_dir = results_dir or default_results
    output_dir, results_dir = Path(output_dir), Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    training.guard_output_dir(output_dir, resume)

    plan = training.resolve_batch_plan(model_key, num_workers=num_workers)
    logger.info(
        f"{model_key}: resolucion {plan.target_size} | batch {plan.batch_size} x "
        f"{plan.grad_accum} = {plan.effective_batch} | checkpointing {plan.gradient_checkpointing}"
    )

    dls = loaders.build_dataloaders(
        model_key,
        train_policy=train_policy,
        source=source,
        batch_size=plan.batch_size,
        num_workers=plan.num_workers,
        splits_to_load=("train", "val", "test"),
    )
    train_ds, val_ds, test_ds = (dls[s].dataset for s in ("train", "val", "test"))

    model = training.build_model(model_key, id2label, label2id)
    recipe = training.TrainingRecipe(batch_size=plan.batch_size, grad_accum_steps=plan.grad_accum)
    trainer = training.build_trainer(
        model,
        recipe,
        output_dir=output_dir,
        train_dataset=train_ds,
        val_dataset=val_ds,
        num_workers=plan.num_workers,
        gradient_checkpointing=plan.gradient_checkpointing,
    )

    elapsed_train = training.run_training(trainer, resume=resume)
    training.save_best_model(trainer, output_dir)

    log_history = trainer.state.log_history
    plots.history_table(log_history).to_csv(results_dir / "training_history.csv", index=False)
    plots.plot_training_curves(
        log_history,
        f"{model_key} - {config.MODELS[model_key]}",
        results_dir,
        prefix=f"curvas-{model_key}",
    )

    test_preds, test_metrics, _ = evaluation.evaluate_test(trainer, test_ds, id2label)
    test_preds.to_csv(results_dir / "predictions_test.csv", index=False)
    evaluation.per_class_report(test_preds, id2label).to_csv(
        results_dir / "report_por_clase_test.csv"
    )
    evaluation.frequent_confusions(test_preds).to_csv(
        results_dir / "confusiones_test.csv", index=False
    )

    subset = evaluation.load_benchmark_subset()
    benchmark = evaluation.benchmark_metrics(test_preds, subset)
    benchmark.to_csv(results_dir / "benchmark_por_tercil.csv")

    sample = test_ds[0]["pixel_values"].unsqueeze(0)
    costo, latencia = evaluation.architectural_cost(model, sample, trainer.args.device)

    metrics = assemble_metrics(
        model_key, recipe, trainer, elapsed_train, test_metrics, benchmark, costo, latencia
    )
    destino = save_metrics(metrics, results_dir)
    logger.success(f"benchmark de {model_key} completo -> {destino}")
    return metrics
