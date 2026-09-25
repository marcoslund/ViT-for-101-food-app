"""Fine-tuning comun a los modelos del benchmark.

El notebook solo llama a estas funciones: la construccion del modelo y el loop de
entrenamiento viven aca por la misma razon que ``evaluation.py``. Si dos modelos se
entrenan con codigo distinto, la diferencia medida deja de ser una diferencia entre
arquitecturas. Todo es generico sobre ``model_key`` (una clave del registry
``config.MODELS``), asi que sirve igual para swin, vit o mobilevit.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import shutil
import time

from loguru import logger
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch
from transformers import (
    AutoModelForImageClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from vit_for_101_food_app.config import MODELS, SEED


@dataclass(frozen=True)
class TrainingRecipe:
    """Receta de fine-tuning, la misma para todos los modelos del benchmark.

    ``learning_rate`` 5e-4 es alto para fine-tuning y viene del criterio del proyecto,
    no de evidencia: se mantiene para que la comparacion entre arquitecturas sea justa.
    Para swin-base a 384px puede ser agresivo; si el train diverge (loss NaN), bajarlo
    a 5e-5 es lo primero a probar.
    """

    epochs: int = 20
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    batch_size: int = 8
    grad_accum_steps: int = 4  # batch efectivo = batch_size * grad_accum_steps
    early_stopping_patience: int = 3
    metric_for_best: str = "f1_macro"
    seed: int = SEED


def build_model(model_key: str, id2label: dict[int, str], label2id: dict[str, int]):
    """Carga el checkpoint del registry y le pone una cabeza nueva de 101 clases."""
    return AutoModelForImageClassification.from_pretrained(
        MODELS[model_key],
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )


def count_parameters(model) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
    }


def guard_output_dir(output_dir: Path, resume: bool) -> None:
    """Frena si la carpeta ya tiene checkpoints y no se pidio retomar la corrida.

    Sin esto, empezar de cero sobre checkpoints viejos los mezcla en silencio.
    """
    previos = sorted(Path(output_dir).glob("checkpoint-*"))
    if previos and not resume:
        raise RuntimeError(
            f"{output_dir} ya tiene checkpoints ({[p.name for p in previos]}). "
            "Pone RESUME = True para seguir esa corrida, o borra la carpeta para empezar de cero."
        )


def build_trainer(
    model,
    recipe: TrainingRecipe,
    output_dir: Path,
    train_dataset,
    val_dataset,
    num_workers: int,
    gradient_checkpointing: bool = True,
) -> Trainer:
    """Arma el Trainer de HuggingFace con la receta del benchmark.

    ``gradient_checkpointing`` cambia memoria por computo: es lo que hace entrar a
    swin-base-384 en una T4 de 16 GB. ``fp16`` solo si hay GPU.
    """
    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=recipe.epochs,
        per_device_train_batch_size=recipe.batch_size,
        per_device_eval_batch_size=recipe.batch_size,
        gradient_accumulation_steps=recipe.grad_accum_steps,
        learning_rate=recipe.learning_rate,
        weight_decay=recipe.weight_decay,
        warmup_ratio=recipe.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model=recipe.metric_for_best,
        greater_is_better=True,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=num_workers,
        report_to="none",
        seed=int(recipe.seed),
    )
    return Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=recipe.early_stopping_patience)],
    )


def run_training(trainer: Trainer, resume: bool = False) -> float:
    """Entrena hasta el techo de epocas (o hasta el early stopping). Devuelve segundos."""
    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume or None)
    elapsed = time.time() - t0
    logger.success(
        f"entrenamiento: {elapsed / 60:.1f} min | best {trainer.state.best_metric:.4f} "
        f"en {trainer.state.best_model_checkpoint}"
    )
    return elapsed


def save_best_model(trainer: Trainer, output_dir: Path) -> Path:
    """Guarda el mejor modelo (ya cargado por load_best_model_at_end) en output_dir/best."""
    best = Path(output_dir) / "best"
    trainer.save_model(str(best))
    logger.success(f"mejor modelo guardado en {best}")
    return best


def zip_model(model_dir: Path, zip_path: Path) -> Path:
    """Comprime la carpeta del modelo en un .zip listo para descargar."""
    zip_path = Path(zip_path)
    archivo = shutil.make_archive(str(zip_path.with_suffix("")), "zip", root_dir=str(model_dir))
    logger.success(f"modelo comprimido en {archivo}")
    return Path(archivo)


def download_to_browser(path: Path) -> None:
    """En Colab dispara la descarga al navegador; fuera de Colab solo informa la ruta."""
    try:
        from google.colab import files  # type: ignore

        files.download(str(path))
    except ImportError:
        logger.info(f"no es Colab: el archivo quedo en {path}")


def software_versions() -> dict[str, str | None]:
    import transformers

    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def recipe_as_dict(recipe: TrainingRecipe) -> dict:
    return asdict(recipe)
