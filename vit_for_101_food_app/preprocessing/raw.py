"""Acceso al Food-101 original y construccion del indice canonico de imagenes.

Este modulo es la frontera con el disco: todo lo demas del paquete trabaja sobre los
DataFrame que devuelve ``load_index``, no sobre archivos. Las exclusiones detectadas en
el EDA se restan aca y no rio abajo, para que ningun consumidor pueda saltearlas.
"""

import json
from pathlib import Path
import subprocess
import tarfile

from loguru import logger
import pandas as pd

from vit_for_101_food_app.config import (
    BENCHMARK_MANIFEST,
    FOOD101_CLASSES,
    FOOD101_DIR,
    FOOD101_META_DIR,
    FOOD101_URL,
)

SPLITS = ("train", "test")
COLUMNAS = ["rel", "class_dir", "label"]


def pretty_label(class_dir: str) -> str:
    """``apple_pie`` -> ``Apple pie``. Mismo formato que los artefactos del EDA."""
    return class_dir.replace("_", " ").capitalize()


def class_names(classes_path: Path = FOOD101_CLASSES) -> list[str]:
    """Las 101 clases en el orden de ``meta/classes.txt``, que fija los indices."""
    return classes_path.read_text().split()


def load_exclusions(manifest_path: Path = BENCHMARK_MANIFEST) -> set[str]:
    """Imagenes que el EDA marco como corruptas o como fuga train/test."""
    if not manifest_path.is_file():
        logger.warning(f"no hay manifiesto en {manifest_path}; sin exclusiones")
        return set()
    exclusiones = json.loads(manifest_path.read_text()).get("exclusiones", {})
    return {rel for lista in exclusiones.values() for rel in lista}


def load_index(
    split: str,
    meta_dir: Path = FOOD101_META_DIR,
    exclusions: set[str] | None = None,
) -> pd.DataFrame:
    """Indice canonico de un split oficial, sin las imagenes excluidas.

    El orden es lexicografico por ``rel`` y estable: el muestreo del split de validacion
    depende de el, asi que un orden no determinista haria el split irreproducible.
    """
    if split not in SPLITS:
        raise ValueError(f"split invalido: {split!r}; esperaba uno de {SPLITS}")

    excluidas = load_exclusions() if exclusions is None else exclusions
    rels = [r for r in (meta_dir / f"{split}.txt").read_text().split() if r not in excluidas]

    df = pd.DataFrame({"rel": sorted(rels)})
    df["class_dir"] = df["rel"].str.split("/").str[0]
    df["label"] = df["class_dir"].map(pretty_label)
    return df[COLUMNAS].reset_index(drop=True)


def ensure_dataset(
    dest: Path = FOOD101_DIR,
    url: str = FOOD101_URL,
    force: bool = False,
) -> Path:
    """Descarga y extrae Food-101 si hace falta. Idempotente: ~5 GB no se bajan dos veces."""
    # Verifica que el dataset este completamente extraido. El tarball escribe meta/
    # antes de images/, asi que una extraccion interrumpida (comun en sesiones de Colab
    # con timeout) deja meta/ completo pero images/ incompleto. Solo meta/ existiendo
    # no es suficiente; hay que verificar que images/ tiene todas las clases.
    meta_dir = dest / "meta"
    images_dir = dest / "images"
    classes_file = meta_dir / "classes.txt"
    if dest.is_dir() and meta_dir.is_dir() and classes_file.is_file() and not force:
        # Solo si classes.txt existe, cuento cuantas clases deberia haber.
        # El tarball escribe meta/ antes que images/, asi que una extraccion interrumpida
        # deja meta/ completo pero images/ incompleto: necesitamos verificar ambos.
        # Contamos solo directorios, no archivos sueltos (.DS_Store u otros).
        expected_classes = len(class_names(classes_file))
        actual_classes = (
            sum(1 for p in images_dir.iterdir() if p.is_dir()) if images_dir.is_dir() else 0
        )
        if actual_classes == expected_classes:
            logger.info(f"Food-101 ya esta en {dest}")
            return dest

    # Tarball va junto al dest: para default (dest=FOOD101_DIR), tar_path esta bajo
    # RAW_DATA_DIR. Para tests, tar_path queda bajo tmp_path.
    raw_dir = dest.parent
    raw_dir.mkdir(parents=True, exist_ok=True)
    tar_path = raw_dir / "food-101.tar.gz"

    if not tar_path.is_file() or force:
        logger.info(f"descargando {url} (~5 GB, reanudable)")
        subprocess.run(["wget", "-c", "-O", str(tar_path), url], check=True)

    logger.info(f"extrayendo en {raw_dir}")
    with tarfile.open(tar_path) as tar:
        tar.extractall(raw_dir, filter="data")

    logger.success(f"Food-101 disponible en {dest}")
    return dest


def disk_usage_gb(path: Path = FOOD101_DIR) -> float:
    """Espacio que ocupa el dataset, para reportarlo en el notebook."""
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / 1024**3


__all__ = [
    "SPLITS",
    "class_names",
    "disk_usage_gb",
    "ensure_dataset",
    "load_exclusions",
    "load_index",
    "pretty_label",
]
