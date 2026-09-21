"""Split de validacion y mapa de etiquetas, versionados como contrato.

Food-101 no trae split de validacion. Recortamos uno de train con semilla fija y lo
escribimos a disco con su sha256, igual que el subset de benchmark del EDA: los modelos
tienen que validar sobre exactamente las mismas imagenes para que la comparacion valga.

El split de test oficial no se toca nunca; ``load_split("test")`` lo lee de meta/test.txt.
"""

import hashlib
import json
from pathlib import Path

from loguru import logger
import pandas as pd

from vit_for_101_food_app.config import (
    FOOD101_META_DIR,
    LABEL_MAP,
    SEED,
    TRAIN_VAL_MANIFEST,
    TRAIN_VAL_SPLIT,
    VAL_FRACTION,
)
from vit_for_101_food_app.preprocessing import raw

COLUMNAS = ["rel", "class_dir", "label", "split"]


def build_split(
    index: pd.DataFrame,
    val_fraction: float = VAL_FRACTION,
    seed: int = SEED,
) -> pd.DataFrame:
    """Marca cada fila del indice de train como ``train`` o ``val``.

    El muestreo es estratificado por clase, asi que el balance perfecto de Food-101 se
    preserva en las dos partes y la validacion es trivialmente representativa.
    """
    ordenado = index.sort_values("rel").reset_index(drop=True)
    val_idx = (
        ordenado.groupby("class_dir", group_keys=False)
        .sample(frac=val_fraction, random_state=seed)
        .index
    )
    out = ordenado.assign(split="train")
    out.loc[val_idx, "split"] = "val"
    result = out[COLUMNAS].astype(
        {
            "rel": pd.StringDtype(storage="python", na_value=float("nan")),
            "class_dir": pd.StringDtype(storage="python", na_value=float("nan")),
            "label": pd.StringDtype(storage="python", na_value=float("nan")),
            "split": pd.StringDtype(storage="python", na_value=float("nan")),
        }
    )
    return result


def build_label_map(class_order: list[str]) -> dict:
    """Indices de clase, fijados por el orden de ``meta/classes.txt``.

    Se versiona aunque sea derivable: las matrices de confusion de dos modelos solo se
    pueden comparar celda a celda si ambos usaron los mismos indices.
    """
    return {
        "id2label": {i: c for i, c in enumerate(class_order)},
        "label2id": {c: i for i, c in enumerate(class_order)},
    }


def write_artifacts(
    frame: pd.DataFrame,
    class_order: list[str],
    csv_path: Path = TRAIN_VAL_SPLIT,
    manifest_path: Path = TRAIN_VAL_MANIFEST,
    label_map_path: Path = LABEL_MAP,
    seed: int = SEED,
    val_fraction: float = VAL_FRACTION,
) -> dict:
    """Escribe el CSV, su manifiesto con sha256 y el mapa de etiquetas."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False, lineterminator="\n")

    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    conteos = frame.groupby(["split", "class_dir"]).size()

    manifiesto = {
        "dataset": "Food-101 (Bossard et al., ECCV 2014)",
        "split_origen": "train oficial (meta/train.txt)",
        "seed": seed,
        "val_fraction": val_fraction,
        "n_clases": int(frame["class_dir"].nunique()),
        "n_train": int((frame["split"] == "train").sum()),
        "n_val": int((frame["split"] == "val").sum()),
        "train_por_clase": int(conteos["train"].iloc[0]),
        "val_por_clase": int(conteos["val"].iloc[0]),
        "exclusiones_aplicadas": sorted(raw.load_exclusions()),
        "csv_sha256": digest,
    }
    manifest_path.write_text(json.dumps(manifiesto, indent=2, ensure_ascii=False) + "\n")

    mapa = build_label_map(class_order)
    label_map_path.write_text(json.dumps(mapa, indent=2, ensure_ascii=False) + "\n")

    logger.success(
        f"split escrito: {manifiesto['n_train']} train / {manifiesto['n_val']} val "
        f"en {manifiesto['n_clases']} clases"
    )
    return manifiesto


def load_split(
    split: str,
    csv_path: Path = TRAIN_VAL_SPLIT,
    meta_dir: Path = FOOD101_META_DIR,
    exclusions: set[str] | None = None,
) -> pd.DataFrame:
    """Devuelve ``train``, ``val`` o ``test``.

    ``test`` no sale del CSV sino del split oficial: es el conjunto del benchmark y no
    lo tocamos.
    """
    if split == "test":
        return raw.load_index("test", meta_dir=meta_dir, exclusions=exclusions).assign(
            split="test"
        )[COLUMNAS]

    if split not in ("train", "val"):
        raise ValueError(f"split invalido: {split!r}; esperaba train, val o test")

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"falta {csv_path}; generalo con: python -m vit_for_101_food_app.dataset split"
        )

    frame = pd.read_csv(csv_path)
    return frame[frame["split"] == split].reset_index(drop=True)


def load_label_map(path: Path = LABEL_MAP) -> tuple[dict[int, str], dict[str, int]]:
    """``(id2label, label2id)`` con las claves de ``id2label`` como int."""
    mapa = json.loads(path.read_text())
    return {int(k): v for k, v in mapa["id2label"].items()}, mapa["label2id"]
