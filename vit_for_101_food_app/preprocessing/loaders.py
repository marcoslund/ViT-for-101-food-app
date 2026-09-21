"""Dataset y DataLoaders: donde se junta todo lo anterior.

Las claves que devuelve ``__getitem__`` son las que espera el Trainer de HuggingFace,
asi que el loop de entrenamiento puede consumirlo sin adaptadores.
"""

from pathlib import Path
import random

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from vit_for_101_food_app.config import (
    CACHE_DIR,
    CACHE_SHORT_SIDE,
    FOOD101_IMAGES_DIR,
    FOOD101_META_DIR,
    LABEL_MAP,
    SEED,
    TRAIN_VAL_SPLIT,
)
from vit_for_101_food_app.preprocessing import cache as cache_mod
from vit_for_101_food_app.preprocessing import policies, processors, splits

SPLITS = ("train", "val", "test")


def _worker_init_fn(worker_id: int) -> None:
    """Semilla determinista por worker, derivada de la semilla del generator del
    DataLoader (torch se la pasa via initial_seed()). Sin esto, cada worker de
    NumPy/random arranca con entropia del SO y el augmentation deja de ser
    reproducible entre corridas, aunque el split y el shuffle si lo sean."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class Food101Dataset(Dataset):
    """Imagenes de Food-101 con el transform de un modelo concreto aplicado al vuelo."""

    def __init__(
        self,
        frame: pd.DataFrame,
        images_root: Path,
        transform,
        label2id: dict[str, int],
        suffix: str = ".jpg",
    ):
        self._rels = frame["rel"].tolist()
        self._labels = [int(label2id[c]) for c in frame["class_dir"]]
        self._root = Path(images_root)
        self._transform = transform
        self._suffix = suffix

    def __len__(self) -> int:
        return len(self._rels)

    def __getitem__(self, idx: int) -> dict:
        ruta = self._root / f"{self._rels[idx]}{self._suffix}"
        # convert("RGB") incondicional: grayscale, CMYK y paleta rompen un modelo
        # que asume 3 canales (hallazgo #7, seccion 3 del EDA).
        with Image.open(ruta) as im:
            imagen = im.convert("RGB")
        return {"pixel_values": self._transform(imagen), "labels": self._labels[idx]}


def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels": torch.tensor([b["labels"] for b in batch], dtype=torch.long),
    }


def build_dataloaders(
    model_key: str,
    train_policy: str = "standard",
    source: str = "cache",
    batch_size: int = 32,
    num_workers: int | None = None,
    splits_to_load: tuple[str, ...] = SPLITS,
    images_root: Path | None = None,
    csv_path: Path = TRAIN_VAL_SPLIT,
    label_map_path: Path = LABEL_MAP,
    meta_dir: Path = FOOD101_META_DIR,
    exclusions: set[str] | None = None,
    seed: int = SEED,
) -> dict[str, DataLoader]:
    """DataLoaders de train, val y test para un modelo del registry.

    ``source="raw"`` lee los JPEG originales en vez del cache: sirve para verificar que
    la segunda pasada de compresion del cache no cambia los resultados.

    ``seed`` fija el generator del DataLoader de train (el unico con ``shuffle=True``) y
    la semilla de cada worker: sin esto el shuffle y el augmentation estocastico
    (``RandomResizedCrop``, ``RandAugment``, ``RandomErasing``) no son reproducibles entre
    corridas, aunque el split este versionado.
    """
    spec = processors.spec_for(model_key)  # KeyError explicito si no esta en el registry

    if source not in ("cache", "raw"):
        raise ValueError(f"source invalido: {source!r}; esperaba 'cache' o 'raw'")
    if images_root is None:
        images_root = CACHE_DIR if source == "cache" else FOOD101_IMAGES_DIR

    if source == "cache" and not cache_mod.cache_is_valid(
        cache_dir=images_root, short_side=CACHE_SHORT_SIDE
    ):
        raise RuntimeError(
            f"el cache en {images_root} no existe o no coincide con CACHE_SHORT_SIDE="
            f"{CACHE_SHORT_SIDE}; regeneralo con: "
            "python -m vit_for_101_food_app.dataset cache"
        )

    # Semilla el generator global de torch, no solo el del sampler: v2.RandomResizedCrop,
    # RandAugment y RandomErasing sacan sus numeros del generator por defecto de torch, no
    # de uno explicito. Con num_workers=0 no hay _worker_loop que los siembre (eso solo
    # pasa dentro de los procesos hijo), asi que sin esta linea la augmentation seria
    # reproducible en shuffle pero no en contenido cuando se corre sincronico.
    torch.manual_seed(seed)

    _, label2id = splits.load_label_map(label_map_path)
    # num_workers=None (default) resuelve a 4. Un 0 explicito tiene que quedar en 0:
    # "0 or 4" evaluaria a 4 y pisaria silenciosamente el pedido de carga sincronica.
    # El conteo de workers tampoco depende de source: no tiene nada que ver con de
    # donde se lee la imagen.
    workers = 4 if num_workers is None else num_workers

    transforms = {
        "train": policies.build_transform(spec, train_policy),
        "val": policies.build_transform(spec, "eval"),
        "test": policies.build_transform(spec, "eval"),
    }

    salida: dict[str, DataLoader] = {}
    for nombre in splits_to_load:
        frame = splits.load_split(
            nombre, csv_path=csv_path, meta_dir=meta_dir, exclusions=exclusions
        )
        dataset = Food101Dataset(frame, images_root, transforms[nombre], label2id)
        shuffle = nombre == "train"
        salida[nombre] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=workers,
            collate_fn=collate,
            pin_memory=torch.cuda.is_available(),
            drop_last=shuffle,
            generator=torch.Generator().manual_seed(seed) if shuffle else None,
            worker_init_fn=_worker_init_fn if (shuffle and workers > 0) else None,
        )
    return salida
