"""Imágenes del dataset para ilustrar los resultados del benchmark.

Las métricas dicen cuánto acierta cada modelo; estas grillas muestran en qué imágenes.
Se eligen a partir de las predicciones por imagen de ``reports/results`` (con semilla
fija, así la selección es reproducible) y se buscan en este orden:

  1. el cache de imágenes reescaladas (``data/interim/food-101-288``);
  2. el dataset extraído (``data/raw/food-101/images``);
  3. las imágenes ya extraídas por este módulo (``data/interim/comparativa-imagenes``);
  4. el ``food-101.tar.gz`` sin extraer: se recorre una vez y se sacan solo las imágenes
     pedidas, sin descomprimir los 5 GB.

Ninguna de esas carpetas se versiona: el repositorio no distribuye el dataset.
"""

from collections.abc import Iterable
from pathlib import Path
import tarfile

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from vit_for_101_food_app.config import (
    CACHE_DIR,
    FOOD101_IMAGES_DIR,
    INTERIM_DATA_DIR,
    RAW_DATA_DIR,
    SEED,
)
from vit_for_101_food_app.plots import INK, INK_MUTED

IMAGES_DIR = INTERIM_DATA_DIR / "comparativa-imagenes"
TAR_PATH = RAW_DATA_DIR / "food-101.tar.gz"
TAR_PREFIX = "food-101/images/"
THUMB_SIDE = 288


def pick(
    preds: pd.DataFrame,
    mask: pd.Series,
    classes: Iterable[str] | None = None,
    per_class: int = 4,
    n: int | None = None,
    seed: int = SEED,
) -> pd.DataFrame:
    """Elige imágenes que cumplen ``mask``, reproducible con ``seed``.

    Con ``classes``, hasta ``per_class`` imágenes de cada clase, en ese orden. Sin
    ``classes``, hasta ``n`` imágenes del total.
    """
    candidatas = preds[mask].sort_values("rel")
    if classes is None:
        k = min(n or len(candidatas), len(candidatas))
        return candidatas.sample(k, random_state=seed).sort_values("rel")
    partes = []
    for clase in classes:
        de_la_clase = candidatas[candidatas["class_dir"] == clase]
        partes.append(de_la_clase.sample(min(per_class, len(de_la_clase)), random_state=seed))
    return pd.concat(partes) if partes else candidatas.iloc[0:0]


def locate(
    rels: Iterable[str], dirs: Iterable[Path] = (CACHE_DIR, FOOD101_IMAGES_DIR, IMAGES_DIR)
) -> dict[str, Path]:
    """Rutas de las imágenes que ya están en disco, en el orden de prioridad de ``dirs``."""
    encontradas = {}
    for rel in rels:
        for d in dirs:
            ruta = Path(d) / f"{rel}.jpg"
            if ruta.is_file():
                encontradas[rel] = ruta
                break
    return encontradas


def _thumbnail(fileobj, dest: Path, side: int) -> None:
    with Image.open(fileobj) as img:
        img = img.convert("RGB")
        escala = side / min(img.size)
        if escala < 1:
            img = img.resize(
                (round(img.width * escala), round(img.height * escala)), Image.Resampling.LANCZOS
            )
        dest.parent.mkdir(parents=True, exist_ok=True)
        img.save(dest, quality=92)


def extract_from_tar(
    rels: Iterable[str],
    tar_path: Path = TAR_PATH,
    dest: Path = IMAGES_DIR,
    side: int = THUMB_SIDE,
) -> dict[str, Path]:
    """Saca del ``.tar.gz`` solo las imágenes pedidas, reducidas a ``side`` de lado corto.

    Recorre el archivo en modo streaming y corta apenas encontró todas: no hace falta
    extraer el dataset completo para ilustrar un puñado de resultados.
    """
    faltan = {f"{TAR_PREFIX}{rel}.jpg": rel for rel in rels}
    extraidas = {}
    if not faltan:
        return extraidas
    logger.info(f"extrayendo {len(faltan)} imágenes de {tar_path} (una sola pasada)")
    with tarfile.open(tar_path, mode="r|gz") as tar:
        for miembro in tar:
            rel = faltan.pop(miembro.name, None)
            if rel is None:
                continue
            ruta = Path(dest) / f"{rel}.jpg"
            _thumbnail(tar.extractfile(miembro), ruta, side)
            extraidas[rel] = ruta
            if not faltan:
                break
    if faltan:
        logger.warning(f"{len(faltan)} imágenes no están en {tar_path}")
    return extraidas


def ensure_images(rels: Iterable[str], tar_path: Path | None = TAR_PATH) -> dict[str, Path]:
    """Ruta de cada imagen pedida; extrae del ``.tar.gz`` las que no estén en disco.

    Las que no se encuentran en ningún lado quedan fuera del diccionario: las grillas
    las muestran como faltantes en vez de cortar el notebook.
    """
    rels = list(dict.fromkeys(rels))
    rutas = locate(rels)
    faltan = [r for r in rels if r not in rutas]
    if faltan and tar_path is not None and Path(tar_path).is_file():
        rutas.update(extract_from_tar(faltan, tar_path))
    return rutas


def _square(path: Path):
    with Image.open(path) as img:
        img = img.convert("RGB")
        lado = min(img.size)
        izq, arriba = (img.width - lado) // 2, (img.height - lado) // 2
        return img.crop((izq, arriba, izq + lado, arriba + lado))


def plot_rows(
    rows: list[tuple[str, list[tuple[str, str]]]],
    paths: dict[str, Path],
    title: str,
    cell: float = 2.1,
) -> plt.Figure:
    """Grilla de imágenes por filas: cada fila es ``(título, [(rel, epígrafe), ...])``.

    Las imágenes se recortan al cuadrado central, como las ve el modelo después del
    resize. Cada fila es una subfigura con su título a la izquierda, así un título
    largo no le quita lugar a las imágenes.
    """
    ncols = max(len(imagenes) for _, imagenes in rows)
    con_titulo = any(titulo for titulo, _ in rows)
    alto_fila = cell + 0.6 + (0.25 if con_titulo else 0)
    fig = plt.figure(figsize=(cell * ncols, alto_fila * len(rows) + 0.5), layout="constrained")
    fig.patch.set_facecolor("white")
    subfigs = np.atleast_1d(fig.subfigures(len(rows), 1))
    for subfig, (titulo_fila, imagenes) in zip(subfigs, rows):
        subfig.set_facecolor("white")
        if titulo_fila:
            subfig.suptitle(
                titulo_fila, x=0.005, ha="left", fontsize=9.5, color=INK, fontweight="bold"
            )
        axes = np.atleast_1d(subfig.subplots(1, ncols))
        for j, ax in enumerate(axes):
            ax.set_xticks([])
            ax.set_yticks([])
            for lado in ax.spines.values():
                lado.set_visible(False)
            if j >= len(imagenes):
                ax.axis("off")
                continue
            rel, epigrafe = imagenes[j]
            if rel in paths:
                ax.imshow(_square(paths[rel]))
            else:
                ax.text(
                    0.5,
                    0.5,
                    "imagen no\ndisponible",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=INK_MUTED,
                    transform=ax.transAxes,
                )
            ax.set_xlabel(epigrafe, fontsize=7.5, color=INK, labelpad=4)
    fig.suptitle(title, x=0.005, ha="left", fontsize=12, color=INK, fontweight="bold")
    return fig


def rows_by_class(
    examples: pd.DataFrame,
    caption,
    row_title,
) -> list[tuple[str, list[tuple[str, str]]]]:
    """Agrupa ``examples`` en una fila por clase, en el orden en que aparecen."""
    filas = []
    for clase, grupo in examples.groupby("class_dir", sort=False):
        filas.append((row_title(clase), [(r.rel, caption(r)) for r in grupo.itertuples()]))
    return filas
