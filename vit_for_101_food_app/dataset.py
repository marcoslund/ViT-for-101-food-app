"""CLI de preparacion del dataset: descarga, split de validacion y cache.

python -m vit_for_101_food_app.dataset download
python -m vit_for_101_food_app.dataset split
python -m vit_for_101_food_app.dataset cache
"""

from pathlib import Path

from loguru import logger
import typer

from vit_for_101_food_app.config import (
    CACHE_DIR,
    CACHE_JPEG_QUALITY,
    CACHE_SHORT_SIDE,
    FOOD101_DIR,
    FOOD101_IMAGES_DIR,
    FOOD101_META_DIR,
    LABEL_MAP,
    SEED,
    TRAIN_VAL_MANIFEST,
    TRAIN_VAL_SPLIT,
    VAL_FRACTION,
)
from vit_for_101_food_app.preprocessing import cache as cache_mod
from vit_for_101_food_app.preprocessing import raw, splits

app = typer.Typer(help="Preparacion del dataset Food-101 para el benchmark.")


@app.command()
def download(dest: Path = FOOD101_DIR, force: bool = False):
    """Descarga y extrae Food-101 (~5 GB). Idempotente."""
    raw.ensure_dataset(dest=dest, force=force)
    logger.success(f"{raw.disk_usage_gb(dest):.2f} GB en {dest}")


@app.command()
def split(
    meta_dir: Path = FOOD101_META_DIR,
    classes_path: Path | None = None,
    csv_path: Path = TRAIN_VAL_SPLIT,
    manifest_path: Path = TRAIN_VAL_MANIFEST,
    label_map_path: Path = LABEL_MAP,
    val_fraction: float = VAL_FRACTION,
    seed: int = SEED,
):
    """Recorta el split de validacion de train y escribe los artefactos versionados."""
    indice = raw.load_index("train", meta_dir=meta_dir)
    frame = splits.build_split(indice, val_fraction=val_fraction, seed=seed)
    # classes_path=None (default) resuelve SIEMPRE contra meta_dir, no contra el
    # default global FOOD101_CLASSES: un --meta-dir explicito no puede quedar pisado
    # por un archivo que resulte existir en la ruta por defecto (ver label_map.json,
    # que fija los indices de clase para comparar modelos entre si).
    clases = raw.class_names(
        classes_path if classes_path is not None else meta_dir / "classes.txt"
    )
    info = splits.write_artifacts(
        frame,
        clases,
        csv_path=csv_path,
        manifest_path=manifest_path,
        label_map_path=label_map_path,
        seed=seed,
        val_fraction=val_fraction,
    )
    typer.echo(f"sha256: {info['csv_sha256']}")


@app.command()
def cache(
    src_dir: Path = FOOD101_IMAGES_DIR,
    cache_dir: Path = CACHE_DIR,
    csv_path: Path = TRAIN_VAL_SPLIT,
    meta_dir: Path = FOOD101_META_DIR,
    short_side: int = CACHE_SHORT_SIDE,
    quality: int = CACHE_JPEG_QUALITY,
    workers: int = 0,
    force: bool = False,
):
    """Reescala todo el dataset al lado corto configurado. Idempotente."""
    rels: list[str] = []
    for nombre in ("train", "val", "test"):
        rels += splits.load_split(nombre, csv_path=csv_path, meta_dir=meta_dir)["rel"].tolist()

    info = cache_mod.build_cache(
        sorted(set(rels)),
        src_dir=src_dir,
        cache_dir=cache_dir,
        short_side=short_side,
        quality=quality,
        workers=workers or None,
        force=force,
    )
    typer.echo(f"{info['n_imagenes']} imagenes, {cache_mod.disk_usage_mb(cache_dir):.0f} MB")


if __name__ == "__main__":
    app()
