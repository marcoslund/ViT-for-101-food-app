"""Pasada offline que reescala Food-101 al lado corto configurado.

Decodificar JPEG de 512 px es el cuello de botella real del entrenamiento en Colab: el
mismo trabajo se repite en cada epoca y por cada modelo. Esta pasada lo hace una vez.

Solo se toca el lado corto; el aspect ratio se preserva porque la geometria de recorte
es decision del transform, no del cache. La segunda pasada de compresion JPEG que esto
introduce es una perdida real pero simetrica entre todos los modelos, asi que no sesga
la comparacion: ``loaders.build_dataloaders(source="raw")`` permite verificarlo.
"""

from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import tempfile

from loguru import logger
from PIL import Image
from tqdm import tqdm

from vit_for_101_food_app.config import (
    CACHE_DIR,
    CACHE_JPEG_QUALITY,
    CACHE_MANIFEST,
    CACHE_SHORT_SIDE,
    FOOD101_IMAGES_DIR,
)

MANIFIESTO = CACHE_MANIFEST.name


def source_path(rel: str, src_dir: Path = FOOD101_IMAGES_DIR) -> Path:
    return src_dir / f"{rel}.jpg"


def cached_path(rel: str, cache_dir: Path = CACHE_DIR) -> Path:
    return cache_dir / f"{rel}.jpg"


def resize_one(
    rel: str,
    src_dir: Path,
    cache_dir: Path,
    short_side: int,
    quality: int,
    force: bool,
) -> tuple[str, str]:
    """Reescala una imagen. Devuelve ``(rel, estado)`` con estado en
    ``{"escrita", "salteada", "fallida"}``."""
    destino = cached_path(rel, cache_dir)
    if destino.is_file() and not force:
        return rel, "salteada"

    try:
        with Image.open(source_path(rel, src_dir)) as im:
            # Incondicional: resuelve grayscale, CMYK y paleta de una sola vez.
            im = im.convert("RGB")
            corto = min(im.size)
            if corto > short_side:
                escala = short_side / corto
                nuevo = (round(im.width * escala), round(im.height * escala))
                im = im.resize(nuevo, Image.BILINEAR)
            destino.parent.mkdir(parents=True, exist_ok=True)
            # Escribir a archivo temporal en el mismo directorio, luego reemplazar
            # atomicamente. Si el proceso muere durante save(), queda un .tmp en lugar
            # de un JPEG corrompido en la ruta final. No hay garanta para archivos
            # corrompidos preexistentes de corridas antiguas no-atomicas.
            fd, temp_path = tempfile.mkstemp(
                suffix=".tmp", dir=destino.parent, prefix=destino.stem + "_"
            )
            try:
                with os.fdopen(fd, "wb") as f:
                    im.save(f, "JPEG", quality=quality, optimize=True)
                os.replace(temp_path, destino)
            except Exception:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
                raise
    except Exception as exc:  # noqa: BLE001 archivo faltante, truncado o ilegible
        logger.warning(f"no se pudo cachear {rel}: {type(exc).__name__}: {exc}")
        return rel, "fallida"

    return rel, "escrita"


def build_cache(
    rels: list[str],
    src_dir: Path = FOOD101_IMAGES_DIR,
    cache_dir: Path = CACHE_DIR,
    short_side: int = CACHE_SHORT_SIDE,
    quality: int = CACHE_JPEG_QUALITY,
    workers: int | None = None,
    force: bool = False,
) -> dict:
    """Construye el cache. Idempotente: una corrida interrumpida se retoma sin repetir."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    workers = workers or min(8, os.cpu_count() or 1)

    conteos = {"escrita": 0, "salteada": 0, "fallida": 0}
    fallidas: list[str] = []
    args = (src_dir, cache_dir, short_side, quality, force)

    if workers == 1:
        pares = [resize_one(rel, *args) for rel in tqdm(rels, desc="cache")]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futuros = [pool.submit(resize_one, rel, *args) for rel in rels]
            pares = [f.result() for f in tqdm(futuros, desc="cache")]

    for rel, estado in pares:
        conteos[estado] += 1
        if estado == "fallida":
            fallidas.append(rel)

    manifiesto = {
        "short_side": short_side,
        "quality": quality,
        "n_imagenes": conteos["escrita"] + conteos["salteada"],
        "n_escritas": conteos["escrita"],
        "n_salteadas": conteos["salteada"],
        "n_fallidas": conteos["fallida"],
        "fallidas": sorted(fallidas),
        "pillow": Image.__version__,
    }
    (cache_dir / MANIFIESTO).write_text(json.dumps(manifiesto, indent=2) + "\n")

    logger.success(
        f"cache: {conteos['escrita']} escritas, {conteos['salteada']} salteadas, "
        f"{conteos['fallida']} fallidas en {cache_dir}"
    )
    return manifiesto


def cache_is_valid(cache_dir: Path = CACHE_DIR, short_side: int = CACHE_SHORT_SIDE) -> bool:
    """El cache existe y se genero con los parametros vigentes.

    Si alguien sube ``CACHE_SHORT_SIDE`` porque se sumo una arquitectura de mayor
    resolucion, esto lo detecta en vez de entrenar sobre imagenes demasiado chicas.
    """
    manifiesto_path = cache_dir / MANIFIESTO
    if not manifiesto_path.is_file():
        return False
    return json.loads(manifiesto_path.read_text()).get("short_side") == short_side


def disk_usage_mb(cache_dir: Path = CACHE_DIR) -> float:
    total = sum(f.stat().st_size for f in cache_dir.rglob("*.jpg"))
    return total / 1024**2
