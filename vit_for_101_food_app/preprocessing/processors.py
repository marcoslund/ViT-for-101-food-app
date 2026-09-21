"""Traduce el AutoImageProcessor de cada modelo a un dataclass inspeccionable.

Este es el UNICO modulo del proyecto que importa ``transformers``, y el unico que
conoce resoluciones, medias y desvios. Todo se lee del checkpoint en runtime: escribir
esos valores a mano es exactamente como se arruina una comparacion entre arquitecturas
(ver la seccion 8 del EDA).

Los processors de distintas arquitecturas NO comparten estructura. Algunos hacen un
resize cuadrado directo, otros redimensionan el lado corto y despues recortan al centro;
algunos normalizan y otros no. ``ProcessorSpec`` captura esa estructura, no un par de
numeros, y por eso ``policies.build_transform`` puede reconstruir cualquiera de las dos.
"""

from dataclasses import dataclass
from functools import cache

from transformers import AutoImageProcessor

from vit_for_101_food_app.config import MODELS


@dataclass(frozen=True)
class ProcessorSpec:
    """Lo que un modelo le hace a una imagen, leido de su propio processor."""

    key: str
    checkpoint: str
    do_resize: bool
    size: dict[str, int]
    do_center_crop: bool
    crop_size: dict[str, int] | None
    resample: int
    rescale_factor: float
    do_normalize: bool
    image_mean: tuple[float, ...] | None
    image_std: tuple[float, ...] | None
    do_flip_channel_order: bool

    @property
    def resize_shortest(self) -> int | None:
        """Lado corto al que redimensiona, o ``None`` si hace un resize cuadrado."""
        if self.do_resize and "shortest_edge" in self.size:
            return int(self.size["shortest_edge"])
        return None

    @property
    def target_size(self) -> int:
        """Resolucion cuadrada que efectivamente entra al modelo."""
        caja = self.crop_size if (self.do_center_crop and self.crop_size) else self.size
        if "shortest_edge" in caja:
            return int(caja["shortest_edge"])
        alto, ancho = int(caja["height"]), int(caja["width"])
        if alto != ancho:
            raise ValueError(f"{self.key}: se esperaba entrada cuadrada, no {alto}x{ancho}")
        return alto

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return (3, self.target_size, self.target_size)


def _tupla(valor) -> tuple[float, ...] | None:
    return tuple(float(v) for v in valor) if valor is not None else None


def _caja(valor) -> dict[str, int] | None:
    return {k: int(v) for k, v in valor.items()} if valor else None


@cache
def load_processor(key: str):
    """El AutoImageProcessor crudo. Lo usa el test de equivalencia como referencia."""
    if key not in MODELS:
        raise KeyError(f"{key!r} no esta en config.MODELS; disponibles: {sorted(MODELS)}")
    return AutoImageProcessor.from_pretrained(MODELS[key])


@cache
def spec_for(key: str) -> ProcessorSpec:
    """Lee el processor del checkpoint y lo traduce a ProcessorSpec."""
    processor = load_processor(key)
    d = processor.to_dict()

    # do_normalize puede venir None (MobileViT). None NO significa "usa el default":
    # significa que no normaliza. La presencia de image_mean/image_std no alcanza para
    # inferirlo -- MobileViT los declara y aun asi no normaliza.
    return ProcessorSpec(
        key=key,
        checkpoint=MODELS[key],
        do_resize=bool(d.get("do_resize") or False),
        size=_caja(d.get("size")) or {},
        do_center_crop=bool(d.get("do_center_crop") or False),
        crop_size=_caja(d.get("crop_size")),
        resample=int(d.get("resample") or 2),
        rescale_factor=float(d.get("rescale_factor") or 1 / 255),
        do_normalize=bool(d.get("do_normalize") or False),
        image_mean=_tupla(d.get("image_mean")),
        image_std=_tupla(d.get("image_std")),
        do_flip_channel_order=bool(d.get("do_flip_channel_order") or False),
    )


def all_specs() -> dict[str, ProcessorSpec]:
    return {key: spec_for(key) for key in MODELS}


def ficha_tecnica() -> list[dict]:
    """Tabla para el informe: que le hace cada modelo a la imagen de entrada."""
    filas = []
    for spec in all_specs().values():
        filas.append(
            {
                "modelo": spec.key,
                "checkpoint": spec.checkpoint,
                "resolucion": spec.target_size,
                "resize": spec.resize_shortest or f"{spec.target_size}x{spec.target_size}",
                "center_crop": spec.do_center_crop,
                "normaliza": spec.do_normalize,
                "mean": spec.image_mean if spec.do_normalize else None,
                "orden_canales": "BGR" if spec.do_flip_channel_order else "RGB",
            }
        )
    return filas
