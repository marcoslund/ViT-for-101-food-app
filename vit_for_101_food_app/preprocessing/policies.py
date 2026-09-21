"""Politicas de augmentation y construccion de los transforms.

La separacion central del diseno:

    build_transform(spec, policy) = politica.geometria(spec.target_size)
                                  + cola de tensor derivada del spec
                                  + politica.after_tensor()

La politica aporta SOLO la parte geometrica. La normalizacion, el reescalado y el orden
de canales salen siempre del ProcessorSpec, de modo que es estructuralmente imposible
que un modelo reciba las estadisticas de otro.

La imagen se convierte a tensor uint8 (``v2.ToImage``) ANTES de la geometria, igual que
hacen los processors de HuggingFace (backend torchvision): redimensionar un tensor no da
bit a bit lo mismo que redimensionar la imagen PIL. Por eso la conversion va primero,
para que la politica ``eval`` coincida con el processor hasta la tolerancia del test de
equivalencia.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
import math

import torch
from torchvision.transforms import InterpolationMode, v2

from vit_for_101_food_app.preprocessing.processors import ProcessorSpec

# Codigos de resample de PIL -> modos de torchvision.
_INTERPOLACION = {
    0: InterpolationMode.NEAREST,
    1: InterpolationMode.LANCZOS,
    2: InterpolationMode.BILINEAR,
    3: InterpolationMode.BICUBIC,
}

_RESCALE_ESTANDAR = 1 / 255


@dataclass(frozen=True)
class Policy:
    """Una politica es la parte geometrica del pipeline, y nada mas."""

    name: str
    geometry: Callable[[ProcessorSpec, InterpolationMode], list]
    after_tensor: Callable[[], list] = field(default=list)
    descripcion: str = ""


def _interp(spec: ProcessorSpec) -> InterpolationMode:
    return _INTERPOLACION.get(spec.resample, InterpolationMode.BILINEAR)


def _geometria_eval(spec: ProcessorSpec, interp: InterpolationMode) -> list:
    """Replica exacta de lo que hace el processor: resize y, si corresponde, recorte."""
    ops = []
    if spec.do_resize:
        if spec.resize_shortest is not None:
            ops.append(v2.Resize(spec.resize_shortest, interpolation=interp))
        else:
            ops.append(v2.Resize((spec.size["height"], spec.size["width"]), interpolation=interp))
    if spec.do_center_crop and spec.crop_size:
        ops.append(v2.CenterCrop((spec.crop_size["height"], spec.crop_size["width"])))
    return ops


def _geometria_standard(spec: ProcessorSpec, interp: InterpolationMode) -> list:
    return [
        v2.RandomResizedCrop(
            spec.target_size, scale=(0.65, 1.0), interpolation=interp, antialias=True
        ),
        v2.RandomHorizontalFlip(),
    ]


def _geometria_strong(spec: ProcessorSpec, interp: InterpolationMode) -> list:
    return [*_geometria_standard(spec, interp), v2.RandAugment(interpolation=interp)]


def _geometria_resize_only(spec: ProcessorSpec, interp: InterpolationMode) -> list:
    """Ablacion de la seccion 8 #5 del EDA: sin recorte, para descartar que las
    diferencias entre modelos vengan de cuanta imagen descarta el CenterCrop."""
    return [v2.Resize((spec.target_size, spec.target_size), interpolation=interp)]


POLICIES: dict[str, Policy] = {
    "eval": Policy(
        name="eval",
        geometry=_geometria_eval,
        descripcion="replica del processor; validacion y test",
    ),
    "standard": Policy(
        name="standard",
        geometry=_geometria_standard,
        descripcion="RandomResizedCrop(0.65-1.0) + HorizontalFlip",
    ),
    "strong": Policy(
        name="strong",
        geometry=_geometria_strong,
        after_tensor=lambda: [v2.RandomErasing(p=0.25)],
        descripcion="standard + RandAugment + RandomErasing",
    ),
    "resize_only": Policy(
        name="resize_only",
        geometry=_geometria_resize_only,
        descripcion="sin recorte; ablacion de la seccion 8 #5 del EDA",
    ),
}


def _cola_de_tensor(spec: ProcessorSpec) -> list:
    """Reescalado, normalizacion y orden de canales. Siempre derivada del spec.

    El orden replica al de los processors de HuggingFace: rescale, normalize, flip.
    ``v2.ToImage()`` NO va aca: HuggingFace (transformers>=5, backend torchvision)
    convierte a tensor uint8 ANTES de redimensionar, y el resize sobre tensor no da
    bit a bit lo mismo que el resize sobre PIL. Por eso ToImage se aplica antes de la
    geometria en build_transform, no en esta cola.
    """
    ops: list = []

    if math.isclose(spec.rescale_factor, _RESCALE_ESTANDAR, rel_tol=1e-9):
        ops.append(v2.ToDtype(torch.float32, scale=True))
    else:
        ops.append(v2.ToDtype(torch.float32, scale=False))
        ops.append(v2.Lambda(lambda t, f=spec.rescale_factor: t * f))

    # Solo si do_normalize es verdadero. NUNCA inferirlo de image_mean/image_std:
    # MobileViT los declara y no normaliza. Ver processors.spec_for.
    if spec.do_normalize:
        ops.append(v2.Normalize(mean=list(spec.image_mean), std=list(spec.image_std)))

    if spec.do_flip_channel_order:
        ops.append(v2.Lambda(lambda t: t.flip(-3)))

    return ops


def build_transform(spec: ProcessorSpec, policy: str = "standard") -> Callable:
    """Transform completo para un modelo bajo una politica dada."""
    if policy not in POLICIES:
        raise KeyError(f"politica desconocida: {policy!r}; disponibles: {available()}")

    elegida = POLICIES[policy]
    return v2.Compose(
        [
            v2.ToImage(),  # PIL -> tensor uint8 antes de la geometria, como HuggingFace
            *elegida.geometry(spec, _interp(spec)),
            *_cola_de_tensor(spec),
            *elegida.after_tensor(),
        ]
    )


def available() -> list[str]:
    return sorted(POLICIES)
