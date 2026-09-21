"""Capa de datos del benchmark: de los JPEG originales a tensores listos para entrenar.

Cada modulo tiene una responsabilidad:

- ``raw``        descarga/extraccion de Food-101 e indice canonico de imagenes
- ``splits``     split train/val versionado y mapa de etiquetas
- ``cache``      pasada offline que reescala el dataset
- ``processors`` traduce el AutoImageProcessor de cada modelo a un ProcessorSpec
- ``policies``   politicas de augmentation y construccion de transforms
- ``loaders``    Dataset y DataLoaders de PyTorch

``processors`` es el unico modulo que importa ``transformers``.

``raw``, ``splits`` y ``cache`` no dependen del extra ``deep`` (torch/torchvision/
transformers) y se importan por su ruta completa. Los re-exports de abajo si dependen
de ``deep``; si no esta instalado, quedan sin definir y __all__ vacio, para no romper
la importabilidad basica del paquete.
"""

__all__: list[str] = []

try:
    from vit_for_101_food_app.preprocessing.loaders import (
        Food101Dataset,
        build_dataloaders,
    )
    from vit_for_101_food_app.preprocessing.policies import (
        available,
        build_transform,
    )
    from vit_for_101_food_app.preprocessing.processors import ProcessorSpec, spec_for
except ImportError:  # extra "deep" no instalado
    pass
else:
    __all__ = [
        "Food101Dataset",
        "ProcessorSpec",
        "available",
        "build_dataloaders",
        "build_transform",
        "spec_for",
    ]
