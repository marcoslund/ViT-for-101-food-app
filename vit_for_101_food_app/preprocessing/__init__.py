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
    # El try/except cubre SOLO la deteccion del extra "deep": si alguna de estas tres
    # librerias falta, capturamos el ImportError y dejamos el re-export vacio. Los
    # imports de nuestros propios modulos (loaders, policies, processors) van en el
    # bloque else, fuera del try, para que un ImportError genuino ahi -- un typo en
    # el nombre importado, un bug real de import-time -- se propague en vez de
    # confundirse con "no esta instalado el extra" y desaparecer en silencio.
    import torch  # noqa: F401
    import torchvision  # noqa: F401
    import transformers  # noqa: F401
except ImportError:  # extra "deep" no instalado
    pass
else:
    from vit_for_101_food_app.preprocessing.loaders import (
        Food101Dataset,
        build_dataloaders,
    )
    from vit_for_101_food_app.preprocessing.policies import (
        available,
        build_transform,
    )
    from vit_for_101_food_app.preprocessing.processors import ProcessorSpec, spec_for

    __all__ = [
        "Food101Dataset",
        "ProcessorSpec",
        "available",
        "build_dataloaders",
        "build_transform",
        "spec_for",
    ]
