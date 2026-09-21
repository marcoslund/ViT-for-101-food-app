"""Capa de datos del benchmark: de los JPEG originales a tensores listos para entrenar.

Cada modulo tiene una responsabilidad:

- ``raw``        descarga/extraccion de Food-101 e indice canonico de imagenes
- ``splits``     split train/val versionado y mapa de etiquetas
- ``cache``      pasada offline que reescala el dataset
- ``processors`` traduce el AutoImageProcessor de cada modelo a un ProcessorSpec
- ``policies``   politicas de augmentation y construccion de transforms
- ``loaders``    Dataset y DataLoaders de PyTorch

``processors`` es el unico modulo que importa ``transformers``.
"""
