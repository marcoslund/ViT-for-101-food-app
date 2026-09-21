"""Arbol Food-101 falso: la estructura real pesa 5 GB y no esta disponible en CI."""

import json

from PIL import Image
import pytest

CLASES = ["apple_pie", "baby_back_ribs", "waffles"]
IMG_POR_CLASE_TRAIN = 20
IMG_POR_CLASE_TEST = 8


@pytest.fixture
def food101_falso(tmp_path):
    """Replica la estructura de Food-101 con imagenes chicas de colores distintos.

    Devuelve un dict con las rutas que consumen los modulos: root, meta, images.
    Las imagenes tienen aspect ratios distintos a proposito, para que los tests de
    geometria del cache y de los transforms tengan algo que verificar.
    """
    root = tmp_path / "food-101"
    meta = root / "meta"
    images = root / "images"
    meta.mkdir(parents=True)

    tamanos = [(512, 384), (384, 512), (512, 512)]
    train_rels, test_rels = [], []

    for ci, clase in enumerate(CLASES):
        (images / clase).mkdir(parents=True)
        for i in range(IMG_POR_CLASE_TRAIN + IMG_POR_CLASE_TEST):
            rel = f"{clase}/{ci}{i:04d}"
            ancho, alto = tamanos[i % len(tamanos)]
            color = (30 + ci * 60, 90 + i * 3, 200 - ci * 40)
            Image.new("RGB", (ancho, alto), color).save(images / f"{rel}.jpg", quality=95)
            (train_rels if i < IMG_POR_CLASE_TRAIN else test_rels).append(rel)

    (meta / "train.txt").write_text("\n".join(train_rels) + "\n")
    (meta / "test.txt").write_text("\n".join(test_rels) + "\n")
    (meta / "classes.txt").write_text("\n".join(CLASES) + "\n")

    return {"root": root, "meta": meta, "images": images,
            "train_rels": train_rels, "test_rels": test_rels, "clases": CLASES}


@pytest.fixture
def manifiesto_con_exclusiones(tmp_path, food101_falso):
    """Manifiesto al estilo del EDA, excluyendo dos imagenes conocidas de train."""
    excluidas = food101_falso["train_rels"][:2]
    ruta = tmp_path / "benchmark_subset_manifest.json"
    ruta.write_text(json.dumps({
        "seed": 42,
        "exclusiones": {
            "corruptas_o_ilegibles": [excluidas[0]],
            "fuga_train_test": [excluidas[1]],
        },
    }))
    return {"path": ruta, "excluidas": set(excluidas)}
