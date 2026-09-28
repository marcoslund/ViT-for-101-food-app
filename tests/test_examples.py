"""Selección y extracción de imágenes de ejemplo para ilustrar los resultados."""

import io
import tarfile

import matplotlib
import pandas as pd
from PIL import Image
import pytest

from vit_for_101_food_app.modeling import examples

matplotlib.use("Agg")


@pytest.fixture
def preds():
    rels = [f"{c}/{i}" for c in "ab" for i in range(5)]
    return pd.DataFrame(
        {
            "rel": rels,
            "class_dir": [r[0] for r in rels],
            "ok": [True, False, True, False, True, False, False, True, True, True],
        }
    )


def _jpg(size=(400, 300)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (200, 80, 40)).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def tar_path(tmp_path):
    ruta = tmp_path / "food-101.tar.gz"
    with tarfile.open(ruta, "w:gz") as tar:
        for rel in ("a/0", "a/1", "b/3"):
            datos = _jpg()
            info = tarfile.TarInfo(f"food-101/images/{rel}.jpg")
            info.size = len(datos)
            tar.addfile(info, io.BytesIO(datos))
    return ruta


def test_pick_por_clase_respeta_mascara_orden_y_cupo(preds):
    elegidas = examples.pick(preds, ~preds["ok"], classes=["b", "a"], per_class=1)
    assert elegidas["class_dir"].tolist() == ["b", "a"]
    assert not elegidas["ok"].any()


def test_pick_es_reproducible(preds):
    a = examples.pick(preds, preds["ok"], n=3)
    b = examples.pick(preds, preds["ok"], n=3)
    assert a["rel"].tolist() == b["rel"].tolist()
    assert len(a) == 3


def test_extract_from_tar_saca_solo_lo_pedido_y_reduce(tar_path, tmp_path):
    dest = tmp_path / "imgs"
    rutas = examples.extract_from_tar(["a/1", "b/3", "b/4"], tar_path, dest, side=100)
    assert set(rutas) == {"a/1", "b/3"}  # b/4 no esta en el tar: no corta
    with Image.open(rutas["a/1"]) as img:
        assert min(img.size) == 100
    assert not (dest / "a" / "0.jpg").exists()


def test_ensure_images_prefiere_lo_que_ya_esta_en_disco(tar_path, tmp_path, monkeypatch):
    local = tmp_path / "cache"
    (local / "a").mkdir(parents=True)
    (local / "a" / "0.jpg").write_bytes(_jpg())
    monkeypatch.setattr(examples, "IMAGES_DIR", tmp_path / "extraidas")
    rutas = examples.locate(["a/0"], dirs=[local])
    assert rutas["a/0"].parent == local / "a"


def test_plot_rows_marca_imagenes_faltantes(tar_path, tmp_path):
    rutas = examples.extract_from_tar(["a/0"], tar_path, tmp_path / "imgs")
    fig = examples.plot_rows([("fila", [("a/0", "ok"), ("zz/9", "falta")])], rutas, "titulo")
    textos = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert "imagen no\ndisponible" in textos
