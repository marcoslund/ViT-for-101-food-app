"""El indice canonico es de donde sale todo lo demas. Si las exclusiones no se aplican
aca, cualquier consumidor rio abajo puede saltearlas por olvido."""

import pandas as pd
import pytest

from vit_for_101_food_app.preprocessing import raw


def test_indice_tiene_las_columnas_del_contrato(food101_falso):
    df = raw.load_index("train", meta_dir=food101_falso["meta"], exclusions=set())
    assert list(df.columns) == ["rel", "class_dir", "label"]


def test_indice_de_train_trae_todas_las_imagenes(food101_falso):
    df = raw.load_index("train", meta_dir=food101_falso["meta"], exclusions=set())
    assert len(df) == len(food101_falso["train_rels"])
    assert set(df["rel"]) == set(food101_falso["train_rels"])


def test_indice_de_test_es_disjunto_del_de_train(food101_falso):
    meta = food101_falso["meta"]
    train = raw.load_index("train", meta_dir=meta, exclusions=set())
    test = raw.load_index("test", meta_dir=meta, exclusions=set())
    assert set(train["rel"]).isdisjoint(set(test["rel"]))


def test_indice_esta_ordenado_deterministicamente(food101_falso):
    """El orden fija el muestreo del split: si cambia, el split deja de ser reproducible."""
    meta = food101_falso["meta"]
    a = raw.load_index("train", meta_dir=meta, exclusions=set())
    b = raw.load_index("train", meta_dir=meta, exclusions=set())
    pd.testing.assert_frame_equal(a, b)
    assert a["rel"].is_monotonic_increasing


def test_class_dir_sale_del_prefijo_del_rel(food101_falso):
    df = raw.load_index("train", meta_dir=food101_falso["meta"], exclusions=set())
    assert (df["class_dir"] == df["rel"].str.split("/").str[0]).all()


def test_label_usa_el_mismo_formato_que_los_artefactos_del_eda():
    assert raw.pretty_label("apple_pie") == "Apple pie"
    assert raw.pretty_label("baby_back_ribs") == "Baby back ribs"


def test_las_exclusiones_se_aplican_en_el_indice(food101_falso, manifiesto_con_exclusiones):
    excluidas = manifiesto_con_exclusiones["excluidas"]
    df = raw.load_index("train", meta_dir=food101_falso["meta"], exclusions=excluidas)
    assert excluidas.isdisjoint(set(df["rel"]))
    assert len(df) == len(food101_falso["train_rels"]) - len(excluidas)


def test_load_exclusions_lee_las_dos_listas_del_manifiesto(manifiesto_con_exclusiones):
    leidas = raw.load_exclusions(manifiesto_con_exclusiones["path"])
    assert leidas == manifiesto_con_exclusiones["excluidas"]


def test_load_exclusions_tolera_un_manifiesto_sin_exclusiones(tmp_path):
    ruta = tmp_path / "m.json"
    ruta.write_text('{"seed": 42}')
    assert raw.load_exclusions(ruta) == set()


def test_class_names_respeta_el_orden_del_archivo(food101_falso):
    nombres = raw.class_names(food101_falso["meta"] / "classes.txt")
    assert nombres == food101_falso["clases"]


def test_split_invalido_es_un_error_explicito(food101_falso):
    with pytest.raises(ValueError, match="split"):
        raw.load_index("validacion", meta_dir=food101_falso["meta"], exclusions=set())


def test_ensure_dataset_es_idempotente_si_ya_esta_extraido(food101_falso):
    """Si el dataset ya esta, no vuelve a descargar 5 GB."""
    destino = raw.ensure_dataset(dest=food101_falso["root"])
    assert destino == food101_falso["root"]


def test_ensure_dataset_detecta_extraccion_incompleta(tmp_path, monkeypatch):
    """Una extraccion interrumpida (meta completo, images incompleto) no se trata como lista."""
    # Simula un tarball interrumpido: meta/ completo pero images/ con menos clases.
    dest = tmp_path / "partial"
    dest.mkdir()
    meta = dest / "meta"
    meta.mkdir()
    images = dest / "images"
    images.mkdir()

    # Crea classes.txt diciendo que hay 3 clases.
    (meta / "classes.txt").write_text("apple_pie\nbaby_back_ribs\nwaffles\n")

    # Pero solo crea 2 directorios de clases (simulando extraccion interrumpida).
    (images / "apple_pie").mkdir()
    (images / "baby_back_ribs").mkdir()

    # Intercepta el intento de extraccion para verificar que se intenta (no retorna early).
    extraction_attempted = False

    class MockTar:
        def extractall(self, *args, **kwargs):
            nonlocal extraction_attempted
            extraction_attempted = True

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def mock_tarfile_open(*args, **kwargs):
        return MockTar()

    monkeypatch.setattr("tarfile.open", mock_tarfile_open)

    # Intercepta wget para evitar descarga real.
    def mock_run(cmd, *args, **kwargs):
        pass

    monkeypatch.setattr("subprocess.run", mock_run)

    raw.ensure_dataset(dest=dest, url="http://fake", force=False)
    assert extraction_attempted, "Deberia intentar extraccion con 2 de 3 clases"


def test_ensure_dataset_rechaza_images_con_archivos_sueltos(tmp_path, monkeypatch):
    """N-1 directorios mas un archivo suelto no se trata como N directorios."""
    # Simula images/ incompleto: solo 2 de 3 clases mas un archivo suelto.
    dest = tmp_path / "with_stray"
    dest.mkdir()
    meta = dest / "meta"
    meta.mkdir()
    images = dest / "images"
    images.mkdir()

    # Crea classes.txt diciendo que hay 3 clases.
    (meta / "classes.txt").write_text("apple_pie\nbaby_back_ribs\nwaffles\n")

    # Pero solo crea 2 directorios (incompleto).
    (images / "apple_pie").mkdir()
    (images / "baby_back_ribs").mkdir()

    # Anade un archivo suelto que NO debe contar como clase.
    (images / ".DS_Store").write_text("stray")

    # Intercepta para verificar que se intenta extraccion.
    extraction_attempted = False

    class MockTar:
        def extractall(self, *args, **kwargs):
            nonlocal extraction_attempted
            extraction_attempted = True

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def mock_tarfile_open(*args, **kwargs):
        return MockTar()

    monkeypatch.setattr("tarfile.open", mock_tarfile_open)

    def mock_run(cmd, *args, **kwargs):
        pass

    monkeypatch.setattr("subprocess.run", mock_run)

    raw.ensure_dataset(dest=dest, url="http://fake", force=False)
    assert extraction_attempted, "Deberia intentar extraccion: 2 dirs + 1 file != 3 classes"


def test_load_index_usa_exclusiones_por_defecto_cuando_no_se_pasan(
    food101_falso, manifiesto_con_exclusiones, monkeypatch
):
    """Sin argumento exclusions, load_index aplica las del manifiesto por defecto."""
    # Monkeypatch load_exclusions para devolver las del manifiesto de prueba.
    monkeypatch.setattr(
        "vit_for_101_food_app.preprocessing.raw.load_exclusions",
        lambda manifest_path=None: manifiesto_con_exclusiones["excluidas"],
    )

    # Llama sin el argumento exclusions para ejercer la rama por defecto.
    df = raw.load_index("train", meta_dir=food101_falso["meta"])

    # Las exclusiones deben haberse aplicado.
    excluidas = manifiesto_con_exclusiones["excluidas"]
    assert excluidas.isdisjoint(set(df["rel"]))
    assert len(df) == len(food101_falso["train_rels"]) - len(excluidas)
