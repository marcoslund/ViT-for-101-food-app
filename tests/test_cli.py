"""Los comandos son la superficie que usa el notebook y el Makefile."""

import json

import pytest
from typer.testing import CliRunner

from vit_for_101_food_app import dataset as dataset_cli
from vit_for_101_food_app.config import FOOD101_CLASSES, FOOD101_DIR, FOOD101_META_DIR

runner = CliRunner()


@pytest.fixture
def classes_en_ruta_global():
    """Escribe un ``classes.txt`` en la ruta REAL de ``FOOD101_CLASSES`` (global,
    fuera de ``tmp_path``), con clases distintas a las del arbol falso de
    ``food101_falso``. Sirve para probar que un ``--meta-dir`` explicito no queda
    pisado por el default global cuando ese archivo global resulta existir en disco.

    La limpieza corre en un ``finally``, asi que una aserccion que falla no deja
    basura bajo ``data/raw/``: solo se borran los directorios que este fixture creo
    (verificado antes de crear nada), y si el archivo ya existia se restaura su
    contenido original en vez de borrarlo.
    """
    dirs_creados = [d for d in (FOOD101_DIR, FOOD101_META_DIR) if not d.is_dir()]
    ya_existia = FOOD101_CLASSES.is_file()
    contenido_previo = FOOD101_CLASSES.read_text() if ya_existia else None
    FOOD101_META_DIR.mkdir(parents=True, exist_ok=True)
    FOOD101_CLASSES.write_text("clase_falsa_a\nclase_falsa_b\n")
    try:
        yield
    finally:
        if ya_existia:
            FOOD101_CLASSES.write_text(contenido_previo)
        else:
            FOOD101_CLASSES.unlink(missing_ok=True)
        for d in reversed(dirs_creados):  # hijos antes que padres
            try:
                d.rmdir()
            except OSError:
                pass


def test_split_escribe_los_tres_artefactos(tmp_path, food101_falso):
    resultado = runner.invoke(
        dataset_cli.app,
        [
            "split",
            "--meta-dir",
            str(food101_falso["meta"]),
            "--csv-path",
            str(tmp_path / "s.csv"),
            "--manifest-path",
            str(tmp_path / "m.json"),
            "--label-map-path",
            str(tmp_path / "l.json"),
        ],
    )
    assert resultado.exit_code == 0, resultado.output
    assert (tmp_path / "s.csv").is_file()
    assert json.loads((tmp_path / "m.json").read_text())["seed"] == 42
    assert (tmp_path / "l.json").is_file()


def test_split_es_idempotente(tmp_path, food101_falso):
    args = [
        "split",
        "--meta-dir",
        str(food101_falso["meta"]),
        "--csv-path",
        str(tmp_path / "s.csv"),
        "--manifest-path",
        str(tmp_path / "m.json"),
        "--label-map-path",
        str(tmp_path / "l.json"),
    ]
    runner.invoke(dataset_cli.app, args)
    primero = (tmp_path / "s.csv").read_bytes()
    runner.invoke(dataset_cli.app, args)
    assert (tmp_path / "s.csv").read_bytes() == primero


def test_split_no_deja_que_el_default_global_pise_el_meta_dir_explicito(
    tmp_path, food101_falso, classes_en_ruta_global
):
    """Si FOOD101_CLASSES (ruta global por defecto) existe en disco, un --meta-dir
    explicito tiene que seguir mandando: las clases del label_map tienen que salir
    del arbol falso pasado por --meta-dir, no del archivo global."""
    resultado = runner.invoke(
        dataset_cli.app,
        [
            "split",
            "--meta-dir",
            str(food101_falso["meta"]),
            "--csv-path",
            str(tmp_path / "s.csv"),
            "--manifest-path",
            str(tmp_path / "m.json"),
            "--label-map-path",
            str(tmp_path / "l.json"),
        ],
    )
    assert resultado.exit_code == 0, resultado.output
    id2label = json.loads((tmp_path / "l.json").read_text())["id2label"]
    clases_escritas = set(id2label.values())
    assert clases_escritas == set(food101_falso["clases"])
    assert clases_escritas.isdisjoint({"clase_falsa_a", "clase_falsa_b"})


def test_cache_construye_desde_el_split(tmp_path, food101_falso):
    runner.invoke(
        dataset_cli.app,
        [
            "split",
            "--meta-dir",
            str(food101_falso["meta"]),
            "--csv-path",
            str(tmp_path / "s.csv"),
            "--manifest-path",
            str(tmp_path / "m.json"),
            "--label-map-path",
            str(tmp_path / "l.json"),
        ],
    )
    resultado = runner.invoke(
        dataset_cli.app,
        [
            "cache",
            "--src-dir",
            str(food101_falso["images"]),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--csv-path",
            str(tmp_path / "s.csv"),
            "--meta-dir",
            str(food101_falso["meta"]),
            "--short-side",
            "64",
            "--workers",
            "1",
        ],
    )
    assert resultado.exit_code == 0, resultado.output
    assert (tmp_path / "cache" / "cache_manifest.json").is_file()


def test_verify_compara_contra_el_processor():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from vit_for_101_food_app import features as features_cli

    resultado = runner.invoke(features_cli.app, ["verify"])
    assert resultado.exit_code == 0, resultado.output
    assert "vit" in resultado.output and "mobilevit" in resultado.output
