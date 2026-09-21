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


def test_split_respeta_exclusions_manifest_explicito(tmp_path, food101_falso):
    """--exclusions-manifest en `split` tiene que poder apuntar a un manifiesto propio
    en vez de caer al archivo real del repo. Esto ejercita raw.load_index, que ya
    aplicaba exclusiones correctamente antes de este fix -- la parte nueva es solo que
    el CLI ahora deja elegir que manifiesto usar en vez del default global."""
    excluida = food101_falso["train_rels"][0]
    manifest = tmp_path / "exclusiones.json"
    manifest.write_text(json.dumps({"exclusiones": {"fuga_train_test": [excluida]}}))

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
            "--exclusions-manifest",
            str(manifest),
        ],
    )
    assert resultado.exit_code == 0, resultado.output
    assert excluida not in (tmp_path / "s.csv").read_text()

    # El manifiesto escrito tiene que documentar las exclusiones que realmente se
    # aplicaron (las de --exclusions-manifest), no las del BENCHMARK_MANIFEST real del
    # repo -- de lo contrario, con un --exclusions-manifest distinto del default,
    # train_val_split_manifest.json mentiria sobre que se excluyo.
    manifiesto_escrito = json.loads((tmp_path / "m.json").read_text())
    assert manifiesto_escrito["exclusiones_aplicadas"] == [excluida]


def test_cache_respeta_exclusions_manifest_explicito_al_releer_el_csv(tmp_path, food101_falso):
    """defecto 1 del task brief: splits.load_split ignoraba `exclusions` para train/val
    al releer un CSV ya escrito -- solo lo honraba raw.load_index, en el momento de
    generar el CSV. Este test escribe el split SIN exclusiones (el CSV contiene la fila
    de sobra) y despues invoca `cache` con un manifiesto que la excluye: si `cache`
    (via splits.load_split) no vuelve a filtrar al releer, la imagen se cachea igual.
    Al reproducir la regresion a mano (ver reporte final) esto fallo, confirmando que
    antes del fix el cache la incluia."""
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
    assert food101_falso["train_rels"][0] in (tmp_path / "s.csv").read_text()

    excluida = food101_falso["train_rels"][0]
    manifest = tmp_path / "exclusiones.json"
    manifest.write_text(json.dumps({"exclusiones": {"fuga_train_test": [excluida]}}))

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
            "--exclusions-manifest",
            str(manifest),
        ],
    )
    assert resultado.exit_code == 0, resultado.output
    from vit_for_101_food_app.preprocessing import cache as cache_mod

    assert not cache_mod.cached_path(excluida, cache_dir=tmp_path / "cache").is_file()


def test_verify_compara_contra_el_processor():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from vit_for_101_food_app import features as features_cli

    resultado = runner.invoke(features_cli.app, ["verify"])
    assert resultado.exit_code == 0, resultado.output
    assert "vit" in resultado.output and "mobilevit" in resultado.output
