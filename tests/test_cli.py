"""Los comandos son la superficie que usa el notebook y el Makefile."""

import json

import pytest
from typer.testing import CliRunner

from vit_for_101_food_app import dataset as dataset_cli

runner = CliRunner()


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
