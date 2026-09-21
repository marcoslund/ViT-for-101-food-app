"""Verifica la integridad de los artefactos del EDA versionados en data/processed/.

Estos archivos son el contrato de entrada del benchmark: definen el conjunto exacto
de imagenes que evalua cada modelo. Si alguien los edita a mano o los regenera con
otra semilla, las comparaciones entre modelos dejan de ser validas. Estos tests
existen para que ese cambio no pase desapercibido.
"""

import hashlib
import json

import pandas as pd
import pytest

from vit_for_101_food_app.config import (
    BENCHMARK_MANIFEST,
    BENCHMARK_SUBSET,
    CLASS_DIFFICULTY,
    SEED,
)

TERCILES = {"dificil", "medio", "facil"}


@pytest.fixture(scope="module")
def manifest():
    return json.loads(BENCHMARK_MANIFEST.read_text())


@pytest.fixture(scope="module")
def subset():
    return pd.read_csv(BENCHMARK_SUBSET)


@pytest.fixture(scope="module")
def difficulty():
    return pd.read_csv(CLASS_DIFFICULTY)


def test_artefactos_presentes():
    for path in (BENCHMARK_SUBSET, BENCHMARK_MANIFEST, CLASS_DIFFICULTY):
        assert path.is_file(), f"falta {path}; se genera en notebooks/1.0-eda-food101.ipynb"


def test_subset_coincide_con_su_sha256(manifest):
    """El manifiesto fija el sha256 del CSV: si no coincide, el CSV cambio."""
    digest = hashlib.sha256(BENCHMARK_SUBSET.read_bytes()).hexdigest()
    assert digest == manifest["csv_sha256"]


def test_semilla_del_manifiesto_es_la_del_proyecto(manifest):
    assert manifest["seed"] == SEED


def test_subset_es_estratificado_y_completo(manifest, subset):
    assert len(subset) == manifest["n_imagenes"]
    assert subset["class_dir"].nunique() == manifest["n_clases"] == 101

    por_clase = subset["class_dir"].value_counts()
    assert por_clase.nunique() == 1, "el subset dejo de estar balanceado por clase"
    assert por_clase.iloc[0] == manifest["imagenes_por_clase"]


def test_subset_no_tiene_imagenes_repetidas(subset):
    assert not subset["rel"].duplicated().any()


def test_subset_respeta_las_exclusiones(manifest, subset):
    excluidas = set(manifest["exclusiones"]["corruptas_o_ilegibles"])
    excluidas |= set(manifest["exclusiones"]["fuga_train_test"])
    assert excluidas.isdisjoint(set(subset["rel"]))


def test_subset_tiene_las_columnas_que_consume_el_benchmark(subset):
    assert list(subset.columns) == ["rel", "class_dir", "label", "margen", "tercil", "bytes"]
    assert set(subset["tercil"].unique()) <= TERCILES
    assert (subset["bytes"] > 0).all()


def test_bytes_totales_coinciden_con_el_manifiesto(manifest, subset):
    assert int(subset["bytes"].sum()) == manifest["bytes_totales"]


def test_dificultad_cubre_las_101_clases(difficulty):
    assert len(difficulty) == 101
    assert difficulty["class_dir"].nunique() == 101
    assert set(difficulty["tercil"].unique()) == TERCILES


def test_dificultad_esta_ordenada_de_mas_a_menos_dificil(difficulty):
    assert difficulty["margen"].is_monotonic_increasing


def test_terciles_del_subset_coinciden_con_los_de_dificultad(subset, difficulty):
    esperado = difficulty.set_index("class_dir")["tercil"]
    observado = subset.drop_duplicates("class_dir").set_index("class_dir")["tercil"]
    pd.testing.assert_series_equal(
        observado.sort_index(), esperado.sort_index(), check_names=False
    )
