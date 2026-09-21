"""El split de validacion es un contrato: los modelos del benchmark tienen que validar
sobre exactamente las mismas imagenes. Estos tests lo fijan, igual que test_data.py
fija el contrato del subset del EDA."""

import hashlib
import json

import pandas as pd
import pytest

from vit_for_101_food_app.preprocessing import raw, splits


@pytest.fixture
def indice(food101_falso):
    return raw.load_index("train", meta_dir=food101_falso["meta"], exclusions=set())


@pytest.fixture
def split(indice):
    return splits.build_split(indice, val_fraction=0.10, seed=42)


def test_split_conserva_todas_las_imagenes(indice, split):
    assert len(split) == len(indice)
    assert set(split["rel"]) == set(indice["rel"])


def test_split_solo_tiene_train_y_val(split):
    assert set(split["split"]) == {"train", "val"}


def test_train_y_val_son_disjuntos(split):
    train = set(split.loc[split["split"] == "train", "rel"])
    val = set(split.loc[split["split"] == "val", "rel"])
    assert train.isdisjoint(val)


def test_val_esta_estratificado_por_clase(split):
    """El balance perfecto de Food-101 se preserva en las dos partes."""
    por_clase = split[split["split"] == "val"]["class_dir"].value_counts()
    assert por_clase.nunique() == 1
    assert por_clase.iloc[0] == 2  # 10% de 20 imagenes por clase en el arbol falso


def test_todas_las_clases_aparecen_en_ambas_partes(split):
    for parte in ("train", "val"):
        assert split[split["split"] == parte]["class_dir"].nunique() == 3


def test_el_split_es_reproducible_con_la_misma_semilla(indice):
    a = splits.build_split(indice, val_fraction=0.10, seed=42)
    b = splits.build_split(indice, val_fraction=0.10, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_semillas_distintas_dan_splits_distintos(indice):
    a = splits.build_split(indice, val_fraction=0.10, seed=42)
    b = splits.build_split(indice, val_fraction=0.10, seed=7)
    assert not a["split"].equals(b["split"])


def test_columnas_del_contrato(split):
    assert list(split.columns) == ["rel", "class_dir", "label", "split"]


def test_artefactos_escritos_y_manifiesto_coherente(tmp_path, split, food101_falso):
    csv = tmp_path / "train_val_split.csv"
    manifest = tmp_path / "train_val_split_manifest.json"
    label_map = tmp_path / "label_map.json"

    info = splits.write_artifacts(
        split,
        food101_falso["clases"],
        csv_path=csv,
        manifest_path=manifest,
        label_map_path=label_map,
    )

    assert csv.is_file() and manifest.is_file() and label_map.is_file()
    datos = json.loads(manifest.read_text())
    assert datos["seed"] == 42
    assert datos["n_train"] + datos["n_val"] == len(split)
    assert datos["csv_sha256"] == hashlib.sha256(csv.read_bytes()).hexdigest()
    assert datos["csv_sha256"] == info["csv_sha256"]


def test_el_csv_escrito_se_relee_identico(tmp_path, split, food101_falso):
    csv = tmp_path / "s.csv"
    splits.write_artifacts(
        split,
        food101_falso["clases"],
        csv_path=csv,
        manifest_path=tmp_path / "m.json",
        label_map_path=tmp_path / "l.json",
    )
    # check_dtype=False a proposito: el contrato son los VALORES y el sha256 del CSV,
    # no el dtype en memoria. Exigir igualdad de dtype obligaba a build_split a forzar
    # StringDtype, lo que ataba la libreria a una version de pandas (ver splits.py).
    pd.testing.assert_frame_equal(pd.read_csv(csv), split, check_dtype=False)


def test_label_map_es_una_biyeccion(food101_falso):
    mapa = splits.build_label_map(food101_falso["clases"])
    id2label, label2id = mapa["id2label"], mapa["label2id"]
    assert len(id2label) == len(label2id) == len(food101_falso["clases"])
    for idx, clase in id2label.items():
        assert label2id[clase] == idx


def test_label_map_respeta_el_orden_de_classes_txt(food101_falso):
    mapa = splits.build_label_map(food101_falso["clases"])
    assert [mapa["id2label"][i] for i in range(len(food101_falso["clases"]))] == food101_falso[
        "clases"
    ]


def test_load_split_devuelve_cada_parte(tmp_path, split, food101_falso):
    csv = tmp_path / "s.csv"
    splits.write_artifacts(
        split,
        food101_falso["clases"],
        csv_path=csv,
        manifest_path=tmp_path / "m.json",
        label_map_path=tmp_path / "l.json",
    )
    train = splits.load_split("train", csv_path=csv)
    val = splits.load_split("val", csv_path=csv)
    assert len(train) + len(val) == len(split)
    assert set(train["split"]) == {"train"}


def test_load_split_de_test_va_al_split_oficial(tmp_path, split, food101_falso):
    """El test oficial no pasa por el CSV: se lee de meta/test.txt, intacto."""
    csv = tmp_path / "s.csv"
    splits.write_artifacts(
        split,
        food101_falso["clases"],
        csv_path=csv,
        manifest_path=tmp_path / "m.json",
        label_map_path=tmp_path / "l.json",
    )
    test = splits.load_split(
        "test", csv_path=csv, meta_dir=food101_falso["meta"], exclusions=set()
    )
    assert set(test["rel"]) == set(food101_falso["test_rels"])
    assert set(test["split"]) == {"test"}


def test_load_split_de_train_y_val_respetan_exclusiones(tmp_path, split, food101_falso):
    """defecto 1 del task brief: load_split reenviaba exclusions a raw.load_index solo
    para test, y el branch de train/val filtraba unicamente por la columna split,
    descartando el argumento. Si esto regresionara, las dos primeras aserciones de
    abajo (que exigen que las excluidas no aparezcan en train NI en val) fallarian --
    antes de este fix, la fila excluida de train seguia presente en el CSV releido."""
    csv = tmp_path / "s.csv"
    splits.write_artifacts(
        split,
        food101_falso["clases"],
        csv_path=csv,
        manifest_path=tmp_path / "m.json",
        label_map_path=tmp_path / "l.json",
    )
    # Elegimos una exclusion de train y otra de val, tomadas del propio split ya escrito.
    excluida_train = split.loc[split["split"] == "train", "rel"].iloc[0]
    excluida_val = split.loc[split["split"] == "val", "rel"].iloc[0]
    exclusiones = {excluida_train, excluida_val}

    train = splits.load_split("train", csv_path=csv, exclusions=exclusiones)
    val = splits.load_split("val", csv_path=csv, exclusions=exclusiones)

    assert excluida_train not in set(train["rel"])
    assert excluida_val not in set(val["rel"])
    assert len(train) + len(val) == len(split) - len(exclusiones)


def test_load_split_sin_exclusiones_no_descarta_filas(tmp_path, split, food101_falso):
    """exclusions=set() (el caso que ejercitan casi todos los demas tests) tiene que
    devolver el split completo, sin filtrar nada."""
    csv = tmp_path / "s.csv"
    splits.write_artifacts(
        split,
        food101_falso["clases"],
        csv_path=csv,
        manifest_path=tmp_path / "m.json",
        label_map_path=tmp_path / "l.json",
    )
    train = splits.load_split("train", csv_path=csv, exclusions=set())
    val = splits.load_split("val", csv_path=csv, exclusions=set())
    assert len(train) + len(val) == len(split)


def test_ninguna_imagen_de_test_aparece_en_train_ni_val(tmp_path, split, food101_falso):
    csv = tmp_path / "s.csv"
    splits.write_artifacts(
        split,
        food101_falso["clases"],
        csv_path=csv,
        manifest_path=tmp_path / "m.json",
        label_map_path=tmp_path / "l.json",
    )
    test = splits.load_split(
        "test", csv_path=csv, meta_dir=food101_falso["meta"], exclusions=set()
    )
    assert set(test["rel"]).isdisjoint(set(split["rel"]))


def test_load_label_map_hace_el_roundtrip_con_claves_int(tmp_path, food101_falso):
    """load_label_map es la unica interfaz sin test del modulo, y la siguiente fase
    mete id2label directo en from_pretrained: si las claves volviesen como str (el
    default de json.loads), esa llamada fallaria o se degradaria en silencio. Este
    test escribe con write_artifacts y relee con load_label_map, así que fallaria si
    la coercion a int desapareciera o si el roundtrip perdiera alguna clase."""
    label_map_path = tmp_path / "l.json"
    splits.write_artifacts(
        splits.build_split(
            raw.load_index("train", meta_dir=food101_falso["meta"], exclusions=set()),
            val_fraction=0.10,
            seed=42,
        ),
        food101_falso["clases"],
        csv_path=tmp_path / "s.csv",
        manifest_path=tmp_path / "m.json",
        label_map_path=label_map_path,
    )

    id2label, label2id = splits.load_label_map(label_map_path)

    assert set(id2label) == set(range(len(food101_falso["clases"])))
    for idx, clase in id2label.items():
        assert isinstance(idx, int)
        assert label2id[clase] == idx
    assert [id2label[i] for i in range(len(food101_falso["clases"]))] == food101_falso["clases"]


def test_load_label_map_da_un_error_guiado_si_falta_el_archivo(tmp_path):
    with pytest.raises(FileNotFoundError, match="dataset split"):
        splits.load_label_map(tmp_path / "no-existe.json")
