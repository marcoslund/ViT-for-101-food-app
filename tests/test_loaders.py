"""El Dataset devuelve exactamente las claves que espera el Trainer de HuggingFace,
para que el loop de entrenamiento no necesite adaptadores."""

import multiprocessing as mp

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("transformers")

import torch

from vit_for_101_food_app.config import MODELS
from vit_for_101_food_app.preprocessing import (
    loaders,
    policies,
    processors,
    raw,
    splits,
)


@pytest.fixture
def frame(food101_falso):
    return raw.load_index("train", meta_dir=food101_falso["meta"], exclusions=set())


@pytest.fixture
def label2id(food101_falso):
    return splits.build_label_map(food101_falso["clases"])["label2id"]


def _dataset(frame, food101_falso, label2id, key="vit", policy="eval"):
    return loaders.Food101Dataset(
        frame,
        images_root=food101_falso["images"],
        transform=policies.build_transform(processors.spec_for(key), policy),
        label2id=label2id,
    )


def test_len_es_el_del_frame(frame, food101_falso, label2id):
    assert len(_dataset(frame, food101_falso, label2id)) == len(frame)


def test_item_tiene_las_claves_del_trainer(frame, food101_falso, label2id):
    item = _dataset(frame, food101_falso, label2id)[0]
    assert set(item) == {"pixel_values", "labels"}


@pytest.mark.parametrize("key", list(MODELS))
def test_pixel_values_tiene_la_resolucion_nativa(key, frame, food101_falso, label2id):
    item = _dataset(frame, food101_falso, label2id, key=key)[0]
    assert item["pixel_values"].shape == processors.spec_for(key).input_shape
    assert item["pixel_values"].dtype == torch.float32


def test_labels_son_indices_validos(frame, food101_falso, label2id):
    ds = _dataset(frame, food101_falso, label2id)
    etiquetas = {ds[i]["labels"] for i in range(len(ds))}
    assert etiquetas <= set(label2id.values())
    assert all(isinstance(e, int) for e in etiquetas)


def test_la_etiqueta_corresponde_a_la_clase_de_la_fila(frame, food101_falso, label2id):
    ds = _dataset(frame, food101_falso, label2id)
    for i in (0, len(ds) // 2, len(ds) - 1):
        assert ds[i]["labels"] == label2id[frame["class_dir"].iloc[i]]


def test_collate_arma_un_batch(frame, food101_falso, label2id):
    ds = _dataset(frame, food101_falso, label2id)
    batch = loaders.collate([ds[0], ds[1], ds[2]])
    assert batch["pixel_values"].shape[0] == 3
    assert batch["labels"].shape == (3,)
    assert batch["labels"].dtype == torch.long


def test_un_batch_del_dataloader_entero(frame, food101_falso, label2id):
    ds = _dataset(frame, food101_falso, label2id)
    dl = torch.utils.data.DataLoader(ds, batch_size=4, collate_fn=loaders.collate)
    batch = next(iter(dl))
    assert batch["pixel_values"].shape == (4, *processors.spec_for("vit").input_shape)


def test_lee_del_cache_cuando_se_le_pide(tmp_path, frame, food101_falso, label2id):
    from vit_for_101_food_app.preprocessing import cache

    destino = tmp_path / "cache"
    cache.build_cache(
        frame["rel"].tolist(),
        src_dir=food101_falso["images"],
        cache_dir=destino,
        short_side=64,
        quality=95,
        workers=1,
    )
    ds = loaders.Food101Dataset(
        frame,
        images_root=destino,
        transform=policies.build_transform(processors.spec_for("vit"), "eval"),
        label2id=label2id,
    )
    assert ds[0]["pixel_values"].shape == processors.spec_for("vit").input_shape


def test_build_dataloaders_arma_los_tres_splits(tmp_path, food101_falso):
    csv = tmp_path / "s.csv"
    indice = raw.load_index("train", meta_dir=food101_falso["meta"], exclusions=set())
    split = splits.build_split(indice, val_fraction=0.10, seed=42)
    splits.write_artifacts(
        split,
        food101_falso["clases"],
        csv_path=csv,
        manifest_path=tmp_path / "m.json",
        label_map_path=tmp_path / "l.json",
    )

    dls = loaders.build_dataloaders(
        "vit",
        source="raw",
        batch_size=4,
        num_workers=0,
        images_root=food101_falso["images"],
        csv_path=csv,
        label_map_path=tmp_path / "l.json",
        meta_dir=food101_falso["meta"],
        exclusions=set(),
    )
    assert set(dls) == {"train", "val", "test"}
    for dl in dls.values():
        batch = next(iter(dl))
        assert batch["pixel_values"].shape[1:] == processors.spec_for("vit").input_shape


def test_val_y_test_no_usan_augmentation(tmp_path, food101_falso):
    """Una politica estocastica en validacion haria la metrica irreproducible."""
    csv = tmp_path / "s.csv"
    indice = raw.load_index("train", meta_dir=food101_falso["meta"], exclusions=set())
    split = splits.build_split(indice, val_fraction=0.10, seed=42)
    splits.write_artifacts(
        split,
        food101_falso["clases"],
        csv_path=csv,
        manifest_path=tmp_path / "m.json",
        label_map_path=tmp_path / "l.json",
    )

    dls = loaders.build_dataloaders(
        "vit",
        train_policy="standard",
        source="raw",
        batch_size=2,
        num_workers=0,
        images_root=food101_falso["images"],
        csv_path=csv,
        label_map_path=tmp_path / "l.json",
        meta_dir=food101_falso["meta"],
        exclusions=set(),
    )
    a = next(iter(dls["val"]))["pixel_values"]
    b = next(iter(dls["val"]))["pixel_values"]
    torch.testing.assert_close(a, b)


def test_build_dataloaders_respeta_exclusiones_en_los_tres_splits(tmp_path, food101_falso):
    """defecto 1 del task brief: splits.load_split honraba exclusions solo para test y
    lo descartaba para train/val, asi que build_dataloaders dejaba una imagen excluida
    presente en dos tercios del benchmark. Si esa regresion volviera, las exclusiones
    de train o val (o ambas) reaparecerian en el DataFrame subyacente y esto fallaria."""
    csv = tmp_path / "s.csv"
    indice = raw.load_index("train", meta_dir=food101_falso["meta"], exclusions=set())
    split = splits.build_split(indice, val_fraction=0.10, seed=42)
    splits.write_artifacts(
        split,
        food101_falso["clases"],
        csv_path=csv,
        manifest_path=tmp_path / "m.json",
        label_map_path=tmp_path / "l.json",
    )

    excluida_train = split.loc[split["split"] == "train", "rel"].iloc[0]
    excluida_val = split.loc[split["split"] == "val", "rel"].iloc[0]
    excluida_test = food101_falso["test_rels"][0]
    exclusiones = {excluida_train, excluida_val, excluida_test}

    dls = loaders.build_dataloaders(
        "vit",
        source="raw",
        batch_size=4,
        num_workers=0,
        images_root=food101_falso["images"],
        csv_path=csv,
        label_map_path=tmp_path / "l.json",
        meta_dir=food101_falso["meta"],
        exclusions=exclusiones,
    )

    rels_por_split = {nombre: set(dl.dataset._rels) for nombre, dl in dls.items()}
    assert excluida_train not in rels_por_split["train"]
    assert excluida_val not in rels_por_split["val"]
    assert excluida_test not in rels_por_split["test"]


def _build_train_loader(tmp_path, food101_falso, seed=None):
    csv = tmp_path / "s.csv"
    if not csv.is_file():
        indice = raw.load_index("train", meta_dir=food101_falso["meta"], exclusions=set())
        split = splits.build_split(indice, val_fraction=0.10, seed=42)
        splits.write_artifacts(
            split,
            food101_falso["clases"],
            csv_path=csv,
            manifest_path=tmp_path / "m.json",
            label_map_path=tmp_path / "l.json",
        )
    kwargs = {
        "train_policy": "standard",
        "source": "raw",
        "batch_size": 8,
        "num_workers": 0,
        "images_root": food101_falso["images"],
        "csv_path": csv,
        "label_map_path": tmp_path / "l.json",
        "meta_dir": food101_falso["meta"],
        "exclusions": set(),
        "splits_to_load": ("train",),
    }
    if seed is not None:
        kwargs["seed"] = seed
    return loaders.build_dataloaders("vit", **kwargs)["train"]


def test_misma_semilla_da_el_mismo_primer_batch_de_train(tmp_path, food101_falso):
    """defecto 4 del task brief: sin generator ni manual_seed, dos construcciones del
    dataloader de train (shuffle=True, policy='standard' con RandomResizedCrop +
    HorizontalFlip estocasticos) dan un orden Y un contenido de batch distintos cada
    vez. El pool de train del arbol falso tiene 54 imagenes (3 clases x 18, tras el
    10% de val) y batch_size=8: bastante grande para que un shuffle no sembrado
    coincida por azar. Si build_dataloaders dejara de sembrar el generator o el RNG
    global de torch, este test fallaria -- lo verificamos abajo reproduciendo la
    regresion a mano."""
    dl_a = _build_train_loader(tmp_path, food101_falso, seed=123)
    dl_b = _build_train_loader(tmp_path, food101_falso, seed=123)

    batch_a = next(iter(dl_a))
    batch_b = next(iter(dl_b))

    torch.testing.assert_close(batch_a["pixel_values"], batch_b["pixel_values"])
    torch.testing.assert_close(batch_a["labels"], batch_b["labels"])


def test_semillas_distintas_dan_batches_distintos(tmp_path, food101_falso):
    """Confirma que el seed realmente controla el resultado, no que build_dataloaders
    ignora el argumento y siempre da lo mismo (lo que haria pasar el test anterior
    por una razon incorrecta)."""
    dl_a = _build_train_loader(tmp_path, food101_falso, seed=123)
    dl_b = _build_train_loader(tmp_path, food101_falso, seed=999)

    batch_a = next(iter(dl_a))
    batch_b = next(iter(dl_b))

    distinto = not torch.allclose(
        batch_a["pixel_values"], batch_b["pixel_values"]
    ) or not torch.equal(batch_a["labels"], batch_b["labels"])
    assert distinto, "semillas distintas no pueden dar exactamente el mismo batch"


def test_modelo_desconocido_es_un_error_explicito(tmp_path, food101_falso):
    with pytest.raises(KeyError):
        loaders.build_dataloaders("resnet", source="raw", images_root=food101_falso["images"])


def test_source_cache_con_cache_invalido_o_ausente_es_un_error_guiado(tmp_path, food101_falso):
    """defecto 2 del task brief: cache_is_valid existia pero build_dataloaders(source=
    'cache') nunca la llamaba, asi que un cache generado con un CACHE_SHORT_SIDE viejo
    (o directamente ausente) se leia en silencio. Si esta llamada se sacara, esto
    fallaria porque no habria excepcion."""
    with pytest.raises(RuntimeError, match="dataset cache"):
        loaders.build_dataloaders(
            "vit",
            source="cache",
            images_root=tmp_path / "cache-inexistente",
            label_map_path=tmp_path / "l.json",
        )


def test_source_invalido_es_un_error_explicito(tmp_path, food101_falso):
    with pytest.raises(ValueError, match="source"):
        loaders.build_dataloaders(
            "vit",
            source="drive",
            images_root=food101_falso["images"],
        )


def test_num_workers_explicito_en_cero_no_usa_workers(monkeypatch, tmp_path, food101_falso):
    """num_workers=0 explicito tiene que respetarse literal: 0 or 4 == 4 es el bug que
    esto pisa (ver defecto 1 del task brief). Si build_dataloaders volviera a calcular
    ``num_workers or 4``, este test construiria el DataLoader con 4 procesos en vez de 0
    y lo detectariamos interceptando el kwarg real que recibe DataLoader.__init__."""
    csv = tmp_path / "s.csv"
    indice = raw.load_index("train", meta_dir=food101_falso["meta"], exclusions=set())
    split = splits.build_split(indice, val_fraction=0.10, seed=42)
    splits.write_artifacts(
        split,
        food101_falso["clases"],
        csv_path=csv,
        manifest_path=tmp_path / "m.json",
        label_map_path=tmp_path / "l.json",
    )

    vistos = []
    original_init = torch.utils.data.DataLoader.__init__

    def espia(self, *args, **kwargs):
        vistos.append(kwargs.get("num_workers"))
        return original_init(self, *args, **kwargs)

    monkeypatch.setattr(torch.utils.data.DataLoader, "__init__", espia)

    loaders.build_dataloaders(
        "vit",
        source="raw",
        batch_size=2,
        num_workers=0,
        images_root=food101_falso["images"],
        csv_path=csv,
        label_map_path=tmp_path / "l.json",
        meta_dir=food101_falso["meta"],
        exclusions=set(),
        splits_to_load=("val",),
    )
    assert vistos == [0]


def test_num_workers_none_usa_el_default_de_cuatro(tmp_path, food101_falso, monkeypatch):
    """num_workers=None (el default del parametro) tiene que resolver a 4, sin importar
    ``source``: el conteo de workers no tiene nada que ver con de donde se lee la imagen."""
    csv = tmp_path / "s.csv"
    indice = raw.load_index("train", meta_dir=food101_falso["meta"], exclusions=set())
    split = splits.build_split(indice, val_fraction=0.10, seed=42)
    splits.write_artifacts(
        split,
        food101_falso["clases"],
        csv_path=csv,
        manifest_path=tmp_path / "m.json",
        label_map_path=tmp_path / "l.json",
    )

    vistos = []
    original_init = torch.utils.data.DataLoader.__init__

    def espia(self, *args, **kwargs):
        vistos.append(kwargs.get("num_workers"))
        return original_init(self, *args, **kwargs)

    monkeypatch.setattr(torch.utils.data.DataLoader, "__init__", espia)

    loaders.build_dataloaders(
        "vit",
        source="raw",
        images_root=food101_falso["images"],
        csv_path=csv,
        label_map_path=tmp_path / "l.json",
        meta_dir=food101_falso["meta"],
        exclusions=set(),
        splits_to_load=("val",),
    )
    assert vistos == [4]


def test_dataloader_con_dos_workers_itera_un_batch(frame, food101_falso, label2id):
    """Con num_workers>0 el Dataset (transform incluido) se manda a procesos hijos por
    pickle. Task 5 (commit ceb6d6b, "Fix ronda 3") reemplazo los v2.Lambda con closures
    de build_transform por callables a nivel de modulo especificamente para que esto
    funcione bajo el start method 'spawn' (default en macOS, y no solo bajo 'fork', que
    lo esconde porque el worker hereda memoria en vez de picklear). Este test corre el
    DataLoader de verdad, con num_workers=2, y exige un batch real -- forma y dtype
    correctos en ambas claves -- en vez de solo comprobar que la iteracion no vuela: un
    DataLoader que devolviera basura tampoco lanzaria una excepcion."""
    ds = _dataset(frame, food101_falso, label2id)
    dl = torch.utils.data.DataLoader(ds, batch_size=4, num_workers=2, collate_fn=loaders.collate)
    metodo_arranque = mp.get_start_method(allow_none=False)

    try:
        batch = next(iter(dl))
    except Exception as exc:  # noqa: BLE001 -- se re-lanza como fallo de assert con contexto
        pytest.fail(
            f"DataLoader con num_workers=2 fallo bajo start method {metodo_arranque!r}: "
            f"{type(exc).__name__}: {exc}"
        )

    assert batch["pixel_values"].shape == (4, *processors.spec_for("vit").input_shape)
    assert batch["pixel_values"].dtype == torch.float32
    assert batch["labels"].shape == (4,)
    assert batch["labels"].dtype == torch.long
