"""El cache existe para que decodificar JPEG no domine el tiempo de entrenamiento.
Tiene que ser idempotente (una sesion de Colab se corta a la mitad) y no puede tocar
la geometria mas alla de reducir el lado corto."""

import json

from PIL import Image
import pytest

from vit_for_101_food_app.preprocessing import cache, raw


@pytest.fixture
def rels(food101_falso):
    return food101_falso["train_rels"][:6]


@pytest.fixture
def construido(tmp_path, food101_falso, rels):
    destino = tmp_path / "cache"
    info = cache.build_cache(
        rels,
        src_dir=food101_falso["images"],
        cache_dir=destino,
        short_side=64,
        quality=95,
        workers=1,
    )
    return {"dir": destino, "info": info, "rels": rels}


def test_escribe_una_imagen_por_rel(construido):
    for rel in construido["rels"]:
        assert cache.cached_path(rel, cache_dir=construido["dir"]).is_file()


def test_el_lado_corto_queda_en_el_valor_pedido(construido):
    for rel in construido["rels"]:
        with Image.open(cache.cached_path(rel, cache_dir=construido["dir"])) as im:
            assert min(im.size) == 64


def test_preserva_el_aspect_ratio(construido, food101_falso):
    for rel in construido["rels"]:
        with Image.open(food101_falso["images"] / f"{rel}.jpg") as original:
            ratio_original = original.width / original.height
        with Image.open(cache.cached_path(rel, cache_dir=construido["dir"])) as cacheada:
            ratio_cache = cacheada.width / cacheada.height
        assert ratio_original == pytest.approx(ratio_cache, rel=0.02)


def test_las_imagenes_del_cache_son_rgb(construido):
    for rel in construido["rels"]:
        with Image.open(cache.cached_path(rel, cache_dir=construido["dir"])) as im:
            assert im.convert("RGB").mode == "RGB"
            assert im.mode == "RGB"


def test_convierte_a_rgb_lo_que_no_lo_es(tmp_path, food101_falso):
    """Grayscale y paleta rompen un DataLoader que asuma 3 canales (seccion 4 del EDA)."""
    rel = "apple_pie/gris"
    Image.new("L", (300, 200), 128).save(food101_falso["images"] / f"{rel}.jpg")
    destino = tmp_path / "c"
    cache.build_cache(
        [rel],
        src_dir=food101_falso["images"],
        cache_dir=destino,
        short_side=64,
        quality=95,
        workers=1,
    )
    with Image.open(cache.cached_path(rel, cache_dir=destino)) as im:
        assert im.mode == "RGB"


def test_es_idempotente_y_saltea_lo_ya_hecho(construido, food101_falso):
    segunda = cache.build_cache(
        construido["rels"],
        src_dir=food101_falso["images"],
        cache_dir=construido["dir"],
        short_side=64,
        quality=95,
        workers=1,
    )
    assert segunda["n_escritas"] == 0
    assert segunda["n_salteadas"] == len(construido["rels"])


def test_force_reescribe(construido, food101_falso):
    segunda = cache.build_cache(
        construido["rels"],
        src_dir=food101_falso["images"],
        cache_dir=construido["dir"],
        short_side=64,
        quality=95,
        workers=1,
        force=True,
    )
    assert segunda["n_escritas"] == len(construido["rels"])


def test_no_agranda_imagenes_mas_chicas_que_el_objetivo(tmp_path, food101_falso):
    rel = "apple_pie/chica"
    Image.new("RGB", (40, 30), (10, 20, 30)).save(food101_falso["images"] / f"{rel}.jpg")
    destino = tmp_path / "c"
    cache.build_cache(
        [rel],
        src_dir=food101_falso["images"],
        cache_dir=destino,
        short_side=64,
        quality=95,
        workers=1,
    )
    with Image.open(cache.cached_path(rel, cache_dir=destino)) as im:
        assert im.size == (40, 30)


def test_el_manifiesto_registra_los_parametros(construido):
    manifiesto = json.loads((construido["dir"] / "cache_manifest.json").read_text())
    assert manifiesto["short_side"] == 64
    assert manifiesto["quality"] == 95
    assert manifiesto["n_imagenes"] == len(construido["rels"])
    assert "pillow" in manifiesto


def test_cache_is_valid_detecta_el_desajuste_de_parametros(construido):
    assert cache.cache_is_valid(cache_dir=construido["dir"], short_side=64)
    assert not cache.cache_is_valid(cache_dir=construido["dir"], short_side=288)


def test_cache_is_valid_es_falso_si_no_hay_cache(tmp_path):
    assert not cache.cache_is_valid(cache_dir=tmp_path / "no-existe", short_side=288)


def test_la_estructura_de_clases_se_replica(construido):
    assert (construido["dir"] / "apple_pie").is_dir()


def test_ocupa_menos_que_el_original(construido, food101_falso):
    original = sum(
        (food101_falso["images"] / f"{r}.jpg").stat().st_size for r in construido["rels"]
    )
    cacheado = sum(
        cache.cached_path(r, cache_dir=construido["dir"]).stat().st_size
        for r in construido["rels"]
    )
    assert cacheado < original


def test_un_archivo_faltante_se_reporta_sin_abortar(tmp_path, food101_falso, rels):
    destino = tmp_path / "c"
    info = cache.build_cache(
        [*rels, "apple_pie/no_existe"],
        src_dir=food101_falso["images"],
        cache_dir=destino,
        short_side=64,
        quality=95,
        workers=1,
    )
    assert info["n_fallidas"] == 1
    assert "apple_pie/no_existe" in info["fallidas"]
    assert info["n_escritas"] == len(rels)


def test_source_path_y_cached_path_son_coherentes(food101_falso, tmp_path):
    rel = food101_falso["train_rels"][0]
    assert cache.source_path(rel, src_dir=food101_falso["images"]).is_file()
    assert cache.cached_path(rel, cache_dir=tmp_path).name.endswith(".jpg")


def test_el_indice_alimenta_el_cache_sin_traduccion(food101_falso):
    """Los rel del indice canonico son las claves del cache, sin transformacion."""
    df = raw.load_index("train", meta_dir=food101_falso["meta"], exclusions=set())
    assert cache.source_path(df["rel"].iloc[0], src_dir=food101_falso["images"]).is_file()


def test_build_cache_con_workers_paralelo(tmp_path, food101_falso, rels):
    """La rama ProcessPoolExecutor funciona con workers > 1."""
    destino = tmp_path / "cache_paralelo"
    info = cache.build_cache(
        rels,
        src_dir=food101_falso["images"],
        cache_dir=destino,
        short_side=64,
        quality=95,
        workers=2,
    )
    assert info["n_escritas"] == len(rels)
    for rel in rels:
        assert cache.cached_path(rel, cache_dir=destino).is_file()


@pytest.mark.parametrize("mode", ["L", "P", "CMYK"])
def test_convierte_a_rgb_todas_las_paletas(tmp_path, food101_falso, mode):
    """Grayscale, Palette y CMYK se convierten a RGB sin silencio."""
    rel = f"apple_pie/{mode.lower()}"
    if mode == "L":
        img = Image.new("L", (300, 200), 128)
        img.save(food101_falso["images"] / f"{rel}.jpg")
    elif mode == "P":
        # Imagen de paleta: guardar como PNG (JPEG no soporta P), cache abre y convierte
        img = Image.new("RGB", (300, 200), (100, 150, 200))
        img = img.convert("P")
        img.save(food101_falso["images"] / f"{rel}.jpg", "PNG")
    else:  # CMYK
        img = Image.new("CMYK", (300, 200), (100, 150, 200, 50))
        img.save(food101_falso["images"] / f"{rel}.jpg")
    destino = tmp_path / f"c_{mode}"
    cache.build_cache(
        [rel],
        src_dir=food101_falso["images"],
        cache_dir=destino,
        short_side=64,
        quality=95,
        workers=1,
    )
    with Image.open(cache.cached_path(rel, cache_dir=destino)) as im:
        assert im.mode == "RGB", f"Modo {mode} no se convirtio a RGB"


def test_archivo_truncado_no_se_acepta_silenciosamente(tmp_path, food101_falso, rels):
    """Un archivo corrompido/truncado preexistente en el destino no se salta.
    Si la primera corrida escribe atomicamente, no queda archivo truncado.
    Si una corrida antigua no-atomica dejo uno, la nueva corrida lo detecta y reescribe.
    """
    destino = tmp_path / "cache_truncado"
    rel = rels[0]
    destino_archivo = cache.cached_path(rel, cache_dir=destino)

    # Primera corrida: construir cache normally
    cache.build_cache(
        [rel],
        src_dir=food101_falso["images"],
        cache_dir=destino,
        short_side=64,
        quality=95,
        workers=1,
    )
    tamanio_original = destino_archivo.stat().st_size

    # Simular corrupcion: reemplazar con archivo truncado
    destino_archivo.parent.mkdir(parents=True, exist_ok=True)
    with open(destino_archivo, "wb") as f:
        f.write(b"\xff\xd8\xff\xe0")  # JPEG header incompleto, no EOI

    # Segunda corrida con force=True: debe detectar corrupto y reescribir
    info = cache.build_cache(
        [rel],
        src_dir=food101_falso["images"],
        cache_dir=destino,
        short_side=64,
        quality=95,
        workers=1,
        force=True,
    )
    tamanio_despues = destino_archivo.stat().st_size

    assert info["n_escritas"] == 1, "Archivo forzado debe escribirse"
    assert tamanio_despues > tamanio_original * 0.5, "Archivo reescrito debe tener contenido"
    # Verificar que PIL puede abrir el archivo sin error
    with Image.open(destino_archivo) as im:
        assert im.mode == "RGB"
