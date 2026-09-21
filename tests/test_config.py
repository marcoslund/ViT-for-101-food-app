"""El registry MODELS es el punto de extension del benchmark: agregar una arquitectura
tiene que ser agregar una entrada aca y nada mas. Estos tests fijan ese contrato."""

from pathlib import Path

from vit_for_101_food_app import config


def test_registry_tiene_los_modelos_confirmados():
    assert set(config.MODELS) == {"vit", "mobilevit"}


def test_cada_modelo_apunta_a_un_checkpoint_no_vacio():
    for key, checkpoint in config.MODELS.items():
        assert isinstance(checkpoint, str) and "/" in checkpoint, key


def test_las_claves_del_registry_sirven_como_nombre_de_archivo():
    for key in config.MODELS:
        assert key.isidentifier() and key.islower()


def test_pipelines_ya_no_existe():
    """Duplicaba a mano las resoluciones. Ahora se leen del AutoImageProcessor."""
    assert not hasattr(config, "PIPELINES")


def test_rutas_de_artefactos_bajo_el_proyecto():
    rutas = [
        config.TRAIN_VAL_SPLIT,
        config.TRAIN_VAL_MANIFEST,
        config.LABEL_MAP,
        config.CACHE_DIR,
        config.CACHE_MANIFEST,
        config.FOOD101_CLASSES,
    ]
    for ruta in rutas:
        assert isinstance(ruta, Path)
        assert config.PROJ_ROOT in ruta.parents


def test_artefactos_versionados_van_a_processed():
    for ruta in (config.TRAIN_VAL_SPLIT, config.TRAIN_VAL_MANIFEST, config.LABEL_MAP):
        assert ruta.parent == config.PROCESSED_DATA_DIR


def test_el_cache_es_regenerable_y_va_a_interim():
    assert config.INTERIM_DATA_DIR in config.CACHE_DIR.parents
    assert config.CACHE_MANIFEST.parent == config.CACHE_DIR


def test_parametros_del_cache_y_del_split():
    assert 1 <= config.CACHE_JPEG_QUALITY <= 100
    assert 0 < config.VAL_FRACTION < 1
    assert config.SEED == 42
