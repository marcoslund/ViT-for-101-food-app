"""ProcessorSpec es el unico lugar del proyecto que conoce las resoluciones y las
estadisticas de normalizacion de cada modelo. Si se desvia de lo que el processor hace
de verdad, el benchmark se degrada en silencio."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from dataclasses import FrozenInstanceError

from vit_for_101_food_app.config import MODELS
from vit_for_101_food_app.preprocessing import processors


@pytest.mark.parametrize("key", list(MODELS))
def test_spec_se_construye_para_todo_el_registry(key):
    spec = processors.spec_for(key)
    assert spec.key == key
    assert spec.checkpoint == MODELS[key]


@pytest.mark.parametrize("key", list(MODELS))
def test_target_size_es_un_entero_positivo(key):
    assert processors.spec_for(key).target_size > 0


@pytest.mark.parametrize("key", list(MODELS))
def test_si_normaliza_entonces_declara_mean_y_std(key):
    spec = processors.spec_for(key)
    if spec.do_normalize:
        assert spec.image_mean is not None and spec.image_std is not None
        assert len(spec.image_mean) == len(spec.image_std) == 3


@pytest.mark.parametrize("key", list(MODELS))
def test_spec_es_inmutable(key):
    """Un spec mutable permitiria que un consumidor le cambie la resolucion a otro."""
    spec = processors.spec_for(key)
    with pytest.raises(FrozenInstanceError):
        spec.do_normalize = not spec.do_normalize


def test_spec_del_vit_coincide_con_el_config_verificado():
    spec = processors.spec_for("vit")
    assert spec.do_resize is True
    assert spec.size == {"height": 224, "width": 224}
    assert spec.do_center_crop is False
    assert spec.target_size == 224
    assert spec.do_normalize is True
    assert spec.do_flip_channel_order is False
    assert spec.resize_shortest is None


def test_spec_del_mobilevit_coincide_con_el_config_verificado():
    spec = processors.spec_for("mobilevit")
    assert spec.size == {"shortest_edge": 288}
    assert spec.resize_shortest == 288
    assert spec.do_center_crop is True
    assert spec.crop_size == {"height": 256, "width": 256}
    assert spec.target_size == 256
    assert spec.do_flip_channel_order is True


def test_mobilevit_no_normaliza_aunque_declare_mean_y_std():
    """LA trampa (gotcha #4 del EDA, precisada).

    El config de MobileViT en transformers 5.x declara image_mean e image_std = 0.5,
    pero do_normalize es None y la normalizacion NO se aplica. Cualquier codigo que
    haga `if spec.image_mean: normalize(...)` normaliza MobileViT por error, y el bug
    es silencioso: los tensores siguen teniendo la forma correcta.
    """
    spec = processors.spec_for("mobilevit")
    assert spec.do_normalize is False
    assert spec.image_mean is not None, "el config SI los declara; de ahi viene la trampa"


def test_do_normalize_nunca_es_none():
    """None se coacciona a False, no se propaga como valor ambiguo."""
    for key in MODELS:
        assert isinstance(processors.spec_for(key).do_normalize, bool)


def test_spec_for_esta_cacheada():
    assert processors.spec_for("vit") is processors.spec_for("vit")


def test_all_specs_cubre_el_registry():
    assert set(processors.all_specs()) == set(MODELS)


def test_clave_desconocida_es_un_error_explicito():
    with pytest.raises(KeyError, match="mobilenet"):
        processors.spec_for("mobilenet")
