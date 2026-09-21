"""El riesgo central del diseno: que los transforms construidos a partir del
ProcessorSpec se desvien de lo que el AutoImageProcessor hace de verdad. El primer test
de este archivo es la red de seguridad que cierra ese riesgo."""

import dataclasses

import numpy as np
from PIL import Image
import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("transformers")

import torch

from vit_for_101_food_app.config import MODELS
from vit_for_101_food_app.preprocessing import policies, processors


@pytest.fixture
def imagen():
    """Rectangular y con canales bien distintos: detecta recortes y permutaciones."""
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 256, size=(400, 600, 3), dtype=np.uint8)
    arr[..., 0] = np.clip(arr[..., 0] // 4 + 180, 0, 255)
    arr[..., 2] = arr[..., 2] // 4
    return Image.fromarray(arr)


@pytest.mark.parametrize("key", list(MODELS))
def test_politica_eval_es_identica_al_processor(key, imagen):
    """LA garantia del diseno. Si HuggingFace cambia un default, esto falla ruidosamente
    en vez de degradar el benchmark en silencio."""
    spec = processors.spec_for(key)
    nuestro = policies.build_transform(spec, "eval")(imagen)
    del_processor = processors.load_processor(key)(imagen, return_tensors="pt")["pixel_values"][0]
    torch.testing.assert_close(nuestro, del_processor, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("key", list(MODELS))
@pytest.mark.parametrize("policy", ["eval", "standard", "strong", "resize_only"])
def test_toda_politica_produce_la_resolucion_nativa(key, policy, imagen):
    spec = processors.spec_for(key)
    t = policies.build_transform(spec, policy)(imagen)
    assert t.shape == spec.input_shape
    assert t.dtype == torch.float32


def test_mobilevit_queda_en_bgr_y_sin_normalizar(imagen):
    """Regresion de la gotcha #4 del EDA, sobre el tensor y no sobre el config."""
    t = policies.build_transform(processors.spec_for("mobilevit"), "eval")(imagen)
    assert t.min() >= 0.0 and t.max() <= 1.0, "no tiene que estar normalizado a [-1,1]"

    sin_flip = policies.build_transform(
        dataclasses.replace(processors.spec_for("mobilevit"), do_flip_channel_order=False),
        "eval",
    )(imagen)
    torch.testing.assert_close(t, sin_flip.flip(-3))
    assert not torch.allclose(t, sin_flip), "los canales tienen que estar invertidos"


def test_vit_si_esta_normalizado(imagen):
    t = policies.build_transform(processors.spec_for("vit"), "eval")(imagen)
    assert t.min() < 0.0, "con mean/std 0.5 el rango tiene que cruzar el cero"


def test_la_politica_no_decide_la_normalizacion(imagen):
    """Misma politica geometrica, distinto modelo: la cola de tensor la pone el spec."""
    vit = policies.build_transform(processors.spec_for("vit"), "resize_only")(imagen)
    mvit = policies.build_transform(processors.spec_for("mobilevit"), "resize_only")(imagen)
    assert vit.min() < 0.0
    assert mvit.min() >= 0.0


def test_eval_es_determinista(imagen):
    t = policies.build_transform(processors.spec_for("vit"), "eval")
    torch.testing.assert_close(t(imagen), t(imagen))


def test_standard_es_estocastica(imagen):
    torch.manual_seed(0)
    t = policies.build_transform(processors.spec_for("vit"), "standard")
    assert not torch.allclose(t(imagen), t(imagen))


def test_resize_only_no_recorta(imagen):
    """La ablacion de la seccion 8 #5 del EDA: sin CenterCrop no se pierde area."""
    spec = processors.spec_for("mobilevit")
    t = policies.build_transform(spec, "resize_only")(imagen)
    eval_t = policies.build_transform(spec, "eval")(imagen)
    assert t.shape == eval_t.shape
    assert not torch.allclose(t, eval_t), "recortar y no recortar no pueden dar lo mismo"


def test_politica_desconocida_es_un_error_explicito():
    with pytest.raises(KeyError, match="mixup"):
        policies.build_transform(processors.spec_for("vit"), "mixup")


def test_available_lista_las_politicas():
    assert set(policies.available()) == {"eval", "standard", "strong", "resize_only"}
