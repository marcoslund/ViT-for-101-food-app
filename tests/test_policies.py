"""El riesgo central del diseno: que los transforms construidos a partir del
ProcessorSpec se desvien de lo que el AutoImageProcessor hace de verdad. El primer test
de este archivo es la red de seguridad que cierra ese riesgo."""

import dataclasses
import pickle

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


def _imagen_en_modo(modo: str) -> Image.Image:
    """Construye una imagen PIL en uno de los tres modos no-RGB que el hallazgo #7 del
    EDA encontro en Food-101: escala de grises (L), paleta (P) y CMYK. Sin conversion a
    RGB, cualquiera de los tres produce un tensor con la cantidad de canales equivocada
    -- silenciosamente para mobilevit, que no tiene otra parte del pipeline que dependa
    de la cantidad de canales para fallar ruidoso."""
    rng = np.random.default_rng(7)
    if modo == "L":
        arr = rng.integers(0, 256, size=(400, 600), dtype=np.uint8)
        return Image.fromarray(arr, mode="L")
    if modo == "P":
        arr = rng.integers(0, 256, size=(400, 600), dtype=np.uint8)
        return Image.fromarray(arr, mode="L").convert("P")
    if modo == "CMYK":
        arr = rng.integers(0, 256, size=(400, 600, 4), dtype=np.uint8)
        return Image.fromarray(arr, mode="CMYK")
    raise ValueError(f"modo no soportado en el test: {modo!r}")


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


@pytest.mark.parametrize("key", list(MODELS))
@pytest.mark.parametrize("modo", ["L", "P", "CMYK"])
def test_imagen_no_rgb_produce_la_resolucion_nativa(key, modo):
    """La conversion a RGB es incondicional y a nivel PIL (hallazgo #7 del EDA:
    .convert("RGB") siempre). v2.RGB() por si solo no alcanza -- no sabe llevar CMYK (4
    canales) a 3 --, por eso la conversion tiene que pasar por PIL. Sin ella, mobilevit
    -- que no normaliza -- devolveria un tensor con la cantidad de canales equivocada
    sin ningun error, y eso es exactamente la degradacion silenciosa que este modulo
    existe para evitar. Cubre los tres modos del hallazgo para los dos modelos: ninguna
    combinacion queda exceptuada."""
    spec = processors.spec_for(key)
    t = policies.build_transform(spec, "eval")(_imagen_en_modo(modo))
    assert t.shape == spec.input_shape


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


@pytest.mark.parametrize("key", list(MODELS))
def test_resize_only_es_identico_a_eval_solo_si_no_hay_crop(key, imagen):
    """La ablacion de la seccion 7 #5 del EDA: sin CenterCrop no se pierde area.

    defecto 3 del task brief: el test original solo corria sobre mobilevit, asi que
    nunca ejercito el caso ViT (resize_shortest is None), donde 'resize_only' y 'eval'
    son bit a bit el mismo pipeline -- no hay CenterCrop que ablacionar. Si alguien
    programara un 'resize_only' para los dos modelos, ViT devolveria un duplicado
    exacto de su corrida 'eval' y nada lo señalaria sin este test parametrizado.

    La propiedad real por modelo: identico a 'eval' si y solo si
    spec.resize_shortest is None; distinto en caso contrario.
    """
    spec = processors.spec_for(key)
    t = policies.build_transform(spec, "resize_only")(imagen)
    eval_t = policies.build_transform(spec, "eval")(imagen)
    assert t.shape == eval_t.shape

    if spec.resize_shortest is None:
        torch.testing.assert_close(t, eval_t)
    else:
        assert not torch.allclose(t, eval_t), "recortar y no recortar no pueden dar lo mismo"


@pytest.mark.parametrize("key", list(MODELS))
def test_strong_agrega_randaugment_y_randomerasing_sobre_standard(key):
    """Item diferido del task brief: nada probaba que 'strong' realmente contuviera
    RandAugment y RandomErasing, solo que la salida fuera estocastica -- algo que
    'standard' (RandomResizedCrop + flip) ya garantiza por si solo. Un typo que tirara
    ``after_tensor`` de la Policy (ver policies.POLICIES["strong"]) dejaria 'strong'
    identica a 'standard' en contenido, pero seguiria siendo estocastica: los tests
    existentes de aleatoriedad no lo hubieran notado. Esto inspecciona los tipos reales
    de los pasos compuestos, no solo si el resultado varia."""
    spec = processors.spec_for(key)
    standard = policies.build_transform(spec, "standard")
    strong = policies.build_transform(spec, "strong")

    tipos_standard = [type(paso).__name__ for paso in standard.transforms]
    tipos_strong = [type(paso).__name__ for paso in strong.transforms]

    assert "RandomResizedCrop" in tipos_standard
    assert "RandomHorizontalFlip" in tipos_standard
    assert "RandAugment" not in tipos_standard
    assert "RandomErasing" not in tipos_standard

    assert "RandAugment" in tipos_strong
    assert "RandomErasing" in tipos_strong
    # 'strong' es 'standard' con augmentation extra, no una geometria distinta.
    assert set(tipos_standard) <= set(tipos_strong)


def test_politica_desconocida_es_un_error_explicito():
    with pytest.raises(KeyError, match="mixup"):
        policies.build_transform(processors.spec_for("vit"), "mixup")


def test_available_lista_las_politicas():
    assert set(policies.available()) == {"eval", "standard", "strong", "resize_only"}


@pytest.mark.parametrize("key", list(MODELS))
@pytest.mark.parametrize("policy", list(policies.POLICIES))
def test_el_transform_compuesto_es_picklable(key, policy):
    """DataLoader con num_workers>0 tiene que picklear el Dataset -- transform incluido
    -- para mandarlo a los worker processes. Con start method "spawn" (default en macOS
    y Windows) un v2.Lambda(lambda ...) rompe con AttributeError al picklear, y con
    "fork" (Linux) el bug queda invisible porque el worker hereda la memoria del padre.
    Esto no depende del start method real de la maquina: pickle.dumps alcanza para
    detectarlo en cualquier lado."""
    spec = processors.spec_for(key)
    t = policies.build_transform(spec, policy)
    pickle.dumps(t)
