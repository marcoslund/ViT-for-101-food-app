"""El paquete ``preprocessing`` reexporta su API publica en ``__init__.py``, detras de
un try/except que se traga ImportError si falta el extra "deep". Ningun otro test de
la suite importa esos nombres desde el paquete -- todos importan los submodulos
directamente -- asi que si el re-export se rompiera (typo, rename, un bug de
import-time genuino en loaders.py/policies.py/processors.py) nada lo notaria: el
except lo capturaria, __all__ quedaria en [] en silencio, y los 134 tests restantes
seguirian pasando igual."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("transformers")

from vit_for_101_food_app import preprocessing
from vit_for_101_food_app.preprocessing import (
    Food101Dataset,
    ProcessorSpec,
    available,
    build_dataloaders,
    build_transform,
    loaders,
    policies,
    processors,
    spec_for,
)

API_PUBLICA = {
    "Food101Dataset",
    "ProcessorSpec",
    "available",
    "build_dataloaders",
    "build_transform",
    "spec_for",
}


def test_all_expone_exactamente_la_api_publica_esperada():
    """El set esperado esta escrito a mano: si __all__ se trunca a [] (el fallback
    del except cuando algo se rompe) esto compara contra un conjunto no vacio y
    falla fuerte, en vez de comparar contra otro calculo que truncaria igual."""
    assert set(preprocessing.__all__) == API_PUBLICA


def test_cada_nombre_de_all_es_atributo_real_del_paquete():
    """set(__all__) solo dice que el nombre figura en la lista; esto confirma que
    tambien resuelve a un atributo real, no una entrada huerfana."""
    for nombre in preprocessing.__all__:
        assert hasattr(preprocessing, nombre), nombre


def test_los_reexports_son_los_mismos_objetos_que_los_submodulos():
    """El import top-level de arriba ya fallaria si estos nombres no existieran.
    Ademas de eso, comprobamos identidad: el re-export tiene que apuntar al mismo
    objeto que su submodulo, no a una copia o a un objeto con el mismo nombre."""
    assert Food101Dataset is loaders.Food101Dataset
    assert build_dataloaders is loaders.build_dataloaders
    assert ProcessorSpec is processors.ProcessorSpec
    assert spec_for is processors.spec_for
    assert build_transform is policies.build_transform
    assert available is policies.available
