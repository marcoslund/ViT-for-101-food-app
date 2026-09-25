from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Food-101: se descarga y extrae en data/raw (ignorado por git, ~5 GB)
FOOD101_DIR = RAW_DATA_DIR / "food-101"
FOOD101_IMAGES_DIR = FOOD101_DIR / "images"
FOOD101_META_DIR = FOOD101_DIR / "meta"
FOOD101_CLASSES = FOOD101_META_DIR / "classes.txt"
FOOD101_URL = "https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"

# Artefactos del EDA (notebooks/1.0-eda-food101.ipynb). Estos SI estan versionados:
# definen el conjunto exacto de evaluacion, identico para todos los modelos.
BENCHMARK_SUBSET = PROCESSED_DATA_DIR / "benchmark_subset.csv"
BENCHMARK_MANIFEST = PROCESSED_DATA_DIR / "benchmark_subset_manifest.json"
CLASS_DIFFICULTY = PROCESSED_DATA_DIR / "class_difficulty.csv"

# Semilla unica del proyecto. El manifiesto del subset la registra.
SEED = 42

# Registry de modelos del benchmark: clave corta -> checkpoint de HuggingFace.
#
# Este es el UNICO lugar del proyecto donde se nombra una arquitectura. Todo lo demas
# (resolucion, normalizacion, orden de canales) se lee del AutoImageProcessor en runtime,
# porque escribirlo a mano es exactamente como se rompe la comparacion entre modelos.
# Sumar una arquitectura al benchmark es agregar una linea aca: los tests estan
# parametrizados sobre este dict y la cubren solos.
#
# vit = "google/vit-base-patch16-224-in21k", NO "google/vit-base-patch16-224":
# el EDA (notebooks/1.0-eda-food101.ipynb, "Proximos pasos") nombro el segundo, pero
# el registry usa el primero a proposito. "-in21k" es un encoder puro (ViTModel, sin
# cabeza de clasificacion), que es la base de fine-tuning mas limpia para 101 clases
# nuevas: from_pretrained("...-in21k", num_labels=101) inicializa una cabeza nueva
# desde cero, con el warning esperado de pesos no inicializados, y nada mas.
# "google/vit-base-patch16-224" (sin -in21k) SI trae una cabeza de 1000 clases
# (ImageNet-1k) entrenada; usarlo para 101 clases requiere from_pretrained(...,
# ignore_mismatched_sizes=True) para descartar esa cabeza en vez de reusarla, porque
# el shape 1000 no coincide con 101. Confundir los dos checkpoints en el codigo de
# entrenamiento (Fine-tuning de ViT, todavia sin hacer) es la forma de que esa llamada
# falle por shape mismatch o, peor, cargue pesos de clasificacion que no aplican.
MODELS = {
    "vit": "google/vit-base-patch16-224-in21k",
    "mobilevit": "apple/mobilevit-small",
    "deit": "facebook/deit-tiny-patch16-224",
}

# Artefactos del preprocesamiento. Los tres primeros se versionan: definen el split de
# validacion y los indices de clase, que tienen que ser identicos entre todas las corridas.
TRAIN_VAL_SPLIT = PROCESSED_DATA_DIR / "train_val_split.csv"
TRAIN_VAL_MANIFEST = PROCESSED_DATA_DIR / "train_val_split_manifest.json"
LABEL_MAP = PROCESSED_DATA_DIR / "label_map.json"

# Cache de imagenes reescaladas. Regenerable, no se versiona.
# 288 es la mayor resolucion de entrada que pide algun modelo del registry.
CACHE_SHORT_SIDE = 288
CACHE_JPEG_QUALITY = 95
CACHE_DIR = INTERIM_DATA_DIR / f"food-101-{CACHE_SHORT_SIDE}"
CACHE_MANIFEST = CACHE_DIR / "cache_manifest.json"

# Fraccion de train que se reserva para validacion (Food-101 no trae split de validacion).
VAL_FRACTION = 0.10

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
