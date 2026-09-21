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
FOOD101_URL = "https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"

# Artefactos del EDA (notebooks/1.0-eda-food101.ipynb). Estos SI estan versionados:
# definen el conjunto exacto de evaluacion, identico para todos los modelos.
BENCHMARK_SUBSET = PROCESSED_DATA_DIR / "benchmark_subset.csv"
BENCHMARK_MANIFEST = PROCESSED_DATA_DIR / "benchmark_subset_manifest.json"
CLASS_DIFFICULTY = PROCESSED_DATA_DIR / "class_difficulty.csv"

# Semilla unica del proyecto. El manifiesto del subset la registra.
SEED = 42

# Resoluciones nativas de los modelos del benchmark:
# nombre -> (resize del lado corto, tamano del center-crop)
PIPELINES = {
    "ViT (224)": (256, 224),
    "MobileViT (256)": (288, 256),
}

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
