# ViT for 101 Food App

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Benchmark de arquitecturas transformer de visión sobre **Food-101**, orientado a una decisión de
despliegue: **¿conviene correr el modelo en el dispositivo o servirlo detrás de una API?**

**Visión por Computadora III** — Carrera de Especialización en Inteligencia Artificial (CEIA) y
Maestría en Inteligencia Artificial (MIA), FIUBA.

**Integrantes:** Antonella Nerea Gambarte · Florencia Priscilla Vela · Marcos Lund

## La pregunta

Pensando en una hipotética aplicación para identificar y clasificar platos de comida, comparamos
dos modelos bajo dos regímenes de despliegue distintos:

| Régimen | Modelo | Resolución nativa | Costo dominante |
|---|---|---|---|
| **En dispositivo** | MobileViT | 256×256 | Cómputo local, memoria del teléfono |
| **Servido por API** | ViT (baseline) | 224×224 | Latencia de red, MB transferidos, costo por llamada |

La pregunta de fondo **no es cuál tiene mejor accuracy**, sino si el costo extra de cómputo,
latencia y transferencia del modelo servido por API **se paga en capacidad discriminativa real**.
Si la brecha entre ambos es plana a lo largo del ranking de dificultad de las clases, el modelo en
dispositivo es la elección correcta y la API no se justifica.

### Alcance

No desplegamos nada. Ambos modelos se entrenan y evalúan **localmente, sobre el mismo hardware**.
Lo que se mide es el **costo arquitectónico** —parámetros, FLOPs, latencia de inferencia— y la
conclusión sobre el régimen de despliegue se **argumenta** a partir de eso. La latencia de red y el
costo por llamada de la columna de arriba son **supuestos declarados, no mediciones**.

Más adelante pueden sumarse otras arquitecturas del lado de la API. El EDA está escrito para que
eso no obligue a rehacer nada: todo lo que mide vale para cualquier modelo servido remotamente a
224×224.

## El dataset

[**Food-101**](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) (Bossard et al., ECCV 2014)

| | |
|---|---|
| Clases | 101 |
| Imágenes | 101.000 (1.000 por clase) |
| Split oficial | 750 train / 250 test por clase |
| Resolución | lado máximo 512 px |
| Tamaño | ~5 GB comprimido |

El split de **test fue curado manualmente**; el de **train contiene ruido** intencional (colores
intensos y algunas etiquetas incorrectas), según los propios autores. Usamos los splits oficiales
sin modificar: cualquier split propio haría los resultados incomparables con la literatura.

## Estado del proyecto

- [x] **EDA** — [`notebooks/1.0-eda-food101.ipynb`](notebooks/1.0-eda-food101.ipynb)
- [x] **Preprocesamiento** — [`notebooks/2.0-preprocessing.ipynb`](notebooks/2.0-preprocessing.ipynb)
- [ ] Fine-tuning de MobileViT (en dispositivo)
- [ ] Fine-tuning y evaluación del ViT baseline (servido por API)
- [ ] Tabla comparativa y contraste de la hipótesis

## EDA

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marcoslund/ViT-for-101-food-app/blob/main/notebooks/1.0-eda-food101.ipynb)

El EDA no describe Food-101 en abstracto: cada sección responde una pregunta que condiciona el
diseño del benchmark.

| Sección | Pregunta | Decisión que habilita |
|---|---|---|
| 1. Estructura y splits | ¿Está balanceado? | Elección de métrica (top-1 vs macro-F1) |
| 2. Geometría | ¿Cuánta imagen se pierde al recortar a 224 / 256? | Política de preprocesamiento por modelo |
| 3. Costo de transporte | ¿Cuántos MB hay que mandar a la API? | Tamaño del payload y costo del experimento |
| 4. Integridad | ¿Hay imágenes rotas o en modos raros? | Lista de exclusión del dataloader |
| 5. Ruido y duplicados | ¿Hay mislabels o fuga train↔test? | Validez del set de evaluación |
| 6. Confusión entre clases | ¿Dónde está la dificultad real? | Hipótesis sobre dónde debería ganar el modelo servido por API |
| 7. Subset de benchmark | ¿Sobre qué imágenes exactas evaluamos? | Manifiesto reproducible y acotado en costo |
| 8. Conclusiones | — | Tabla hallazgo → decisión de diseño |

El notebook descarga y extrae Food-101 por su cuenta. La **sección 6** usa CLIP ViT-B/32 y necesita
GPU: en Colab, *Entorno de ejecución → Cambiar tipo de entorno → GPU (T4)*. CLIP es solo el
instrumento de medición del EDA; no compite en el benchmark.

### Costo computacional

Todo se controla desde la celda **0.1 Parámetros**. Las secciones globales (splits, tamaños de
archivo) usan el 100 % de los datos porque son baratas; las que decodifican píxeles trabajan sobre
una muestra estratificada por clase.

| Parámetro | Default | Efecto |
|---|---|---|
| `N_PER_CLASS_INSPECT` | 60 | Integridad, color, nitidez y hashes (§4-§5) |
| `N_PER_CLASS_CLIP` | 30 | Embeddings CLIP (§6) |
| `N_PER_CLASS_BENCH` | 25 | Tamaño del subset de evaluación (§7) |

Si vas a correrlo más de una vez, poné `USE_DRIVE_CACHE = True`: el `.tar.gz` de 5 GB queda
cacheado en Drive y las sesiones siguientes no lo vuelven a descargar.

## Artefactos del EDA

El notebook escribe en `data/processed/`, y **estos tres archivos se commitean** porque son la
entrada del notebook de benchmark:

| Archivo | Contenido |
|---|---|
| `benchmark_subset.csv` | Manifiesto de evaluación: imagen, clase, margen de dificultad, tercil, bytes |
| `benchmark_subset_manifest.json` | Semilla, exclusiones, conteos y `sha256` del CSV |
| `class_difficulty.csv` | Ranking de dificultad de las 101 clases con su vecino más confundible |

Todos los modelos del benchmark evalúan **exactamente el mismo archivo**, sin volver a muestrear.
Esa es la única forma de que las diferencias medidas sean diferencias entre arquitecturas y no
entre muestras. El `sha256` del manifiesto permite verificarlo meses después.

`make test` valida ese contrato: que el CSV coincida con su hash, que el subset siga balanceado en
25 imágenes × 101 clases, que respete las exclusiones y que los terciles sean consistentes con el
ranking de dificultad.

### Guardarlos desde Colab

*Guardar una copia en GitHub* commitea **solo el `.ipynb`**. Los artefactos se generan en
`/content/outputs`, que es disco efímero del runtime. La **sección 9** del notebook los persiste en
`data/processed/` del repo, usando un token leído de los Secrets de Colab.

Configuración, una sola vez:

1. Crear un [fine-grained PAT](https://github.com/settings/personal-access-tokens/new) con
   *Repository access* limitado a este repo y permiso **Contents: Read and write**.
2. En Colab: barra izquierda → icono de llave → secreto `GITHUB_TOKEN` con acceso al notebook.

La celda toca únicamente `data/processed/`, así que no compite con el guardado del notebook, y es
idempotente: si no cambió nada, no crea un commit vacío.

## Preprocesamiento

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marcoslund/ViT-for-101-food-app/blob/main/notebooks/2.0-preprocessing.ipynb)

[`notebooks/2.0-preprocessing.ipynb`](notebooks/2.0-preprocessing.ipynb) ejecuta las decisiones de
la tabla hallazgo → decisión del EDA con el paquete `vit_for_101_food_app.preprocessing`: split de
validación, cache de imágenes y los transforms que replican el `AutoImageProcessor` nativo de cada
modelo. Se verificó localmente contra un subset real de 3 clases de Food-101; el dataset completo
(~4.7 GB) está pensado para correr en Colab.

### Regla central: nunca escribir la geometría a mano

Cada modelo consume su propio `AutoImageProcessor` de HuggingFace, leído en runtime por
`preprocessing.processors.spec_for`. `preprocessing.policies.build_transform` reconstruye esa misma
geometría operando sobre tensores — necesario para poder aplicar augmentation con
`torchvision.transforms.v2` — y `make verify` comprueba que ambos coinciden. En la corrida de
verificación local la diferencia máxima medida fue `1.19e-07` para ViT y `0.0` para MobileViT
(orden del épsilon de `float32`, no una aproximación visual).

Esa equivalencia es lo que permite dejar a la vista una diferencia real entre modelos sin
"corregirla": MobileViT recibe sus canales en **BGR** y **no normaliza**, mientras que ViT recibe
**RGB** normalizado a `[-1, 1]`. Igualar esas dos filas "para que quede prolijo" es exactamente el
bug que documenta la sección 6 del notebook — el modelo entrenaría peor sin que nada falle.

### Artefactos versionados

Además de los tres artefactos del EDA, `data/processed/` versiona el split de validación y el mapa
de etiquetas — el mismo tipo de contrato: todos los modelos tienen que entrenar y validar sobre los
mismos índices.

| Archivo | Contenido |
|---|---|
| `train_val_split.csv` | Split de validación recortado de train (estratificado por clase, semilla fija) |
| `train_val_split_manifest.json` | Semilla, `val_fraction`, conteos por clase y `sha256` del CSV |
| `label_map.json` | `id2label` / `label2id`, fijados por el orden de `meta/classes.txt` |

El split de **test nunca se toca**: `preprocessing.splits.load_split("test")` lee directo de
`meta/test.txt`, no pasa por el CSV.

### `make preprocess`

```bash
make preprocess   # = make data split cache verify
```

| Target | Comando | Qué hace |
|---|---|---|
| `make data` | `dataset.py download` | Descarga y extrae Food-101 (~4.7 GB, idempotente) |
| `make split` | `dataset.py split` | Escribe `train_val_split.csv`, su manifiesto y `label_map.json` |
| `make cache` | `dataset.py cache` | Reescala todo el dataset al lado corto configurado (`CACHE_SHORT_SIDE = 288`) en `data/interim/`, regenerable y no versionado |
| `make verify` | `features.py verify` | Compara los transforms `eval` contra el `AutoImageProcessor` real de cada modelo |

`features.py preview` (fuera de `make preprocess`) escribe una grilla antes/después por modelo en
`reports/figures/`, para inspección visual.

## Setup local

El proyecto usa [uv](https://docs.astral.sh/uv/). Python ≥ 3.11.

```bash
make create_environment     # uv venv
source .venv/bin/activate
make requirements           # uv sync
```

`torch` y `transformers` pesan ~2.5 GB y solo hacen falta para la §6 del EDA, para el
preprocesamiento (`make verify`, y las secciones 3 en adelante del notebook de preprocesamiento) y
para el entrenamiento. Van en un extra aparte:

```bash
uv sync --extra deep
```

Después:

```bash
jupyter lab notebooks/1.0-eda-food101.ipynb
jupyter lab notebooks/2.0-preprocessing.ipynb
```

El dataset se descarga a `data/raw/` (ignorado por git). En Colab no hace falta instalar nada: el
notebook resuelve por su cuenta lo que falte.

Otros comandos: `make lint`, `make format`, `make test`.

## Estructura del proyecto

Generada con [cookiecutter-data-science](https://cookiecutter-data-science.drivendata.org/) v2.3.0.

```
├── LICENSE            <- MIT
├── Makefile           <- Comandos: make requirements, make test, make lint
├── README.md
├── data
│   ├── external       <- Datos de terceros
│   ├── interim        <- Cache de imágenes reescaladas (regenerable, no versionado)
│   ├── processed      <- Artefactos versionados del EDA y del preprocesamiento (ver arriba)
│   └── raw            <- Food-101 sin tocar (~5 GB, ignorado por git)
│
├── docs               <- Documentación del proyecto
│
├── models             <- Modelos entrenados y serializados, predicciones, métricas
│
├── notebooks          <- Notebooks. Convención: número de orden + descripción,
│                         p. ej. `1.0-eda-food101.ipynb`, `2.0-preprocessing.ipynb`
│
├── pyproject.toml     <- Metadatos, dependencias y config de ruff
│
├── references         <- Diccionarios de datos, manuales, material explicativo
│
├── reports            <- Análisis generado (HTML, PDF, LaTeX)
│   └── figures        <- Gráficos para los informes
│
├── tests              <- Tests de integridad de los artefactos del EDA y del preprocesamiento
│
└── vit_for_101_food_app   <- Código fuente del proyecto
    ├── config.py               <- Rutas, semilla y resoluciones de los modelos
    ├── dataset.py              <- CLI: descarga, split de validación y cache
    ├── features.py             <- CLI: verificación e inspección visual de los transforms
    ├── modeling
    │   ├── predict.py          <- Inferencia (local y por API)
    │   └── train.py            <- Fine-tuning de los modelos
    ├── plots.py                <- Visualizaciones
    └── preprocessing
        ├── raw.py              <- Índice canónico de Food-101 y descarga
        ├── splits.py           <- Split de validación y mapa de etiquetas
        ├── cache.py            <- Cache de imágenes reescaladas
        ├── processors.py       <- AutoImageProcessor -> ProcessorSpec, por modelo
        ├── policies.py         <- Transforms de augmentation y evaluación
        └── loaders.py          <- Dataset y DataLoaders
```

## Citación

```bibtex
@inproceedings{bossard14,
  title     = {Food-101 -- Mining Discriminative Components with Random Forests},
  author    = {Bossard, Lukas and Guillaumin, Matthieu and Van Gool, Luc},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2014}
}
```

El código de este repositorio está bajo licencia MIT (ver [`LICENSE`](LICENSE)). El uso del
**dataset** está sujeto al `license_agreement.txt` incluido en la descarga (uso académico / no
comercial). Este repositorio **no distribuye las imágenes**, solo el código que las analiza.
