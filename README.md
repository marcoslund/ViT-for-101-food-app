# ViT for 101 Food App

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Benchmark de arquitecturas transformer de visión sobre **Food-101**, orientado a una pregunta de
diseño: **¿qué arquitectura conviene para una aplicación con recursos acotados —posiblemente un
dispositivo móvil— sin resignar capacidad de clasificación?**

**Visión por Computadora III** — Carrera de Especialización en Inteligencia Artificial (CEIA) y
Maestría en Inteligencia Artificial (MIA), FIUBA.

**Integrantes:** Antonella Nerea Gambarte · Florencia Priscilla Vela · Marcos Lund

## La pregunta

Pensando en una hipotética aplicación para identificar y clasificar platos de comida a partir de una
foto tomada con un dispositivo móvil, comparamos **varias arquitecturas transformer de visión con
distintos compromisos entre capacidad y eficiencia**. Se evalúan arquitecturas pensadas para
dispositivos con recursos limitados (p. ej. MobileViT) y se usa un **ViT estándar como referencia de
mayor capacidad**.

| Rol | Modelo | Resolución nativa |
|---|---|---|
| **Candidato liviano** | MobileViT | 256×256 |
| **Referencia** | ViT (baseline) | 224×224 |

(El registry `config.MODELS` admite sumar más arquitecturas —Swin, DeiT, etc.— sin tocar el pipeline.)

La comparación busca responder dos preguntas:

1. ¿Cuánto rendimiento de clasificación se pierde —si es que se pierde— al usar arquitecturas más
   livianas para este caso?
2. ¿En qué clases se concentra esa diferencia: es pareja entre las 101 categorías o se acumula en los
   platos visualmente más difíciles de distinguir?

Si la brecha entre una arquitectura liviana y la referencia es plana a lo largo del ranking de
dificultad de las clases, la arquitectura liviana es la elección correcta para el dispositivo.

### Alcance

No desplegamos nada. Todos los modelos se entrenan y evalúan **localmente, sobre el mismo hardware**.
Se miden dos familias de métricas: **desempeño** (top-1 accuracy y macro-F1) y **costo
arquitectónico** (parámetros, FLOPs, tamaño del modelo y latencia de inferencia). La adecuación a un
dispositivo con recursos limitados se **argumenta** a partir de esas mediciones.

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
- [ ] Fine-tuning de MobileViT (candidato liviano)
- [ ] Fine-tuning y evaluación del ViT baseline (referencia de mayor capacidad)
- [ ] Tabla comparativa y contraste de la hipótesis

## EDA

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marcoslund/ViT-for-101-food-app/blob/main/notebooks/1.0-eda-food101.ipynb)

El EDA no describe Food-101 en abstracto: cada sección responde una pregunta que condiciona el
diseño del benchmark.

| Sección | Pregunta | Decisión que habilita |
|---|---|---|
| 1. Estructura y splits | ¿Está balanceado? | Elección de métrica (top-1 vs macro-F1) |
| 2. Geometría | ¿Cuánta imagen se pierde al recortar a 224 / 256? | Política de preprocesamiento por modelo |
| 3. Integridad | ¿Hay imágenes rotas o en modos raros? | Lista de exclusión del dataloader |
| 4. Ruido y duplicados | ¿Hay mislabels o fuga train↔test? | Validez del set de evaluación |
| 5. Confusión entre clases | ¿Dónde está la dificultad real? | Hipótesis sobre en qué clases la mayor capacidad debería marcar diferencia |
| 6. Subset de benchmark | ¿Sobre qué imágenes exactas evaluamos? | Manifiesto reproducible y acotado en costo |
| 7. Conclusiones | — | Tabla hallazgo → decisión de diseño |

(La sección 0, Setup, no está en la tabla: prepara el entorno, no responde una pregunta de diseño.)

El notebook descarga y extrae Food-101 por su cuenta. La **sección 5** usa CLIP ViT-B/32 y necesita
GPU: en Colab, *Entorno de ejecución → Cambiar tipo de entorno → GPU (T4)*. CLIP es solo el
instrumento de medición del EDA; no compite en el benchmark.

### Costo computacional

Todo se controla desde la celda **0.1 Parámetros**. Las secciones globales (splits, tamaños de
archivo) usan el 100 % de los datos porque son baratas; las que decodifican píxeles trabajan sobre
una muestra estratificada por clase.

| Parámetro | Default | Efecto |
|---|---|---|
| `N_PER_CLASS_INSPECT` | 60 | Integridad, color, nitidez y hashes (secciones 3-4) |
| `N_PER_CLASS_CLIP` | 30 | Embeddings CLIP (sección 5) |
| `N_PER_CLASS_BENCH` | 25 | Tamaño del subset de evaluación (sección 6) |

Si vas a correrlo más de una vez, poné `USE_DRIVE_CACHE = True`: el `.tar.gz` de 5 GB queda
cacheado en Drive y las sesiones siguientes no lo vuelven a descargar.

## Artefactos del EDA

El notebook escribe en `data/processed/`, y **estos tres archivos se commitean** porque son la
entrada del notebook de benchmark:

| Archivo | Contenido |
|---|---|
| `benchmark_subset.csv` | Manifiesto de evaluación: imagen, clase, margen de dificultad, tercil |
| `benchmark_subset_manifest.json` | Semilla, exclusiones, conteos y `sha256` del CSV |
| `class_difficulty.csv` | Ranking de dificultad de las 101 clases con su vecino más confundible |

Todos los modelos del benchmark evalúan **exactamente el mismo archivo**, sin volver a muestrear.
Esa es la única forma de que las diferencias medidas sean diferencias entre arquitecturas y no
entre muestras. El `sha256` del manifiesto permite verificarlo meses después.

`make test` valida ese contrato: que el CSV coincida con su hash, que el subset siga balanceado en
25 imágenes × 101 clases, que respete las exclusiones y que los terciles sean consistentes con el
ranking de dificultad.

### Guardarlos desde Colab

*Guardar una copia en GitHub* commitea **solo el `.ipynb`**. Los artefactos (`data/processed/*.csv`,
`*.json`) se generan en disco efímero del runtime de Colab y no hay ninguna celda que los suba por su
cuenta: el propio notebook no tiene ningún paso que hable con git o con un token.

Lo que se hace en la práctica, y lo que muestra el historial de commits de este repo (`Corrida del
EDA desde Colab`, `Artefactos del EDA regenerados en la corrida de Colab`): después de correr el
notebook completo en Colab, se descargan los archivos de `data/processed/` (panel de archivos de
Colab, o vía Drive si `USE_DRIVE_CACHE = True`) y se commitean desde el checkout local, como
cualquier otro cambio. No hace falta crear tokens ni configurar secrets para esto.

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

El `AutoImageProcessor` real de `google/vit-base-patch16-224-in21k` hace un **resize cuadrado
directo** a 224×224 (`do_center_crop=False`), no `Resize(256) → CenterCrop(224)`. La sección 2 del
EDA todavía asume ese segundo pipeline por analogía con otros ViT; quedó desactualizada frente a lo
que mide `preprocessing.processors.ficha_tecnica()` (ver `notebooks/2.0-preprocessing.ipynb`,
sección 3) y falta revisarla — no se toca acá porque el EDA está fuera de alcance de este cambio.

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

## Entrenamiento y benchmark

Todo el código de modelo y entrenamiento vive en `vit_for_101_food_app.modeling` y es
**genérico sobre `model_key`** (una clave del registry `config.MODELS`): sumar una
arquitectura al benchmark es agregar una línea al registry, no escribir un pipeline nuevo.

| Módulo | Responsabilidad |
|---|---|
| `training.py` | Construcción del modelo, `TrainingRecipe` (idéntica para todos), `Trainer` de HuggingFace y `resolve_batch_plan` (batch/acumulación/checkpointing según resolución y GPU) |
| `evaluation.py` | Predicciones por imagen, métricas por tercil, reporte por clase, confusiones, FLOPs, tamaño de los pesos (fp32/fp16/int8) y latencia en GPU/CPU —incluida la medición del lado del dispositivo con el modelo **cuantizado a int8** en CPU—, **el mismo código para todos los modelos** |
| `benchmark.py` | Orquestación: verifica artefactos, entrena, evalúa y escribe `metrics.json` + CSV |

Hay **dos formas de correr el mismo pipeline**, que miden exactamente lo mismo:

- **Notebooks** (`notebooks/3.1-mobilevit.ipynb`, `3.2-swin.ipynb`, `3.3-deit.ipynb`) —
  celda por celda, con salidas visibles para el informe. Los tres son idénticos salvo
  `MODEL_KEY`; están pensados para Colab (secciones 0.2 y 0.3 arman el entorno y los datos).
- **CLI headless** — para una corrida desatendida (Kaggle, VM):

  ```bash
  python -m vit_for_101_food_app.modeling.train --model swin      # benchmark completo
  python -m vit_for_101_food_app.modeling.train --model mobilevit --resume
  python -m vit_for_101_food_app.modeling.predict --model swin --model-dir models/swin/full/best
  ```

Cada corrida escribe checkpoints en `models/<model>/full/` y resultados livianos
(`metrics.json`, predicciones, tabla por tercil) en `reports/results/<model>/`, con el
mismo esquema para todos los modelos: la tabla comparativa del benchmark se arma juntando
los `metrics.json` sin reentrenar.

## Setup local

El proyecto usa [uv](https://docs.astral.sh/uv/). Python ≥ 3.11.

```bash
make create_environment     # uv venv
source .venv/bin/activate
make requirements           # uv sync
```

`torch` y `transformers` pesan ~2.5 GB y solo hacen falta para la sección 5 del EDA, para el
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
    │   ├── training.py         <- Modelo, receta y Trainer, genéricos sobre model_key
    │   ├── evaluation.py       <- Protocolo de evaluación común (predicciones, terciles, FLOPs, latencia)
    │   ├── benchmark.py        <- Orquestación: une preprocessing + training + evaluation
    │   ├── train.py            <- CLI headless: corre el benchmark completo de un modelo
    │   └── predict.py          <- CLI de inferencia sobre un modelo ya entrenado
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
