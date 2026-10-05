# ViT for 101 Food App

![](https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter)

Benchmark de arquitecturas transformer de visión sobre **Food-101**, orientado a una pregunta de diseño: **¿qué arquitectura conviene para una aplicación con recursos acotados (posiblemente un dispositivo móvil) sin resignar capacidad de clasificación?**

**Visión por Computadora III**, Carrera de Especialización en Inteligencia Artificial (CEIA) y Maestría en Inteligencia Artificial (MIA), FIUBA.

**Integrantes:** Antonella Nerea Gambarte, Florencia Priscilla Vela, Marcos Lund

## La pregunta

Pensando en una hipotética aplicación para identificar y clasificar platos de comida a partir de una
foto tomada con un dispositivo móvil, comparamos **varias arquitecturas transformer de visión con
distintos compromisos entre capacidad y eficiencia**: MobileViT-S, DeiT-Ti, Swin-T y ViT-B/16.


| Modelo      | Checkpoint                               | Parámetros | Resolución nativa |
| ----------- | ---------------------------------------- | ---------- | ----------------- |
| MobileViT-S | `apple/mobilevit-small`                  | 5,0 M      | 256×256           |
| DeiT-Ti     | `facebook/deit-tiny-patch16-224`         | 5,5 M      | 224×224           |
| Swin-T      | `microsoft/swin-tiny-patch4-window7-224` | 27,6 M     | 224×224           |
| ViT-B/16    | `google/vit-base-patch16-224-in21k`      | 85,9 M     | 224×224           |


Las cuatro viven en el registry `config.MODELS`; sumar una arquitectura es agregar una línea ahí, no
tocar el pipeline.

La comparación busca responder dos preguntas:

1. ¿Cuánto rendimiento de clasificación se pierde, si es que se pierde, al usar arquitecturas más
  livianas para este caso?
2. ¿En qué clases se concentra esa diferencia: es pareja entre las 101 categorías o se acumula en los
  platos visualmente más difíciles de distinguir?

La hipótesis de partida, formulada en el EDA antes de entrenar, era que la capacidad extra de
ViT-B/16 debería notarse sobre todo en las clases visualmente más confundibles, y que una brecha
plana a lo largo del ranking de dificultad favorecería a MobileViT-S. Cómo resultó, en
[Resultados](#resultados).

### Alcance

Todos los modelos se entrenan y evalúan **localmente, sobre el mismo hardware**.
Se miden dos familias de métricas: **desempeño** (top-1 accuracy y macro-F1) y **costo
arquitectónico** (parámetros, FLOPs, tamaño del modelo y latencia de inferencia). La adecuación a un
dispositivo con recursos limitados se **argumenta** a partir de esas mediciones.

## El dataset

**[Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)** (Bossard et al., ECCV 2014)


|               |                                |
| ------------- | ------------------------------ |
| Clases        | 101                            |
| Imágenes      | 101.000 (1.000 por clase)      |
| Split oficial | 750 train / 250 test por clase |
| Resolución    | lado máximo 512 px             |
| Tamaño        | ~5 GB comprimido               |


El split de **test fue curado manualmente**; el de **train contiene ruido** intencional (colores
intensos y algunas etiquetas incorrectas), según los propios autores. Usamos los splits oficiales
sin modificar: cualquier split propio haría los resultados incomparables con la literatura.

## Estado del proyecto

- [x] **EDA** (`[notebooks/1.0-eda-food101.ipynb](notebooks/1.0-eda-food101.ipynb)`)
- [x] **Preprocesamiento** (`[notebooks/2.0-preprocessing.ipynb](notebooks/2.0-preprocessing.ipynb)`)
- [x] **Fine-tuning y evaluación** de las cuatro arquitecturas (`[notebooks/3.1](notebooks/3.1-mobilevit.ipynb)` a `[3.4](notebooks/3.4-vit.ipynb)`)
- [x] **Variaciones de receta** en learning rate y tope de épocas (`[notebooks/3.5](notebooks/3.5-swin-30ep.ipynb)` a `[3.8](notebooks/3.8-mobilevit-lr5e-5-30ep.ipynb)`)
- [x] **Análisis integral y contraste de la hipótesis** (`[notebooks/4.0-comparativa.ipynb](notebooks/4.0-comparativa.ipynb)`)



## Resultados

Ocho corridas: las cuatro arquitecturas con una receta común (lr 5e-4, 20 épocas) y cuatro
variaciones de learning rate y tope de épocas. El análisis integral elige, por F1 de validación, la
receta de cada arquitectura (en **negrita** abajo) y compara esas cuatro sobre las 25.250 imágenes
del test. Los números de cada corrida salen de su `reports/results/<corrida>/metrics.json`, y los
contrastes entre corridas de `[reports/results/comparativa/hallazgos.json](reports/results/comparativa/hallazgos.json)`.


| Modelo          | Receta              | F1 macro (test) | Accuracy (test) | Parámetros | Latencia CPU int8 |
| --------------- | ------------------- | --------------- | --------------- | ---------- | ----------------- |
| **MobileViT-S** | **lr 5e-4, 20 ép.** | **0,861**       | **0,860**       | **5,0 M**  | **75 ms**         |
| MobileViT-S     | lr 5e-5, 20 ép.     | 0,821           | 0,821           | 5,0 M      | 101 ms            |
| MobileViT-S     | lr 5e-5, 30 ép.     | 0,833           | 0,833           | 5,0 M      | 102 ms            |
| **Swin-T**      | **lr 5e-4, 20 ép.** | **0,855**       | **0,855**       | **27,6 M** | **89 ms**         |
| Swin-T          | lr 5e-4, 30 ép.     | 0,855           | 0,855           | 27,6 M     | 98 ms             |
| **DeiT-Ti**     | **lr 5e-4, 20 ép.** | **0,777**       | **0,777**       | **5,5 M**  | **19 ms**         |
| ViT-B/16        | lr 5e-4, 20 ép.     | 0,823           | 0,824           | 85,9 M     | 201 ms            |
| **ViT-B/16**    | **lr 5e-5, 20 ép.** | **0,884**       | **0,884**       | **85,9 M** | **190 ms**        |


- La receta importa tanto como la arquitectura, y no afecta a todos igual. Bajar el learning rate de 5e-4 a 5e-5, con el resto de la receta fijo, sube la accuracy de test de ViT-B/16 de 0,824 a 0,884, pero baja la de MobileViT-S de 0,860 a 0,821; darle 30 épocas a Swin-T en lugar de 20 la deja en 0,855 (sin cambios). Por eso cada arquitectura se evalúa con su mejor receta (elegida por F1 de validación), y no con una receta común: fijarla habría perjudicado al modelo grande y favorecido a los chicos.
- **ViT-B/16 es el mejor, y MobileViT-S pierde poco.** Sobre el test completo,
MobileViT-S alcanza 0,860 de accuracy y ViT-B/16 0,884: una diferencia de 0,023 (IC 95 % de 0,019 a
0,027). Su F1 macro (0,861) equivale al 97 % del de ViT-B/16 (0,884), con 17 veces menos parámetros,
9 veces menos GFLOPs y una latencia en CPU int8 de 75 ms frente a 190 ms (el 39 %). Son los dos
únicos modelos en el frente de Pareto de los tres ejes de costo.
- **La brecha es pareja entre clases fáciles y difíciles.** La diferencia de accuracy entre
MobileViT-S y ViT-B/16 se mantiene casi constante en los tres terciles de dificultad del EDA: 0,912
contra 0,932 en el fácil, 0,874 contra 0,898 en el medio y 0,795 contra 0,821 en el difícil, sin
crecer con la dificultad (ρ de Spearman 0,16). La hipótesis de que la capacidad extra se notaría
sobre todo en las clases confundibles no se confirma. DeiT-Ti, en cambio, sí se despega más cuanto
más difícil es la clase: pasa de 0,857 contra 0,932 en el tercil fácil a 0,692 contra 0,821 en el
difícil.
- **Hay un núcleo de errores que no depende del modelo.** Steak ↔ filet mignon, chocolate cake ↔
chocolate mousse y los dos tartares son las confusiones dominantes en las cuatro arquitecturas; el
4,5 % del test lo fallan las cuatro.

Limitaciones: una semilla por corrida, barrido de recetas incompleto (DeiT-Ti no se probó con lr 5e-5,
Swin-T no probó otro learning rate) y latencias medidas en la CPU de una notebook, no en un teléfono.

## EDA

![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)

Utilizamos el EDA para responder preguntas que condicionan el diseño del benchmark.


| Sección                   | Pregunta                                                | Decisión que habilita                                                      |
| ------------------------- | ------------------------------------------------------- | -------------------------------------------------------------------------- |
| 1. Estructura y splits    | ¿Está balanceado?                                       | Elección de métrica (top-1 vs macro-F1)                                    |
| 2. Geometría              | ¿Cuánta imagen se pierde con el recorte de cada modelo? | Política de preprocesamiento por modelo, leída del checkpoint              |
| 3. Confusión entre clases | ¿Dónde está la dificultad real?                         | Hipótesis sobre en qué clases ViT-B/16 debería marcar diferencia           |
| 4. Subset de benchmark    | ¿Sobre qué imágenes exactas evaluamos?                  | Manifiesto reproducible y acotado en costo                                 |


Para ver la confusion entre clases usamos CLIP ViT-B/32 y necesita GPU: en Colab, *Entorno de ejecución, cambiar tipo de entorno a GPU (T4)*. 

CLIP es solo el instrumento de medición del EDA; no compite en el benchmark.

Si vas a correrlo más de una vez, poné `USE_DRIVE_CACHE = True`: el `.tar.gz` de 5 GB queda
cacheado en Drive y las sesiones siguientes no lo vuelven a descargar.

## Artefactos del EDA

El notebook escribe en `data/processed/`, y **estos tres archivos se commitean** porque son la entrada del notebook de benchmark:


| Archivo                          | Qué contiene                                                                                                                                                                                                  | Para qué se usa                                                                                                                                           |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `benchmark_subset.csv`           | Una fila por imagen de evaluación (2.525 = 25 por clase × 101): ruta relativa, clase, su `margen` de dificultad y el `tercil` (fácil / medio / difícil) al que pertenece.                                     | Fija la lista exacta de imágenes de test que evalúan **todos** los modelos. El `tercil` permite medir la brecha por dificultad sin volver a muestrear.    |
| `benchmark_subset_manifest.json` | Cómo se construyó ese CSV: semilla, imágenes por clase, conteos, las exclusiones (y la nota de que se detectaron sobre una muestra, no sobre el test completo) y el `sha256` del CSV.                         | Hace el subset reproducible y auditable: el `sha256` permite verificar meses después que el CSV no cambió, y deja asentado el alcance de las exclusiones. |
| `class_difficulty.csv`           | Ranking de las 101 clases por dificultad, derivado de embeddings CLIP: cohesión interna de la clase, su clase vecina más confundible y cuánto se le parece, el `margen`, la accuracy zero-shot y el `tercil`. | Funda la hipótesis del benchmark (dónde debería notarse la ventaja de ViT-B/16) y asigna el tercil de dificultad a cada clase e imagen del subset.       |




## Preprocesamiento

![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)

`[notebooks/2.0-preprocessing.ipynb](notebooks/2.0-preprocessing.ipynb)` ejecuta las decisiones de las decisiones de diseño que salieron del EDA con el paquete `vit_for_101_food_app.preprocessing` 

y muestra qué le pasa a una imagen antes de entrar a cada uno de los cuatro modelos. 

El modelo recibe `pixel_values` ya preparados; la receta (resize, recorte, normalización, orden de canales) es distinta por checkpoint. El notebook cubre el split de validación, el cache de imágenes, la ficha técnica de los cuatro modelos, los transforms que replican el `AutoImageProcessor` de cada uno, y cierra con el contrato que verifican
las corridas 3.x antes de entrenar. Corre con el dataset completo (~4.7 GB) en Colab.

### No escribimos la geometría a mano

Cada modelo consume su propio `AutoImageProcessor` de HuggingFace, leído en runtime por
`preprocessing.processors.spec_for`. 

`preprocessing.policies.build_transform` reconstruye esa misma geometría operando sobre tensores (necesario para poder aplicar augmentation con `torchvision.transforms.v2`).

En la corrida de Colab del notebook 2 la diferencia máxima medida fue `1.2e-07` para ViT y DeiT, `4.8e-07` para Swin y `0.0` para MobileViT (orden del épsilon de `float32`, no una aproximación visual).

Según sus processors, 

- ViT-B/16, Swin-T y DeiT-Ti hacen un **resize cuadrado directo** a 224×224 sin recorte y normalizan en RGB (ViT y DeiT con 0,5; Swin con las constantes de ImageNet). 
- MobileViT-S redimensiona el lado corto a 288, **recorta el centro** a 256, **no normaliza** y recibe los canales en **BGR**. Igualar esas filas "para que quede prolijo" sería un error: la sección 6 del notebook muestra que cada tensor es el que su checkpoint espera, y que forzarlos a coincidir haría entrenar al modelo sobre datos que nunca vio, sin que nada falle en ejecución.



### Artefactos versionados

Además de los tres artefactos del EDA, `data/processed/` versiona el split de validación y el mapa
de etiquetas, el mismo tipo de contrato: todos los modelos tienen que entrenar y validar sobre los
mismos índices.


| Archivo                         | Contenido                                                                      |
| ------------------------------- | ------------------------------------------------------------------------------ |
| `train_val_split.csv`           | Split de validación recortado de train (estratificado por clase, semilla fija) |
| `train_val_split_manifest.json` | Semilla, `val_fraction`, conteos por clase y `sha256` del CSV                  |
| `label_map.json`                | `id2label` / `label2id`, fijados por el orden de `meta/classes.txt`            |


El split de **test nunca se toca**: `preprocessing.splits.load_split("test")` lee directo de
`meta/test.txt`, no pasa por el CSV.

### `make preprocess`

```bash
make preprocess   # = make data split cache verify
```


| Target        | Comando               | Qué hace                                                                                                                      |
| ------------- | --------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `make data`   | `dataset.py download` | Descarga y extrae Food-101 (~4.7 GB, idempotente)                                                                             |
| `make split`  | `dataset.py split`    | Escribe `train_val_split.csv`, su manifiesto y `label_map.json`                                                               |
| `make cache`  | `dataset.py cache`    | Reescala todo el dataset al lado corto configurado (`CACHE_SHORT_SIDE = 288`) en `data/interim/`, regenerable y no versionado |
| `make verify` | `features.py verify`  | Compara los transforms `eval` contra el `AutoImageProcessor` real de cada modelo                                              |


`features.py preview` (fuera de `make preprocess`) escribe una grilla antes/después por modelo en
`reports/figures/`, para inspección visual.

## Entrenamiento y benchmark

Todo el código de modelo y entrenamiento vive en `vit_for_101_food_app.modeling` y es
**genérico sobre** `model_key` (una clave del registry `config.MODELS`): sumar una
arquitectura al benchmark es agregar una línea al registry, no escribir un pipeline nuevo.


| Módulo          | Responsabilidad                                                                                                                                                                                                                                                                        |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `training.py`   | Construcción del modelo, `TrainingRecipe` (misma estructura para todos; el learning rate y el tope de épocas se variaron a propósito en los notebooks 3.5 a 3.8), `Trainer` de HuggingFace y `resolve_batch_plan` (batch/acumulación/checkpointing según resolución y GPU)             |
| `evaluation.py` | Predicciones por imagen, métricas por tercil, reporte por clase, confusiones, FLOPs, tamaño de los pesos (fp32/fp16/int8) y latencia en GPU/CPU (incluida la medición del lado del dispositivo con el modelo **cuantizado a int8** en CPU), **el mismo código para todos los modelos** |
| `benchmark.py`  | Orquestación: verifica artefactos, entrena, evalúa y escribe `metrics.json` + CSV                                                                                                                                                                                                      |


Hay **dos formas de correr el mismo pipeline**, que miden exactamente lo mismo:

- **Notebooks** (`notebooks/3.1-mobilevit.ipynb`, `3.2-swin.ipynb`, `3.3-deit.ipynb`,
`3.4-vit.ipynb`), celda por celda, con salidas visibles para el informe. Los cuatro son
idénticos salvo `MODEL_KEY`; están pensados para Colab (secciones 0.2 y 0.3 arman el
entorno y los datos). Los notebooks `3.5` a `3.8` repiten tres de esas corridas variando
la receta (lr 5e-5 para MobileViT y ViT, tope de 30 épocas para Swin y MobileViT) y
escriben en `reports/results/<model>-<variante>/`.
- **CLI headless**, para una corrida desatendida (Kaggle, VM):
  ```bash
  python -m vit_for_101_food_app.modeling.train --model swin      # benchmark completo
  python -m vit_for_101_food_app.modeling.train --model mobilevit --resume
  python -m vit_for_101_food_app.modeling.predict --model swin --model-dir models/swin/full/best
  ```

Cada corrida escribe checkpoints en `models/<model>/full/` y resultados livianos
(`metrics.json`, predicciones, tabla por tercil) en `reports/results/<model>/`, con el
mismo esquema para todos los modelos: la tabla comparativa del benchmark se arma juntando
los `metrics.json` sin reentrenar.

Cada corrida guarda también sus curvas de entrenamiento (`curvas-<model>-*.png`) y logs de
TensorBoard. Para comparar los modelos en MLflow, a partir de `reports/results/`:

```bash
python -m vit_for_101_food_app.tracking
mlflow ui --backend-store-uri sqlite:///mlflow.db
```



### Análisis integral

![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)

`[notebooks/4.0-comparativa.ipynb](notebooks/4.0-comparativa.ipynb)` junta las ocho corridas
de `reports/results/` 

Primero mide el efecto de cada cambio de receta dentro de cada arquitectura (contrastes pareados sobre las mismas 25.250 imágenes de test) y elige, por F1 de validación, la corrida que representa a cada una. 

Con esas cuatro compara desempeño frente a costo, brecha por tercil de dificultad del EDA y por clase, confusiones compartidas y ejemplos del dataset.

Escribe tablas en `reports/results/comparativa/*.csv`, los números clave en
`reports/results/comparativa/hallazgos.json` y figuras en `reports/figures/comparativa-*`.


La lógica vive en `modeling/comparison.py` y `modeling/examples.py`, con tests. Corre con las
dependencias base (sin GPU ni el extra `deep`); solo la sección de ejemplos necesita el
`food-101.tar.gz` en `data/raw/`, del que extrae las ~60 imágenes que muestra.

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

MLflow va en el extra `tracking` (`uv sync --extra deep --extra tracking`).

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
├── models             <- Checkpoints de las corridas (no versionados)
│
├── notebooks          <- Notebooks. Convención: número de orden + descripción,
│                         p. ej. `1.0-eda-food101.ipynb`, `2.0-preprocessing.ipynb`
│
├── pyproject.toml     <- Metadatos, dependencias y config de ruff
│
├── references         <- Diccionarios de datos, manuales, material explicativo
│
├── reports
│   ├── figures        <- Gráficos para los informes
│   └── results        <- Por corrida: metrics.json, predicciones, curvas; y comparativa/ con el análisis integral
│
├── tests              <- Tests del preprocesamiento, la evaluación, la comparación y el tracking
│
└── vit_for_101_food_app   <- Código fuente del proyecto
    ├── config.py               <- Rutas, semilla y resoluciones de los modelos
    ├── dataset.py              <- CLI: descarga, split de validación y cache
    ├── features.py             <- CLI: verificación e inspección visual de los transforms
    ├── modeling
    │   ├── training.py         <- Modelo, receta y Trainer, genéricos sobre model_key
    │   ├── evaluation.py       <- Protocolo de evaluación común (predicciones, terciles, FLOPs, latencia)
    │   ├── benchmark.py        <- Orquestación: une preprocessing + training + evaluation
    │   ├── comparison.py       <- Análisis integral: tablas, contrastes pareados y figuras del notebook 4
    │   ├── examples.py         <- Imágenes del dataset para ilustrar los resultados
    │   ├── train.py            <- CLI headless: corre el benchmark completo de un modelo
    │   └── predict.py          <- CLI de inferencia sobre un modelo ya entrenado
    ├── plots.py                <- Curvas de entrenamiento y estilo de las figuras
    ├── tracking.py             <- Registro de las corridas en MLflow a partir de reports/results
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

El código de este repositorio está bajo licencia MIT (ver `[LICENSE](LICENSE)`). El uso del
**dataset** está sujeto al `license_agreement.txt` incluido en la descarga (uso académico / no
comercial). Este repositorio **no distribuye las imágenes**, solo el código que las analiza.