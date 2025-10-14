# Proyecto EEG MI — Primer Avance

Este repositorio contiene el trabajo inicial de análisis y preprocesamiento de señales EEG de **imaginación motora (MI)**, así como la implementación del primer modelo base **FBCSP + LDA**.  

A continuación se describen las fases clave desarrolladas hasta el momento.

---

## 1. EDA de datos RAW

Antes de cualquier limpieza se realizó un **análisis exploratorio de los datos crudos** (`data/raw`) con los siguientes objetivos:

- **Inventario de archivos:** verificación de sujetos disponibles y runs por sujeto.
- **Conteo de eventos por clase:** Left, Right, Both Fists, Both Feet, Rest.
- **Amplitud extrema:** cálculo de percentil 99 (`p99_uV`) y desviación estándar por canal (`std_uV`) para detectar outliers.
- **Artefactos musculares (EMG):** estimados mediante la **relación de potencia 20–40 Hz** en los canales motores.
- **PSD (densidad espectral de potencia):** inspección en C3, Cz, C4.
- **Separabilidad inicial:** análisis con **t-SNE** sobre log-varianza de las épocas.

**Hallazgos importantes en RAW:**
- Gran variabilidad inter-sujetos en amplitud (50–300 μV).
- Presencia de ruido y artefactos musculares en varios sujetos.
- La mayoría de sujetos muestran **silhouette score negativo** → baja separabilidad entre clases en el estado crudo.
- Dataset heterogéneo, requiere un pipeline de preprocesamiento robusto.

---

## 2. Pipeline de Preprocesamiento

El preprocesamiento busca limpiar los EEG y asegurar que las features extraídas representen la actividad neuronal y no artefactos.


### Explicación paso a paso

1. **Normalización de nombres y montaje**  
   - Estandarización de canales y asignación al sistema 10–20.  
   - Permite localizar fácilmente C3, Cz y C4, fundamentales en MI.

2. **Filtro Notch**  
   - Remueve interferencia eléctrica de 50/60 Hz.  
   - Evita que el ruido de la red contamine las bandas mu y beta.

3. **Clipping de amplitud**  
   - **Softclip:** atenúa valores extremos sin descartarlos.  
   - **Hardclip:** elimina épocas con amplitudes fuera de rango.  
   - Previene que outliers extremos dominen el entrenamiento.

4. **Filtro Bandpass (8–30 Hz)**  
   - Aísla las bandas mu (8–12 Hz) y beta (13–30 Hz).  
   - Son las más asociadas a imaginación motora.

5. **ICA (FastICA / Picard)**  
   - Separa fuentes independientes y elimina componentes de artefactos:  
     - Oculares (EOG),  
     - Musculares (EMG),  
     - Cardiacos (ECG).  
   - Esencial para mejorar la pureza de las señales.

6. **Segmentación en epochs**  
   - Ventana de **0.5–4.5 s post-estímulo**.  
   - Captura la actividad cortical durante la tarea, evitando ruido al inicio/final.

7. **Rechazo automático de épocas**  
   - Se eliminan ensayos con amplitudes excesivas (peak-to-peak).  
   - Garantiza un set final más balanceado y limpio.

8. **Exportación en formato FIF**  
   - Archivos por sujeto (`Sxxx_MI-epo.fif`).  
   - Estandariza y permite reutilizar fácilmente con MNE y modelos posteriores.

**Resultados del preprocesamiento:**
- Reducción significativa de amplitudes extremas.  
- Menor presencia de artefactos EMG.  
- Balance de clases más uniforme por sujeto.  
- Mejora en separabilidad en varios sujetos (Δ silhouette > 0).  

---

## 3. EDA de datos POST

Una vez aplicado el pipeline, se evaluaron nuevamente los datos procesados (`data/processed`) con métricas y visualizaciones:

- **Conteo de épocas por sujeto y clase** → para verificar balance.
- **PSD en C3, Cz, C4 por clase** → confirmación de la actividad en bandas mu/beta.
- **Topomapas de potencia** en bandas mu y beta → patrones espaciales de activación.
- **QA automático:** detección de sujetos problemáticos (muy pocas épocas, EMG residual alto, silhouette muy negativo).
- **Comparación RAW vs POST**:  
  - Δ silhouette → mejora en separabilidad de clases.  
  - Δ amplitud extrema (`p99_uV`) → reducción de outliers.  
  - EMG ratio → caída en la mayoría de sujetos.

**Hallazgos importantes en POST:**
- Disminución clara en amplitudes extremas y ruido.  
- Aumento en la calidad de épocas disponibles.  
- Mejor diferenciación de clases en varios sujetos.  
- Sin embargo, algunos sujetos siguen presentando problemas de EMG o baja separabilidad, que deberán manejarse con **flags de QA** en futuras iteraciones.

---

## Conclusión preliminar

El **pipeline de preprocesamiento** aplicado transforma un dataset crudo, heterogéneo y ruidoso, en un conjunto más limpio y balanceado. Esto sienta las bases para los experimentos con modelos como **FBCSP + LDA** y, en fases posteriores, comparaciones con **SVM, Riemannianos y redes profundas**.

--- 

## 4. FBCSP + LDA

**Objetivo:** clasificar EEG de *Motor Imagery* (μ/β) usando **Filter-Bank CSP (FBCSP)** + **LDA con shrinkage**.  
Se evalúa en dos configuraciones: **INTRA-subject** (K-Fold por ensayos) y **INTER-subject** (folds predefinidos por sujetos desde JSON, con **validación interna por sujetos** y opción de **calibración**).

### Pipeline
1. **Carga y recorte** de épocas MNE (`.fif`) con ventana fija `crop_window` (p. ej., 0.5–3.5 s).
2. **Selección de canales motores** (opcional): `C3, CZ, C4, FC3, FC4, CP3, CPZ, CP4`; alineación de canales en VAL/TEST con `reorder_channels`.
3. **Banco de filtros** (μ/β): subbandas densas de 2 Hz entre 8–30 Hz (11 bandas).
4. **CSP por subbanda**: `n_csp` componentes, `reg='ledoit_wolf'`, `log=True` (devuelve vectores de varianzas proyectadas).  
   - **Fit** con **solo TRAIN** del fold; **transform** en VAL/TEST.
5. **Concatenación de features** de todas las subbandas.
6. **Estandarización** (z-score de features) con `StandardScaler` (fit en TRAIN).
7. **Clasificador**: `LDA(solver='lsqr', shrinkage='auto')`.

### FBCSP + LDA (features)
Para cada sub-banda (8–30 Hz, pasos de 2 Hz):
1) Filtramos; 
2) Ajustamos **CSP** en TRAIN y transformamos VAL/TEST;  
3) Calculamos **log-varianzas** de las `n_csp` componentes;  
4) **Concatenamos** features de todas las sub-bandas.  
Luego estandarizamos y clasificamos con **LDA (shrinkage)**.

### Evaluaciones
- **INTRA-subject** (`run_intra_all`):
  - `StratifiedKFold(k)` por sujeto (split por ensayos).
  - Métricas: Accuracy y F1-macro por fold; promedio ± DE por sujeto; fila **GLOBAL**.
  - Artefactos: CSV, TXT, figuras de matrices de confusión (mosaicos).

- **INTER-subject** (`run_inter_subject_cv_from_json`):
  - Folds desde JSON (`train/test` por sujetos).
  - **Validación interna por sujetos**: fracción `val_ratio_subjects` dentro de TRAIN para ajuste/selección.
  - Calibración per-subject (k-shots): Para cada sujeto de test, tomamos **k=5** épocas por clase como **calibración** y evaluamos en el resto de sus épocas. Durante la calibración se re-ajustan **FBCSP** (fit con TRAIN + k-shots del propio sujeto) y **LDA** (tras estandarización). Este esquema refleja el uso real de BCI: una **breve sesión inicial** de calibración por usuario mejora sustancialmente la transferencia inter-sujeto.
  - Métricas: VAL (acc, F1-macro) y TEST (acc, F1-macro) por fold + **GLOBAL**.
  - Artefactos: CSV consolidado, TXT de métricas, TXT con `classification_report` por fold, figuras de confusión por fold y **GLOBAL**.

### Antileakage y reproducibilidad
- CSP, scaler y LDA se ajustan **exclusivamente** con TRAIN del fold (o TRAIN+CALIB si la calibración está activa).  
- VAL/TEST sólo se **transforman**.  
- Canales de VAL/TEST se **reordenan** para coincidir con TRAIN.  
- Se generan logs con timestamp y parámetros para auditoría.

### Principales hiperparámetros
- `crop_window=(0.5, 3.5)` s  
- `motor_only=True | False`  
- `zscore_epoch=True | False` (z-score por época previo a CSP)  
- `fb_bands`: denso (2 Hz) de 8–30 Hz  
- `n_csp`: típicamente 4–8 (p. ej., 4 ó 6)  
- `val_ratio_subjects≈0.16`, `calibrate_n` (sólo INTER)

### Salidas
- **Tablas** (`/models/fbcsp_lda/tables`): CSV de métricas con fila GLOBAL.
- **Logs** (`/models/fbcsp_lda/logs`): TXT de métricas y `classification_reports_by_fold_*.txt`.
- **Figuras** (`/models/fbcsp_lda/figures`): matrices de confusión por sujeto/fold y GLOBAL.

> **Nota:** El mapeo de clases usa `LabelEncoder` para asegurar etiquetas consistentes. No influye en la señal ni en parámetros del modelo.

## 5. Modelo Riemanniano para MI-EEG (MDM / FgMDM)

**Resumen.** Cada época se representa por su **matriz de covarianza SPD** (SPD significa Symmetric Positive Definite) por sub-banda (8–30 Hz), y se clasifica por **distancia geodésica** a las medias de clase en la geometría Riemanniana (pyRiemann). Usamos **OAS** como estimador de covarianza y **normalización por traza** para estabilizar escala. Dos variantes:
- **MDM**: Minimum Distance to Mean sobre un **bloque-diagonal** que apila las covariancias de todas las bandas.
- **FgMDM**: Filter-geodesic MDM, que **agrega en el manifold** la información multi-banda.

**Preprocesado.**
- Ventana temporal: `crop_window=(0.5, 3.5)` (configurable).
- Canales: `motor_only=True` (C3, Cz, C4, FC3/4, CP3/z/4).
- Banco de bandas: denso 8–30 Hz, paso 2 Hz.
- Covarianza: `Covariances(estimator='oas')` + normalización por traza.

**Validaciones.**
- **INTRA-sujeto (k-fold)**: 5 folds estratificados dentro de cada sujeto; métricas por sujeto + GLOBAL.
- **INTER-sujeto (folds JSON)**: split de **validación por sujetos** dentro de TRAIN; ajuste del espacio **solo con TRAIN**; métricas en VALID y TEST; **matriz de confusión global** y `classification_report` por fold.

**Calibración per-subject (k-shots, recomendada).**
Para cada sujeto de TEST, tomamos **k=5** épocas por clase como **calibración**, recomputamos el espacio con `TRAIN + CALIB_del_sujeto` y **evaluamos en el resto** de sus épocas. Refleja el uso real de BCI con una **breve sesión inicial** de calibración por usuario. (Alternativamente, se puede calibrar con **n sujetos completos** del TEST si el escenario lo requiere.)

**Features (Riemann).** Cada época se representa por **matrices de covarianza SPD** por sub-banda (estimador OAS + normalización por traza).  
**Geometría.** Las SPD viven en un manifold; usamos **distancia geodésica Riemanniana** (afin-invariante) para comparar.  
**Clasificadores.**  
- **MDM**: calcula la **media Riemanniana** por clase y predice por **distancia al centroide**. Multi-banda vía **bloque diagonal**.  
- **FgMDM**: mantiene **una SPD por banda** y **agrega geodésicamente** la info multi-banda; suele rendir mejor.  
**En este repo:** `model='fgmdm'` (por defecto en inter-sujeto) ⇒ el clasificador activo es **FgMDM**.


## 6. EEGNet + Fine-Tuning Progresivo por Sujeto (MI-EEG, 4 clases)

**Objetivo:**  
Clasificar imaginación motora (MI) en 4 clases (`Left`, `Right`, `Both Fists`, `Both Feet`) usando solo **8 canales motores de EEG**, mediante una red **EEGNet** y un protocolo realista de **calibración/fine-tuning progresivo por sujeto**.

---

### ¿Qué problema resolvemos?

Cuando intentamos usar un **BCI (interfaz cerebro-computador)** con personas nuevas, el modelo suele bajar rendimiento porque cada cerebro es distinto (morfología, impedancias, atención, fatiga, etc.).

Nuestra estrategia:
1. **Aprender un modelo global** con muchos sujetos (para captar patrones generales).  
2. **Personalizarlo ligeramente** con pocos ensayos etiquetados del nuevo usuario (**fine-tuning progresivo**), de forma rápida y estable.

---

### ¿Qué es EEG y qué entrada ve la red?

- **EEG:** voltajes medidos en el cuero cabelludo.  
- **Canales usados (motores):** `C3, Cz, C4, FC3, FC4, CP3, CPz, CP4`.

**Procesamiento de entrada:**
- Para cada evento (T1/T2), extraemos una **ventana de 3 segundos**, muestreada a **160 Hz**.  
- Normalizamos cada época (z-score por canal) para estabilizar amplitudes.  

**Cada ejemplo:**  
Matriz de tamaño `(Tiempo × Canales)`  
**Etiquetas:**  
`0=Left`, `1=Right`, `2=Both Fists`, `3=Both Feet`.

---

### ¿Qué es una red neuronal y cómo decide una clase?

Una red neuronal transforma la entrada a través de capas hasta llegar a una **representación útil para clasificar**.  
La última capa produce **4 números** (uno por clase).  
Aplicamos **softmax** → obtenemos probabilidades.  

Entrenamos la red para maximizar la probabilidad de la clase correcta con:  
- **Función de pérdida:** *Cross-Entropy*  
- **Optimizador:** *Adam*

---

### Arquitectura: EEGNet

EEGNet está diseñada específicamente para EEG.  
La entrada tiene forma `(Batch, 1, Tiempo, Canales)` y pasa por **3 bloques convolucionales + cabeza densa**:

Entrada (B, 1, T, C)
│
├─ Bloque 1: Convolución Temporal → aprende filtros en el tiempo
│ + BatchNorm + ELU
│
├─ Bloque 2: Convolución Depthwise Espacial → patrones espaciales
│ + BN + ELU + AvgPool + Dropout
│
├─ Bloque 3: Convolución Separable Temporal → refina patrones
│ + BN + ELU + AvgPool + Dropout
│
└─ Cabeza: Flatten → Dense(80) → Dense(4) + Softmax


**Intuición:**
- **Temporal:** aprende ritmos relevantes (μ/beta).  
- **Espacial (depthwise):** combina canales como una CSP aprendida.  
- **Separable temporal:** refina patrones con pocos parámetros.  
- **Cabeza:** traduce características a probabilidades por clase.

---

### Evaluación: cómo evitamos *data leakage*

Validación **inter-sujeto (K=5 folds):**
- Cada *fold* tiene sujetos **no vistos** en test.  
- Dentro del *train*, se reserva un **15% de sujetos para validación** (early stopping).  
- Así, el modelo no ve nunca el test antes de tiempo.

---

### Entrenamiento Global (Inter-Sujeto)

**Dataset:**  
Hasta **21 ensayos por clase y sujeto** (con reposición si faltan) → dataset balanceado.

**Entrenamiento:**
- Modelo: EEGNet  
- Optimizador: Adam  
- Épocas: 100 (máximo)  
- *Early stopping* por `val_acc`

**Evaluación:**
- `accuracy` y `classification report` en sujetos no vistos.

---

### Fine-Tuning Progresivo por Sujeto (Calibración rápida)

Cuando llega un **nuevo sujeto (de test)**:

1. Hacemos **4-fold CV interno** con solo sus datos (≈ 75% calibración / 25% hold-out).  
2. Entrenamos **tres modos** con *early stopping* + penalización **L2-SP**:

| Modo | Capas entrenadas | Descripción |
|------|------------------|-------------|
| `out` | Solo salida | Ajusta el clasificador final |
| `head` | FC + salida | Personaliza la cabeza entera |
| `spatial+head` | Convs espaciales + separables + cabeza | Congela filtros temporales globales |

Elegimos el modo con mejor `accuracy` en el *hold-out* del sujeto → se usa para predecir todo su set.

**Por qué funciona:**  
Con pocos ensayos del usuario, ajustar pocas capas + L2-SP (penaliza alejarse del modelo global) evita sobreajuste y personaliza la fisiología.

---

### Hiperparámetros

#### Generales
| Parámetro | Valor | Descripción |
|------------|--------|-------------|
| `FS` | 160 Hz | Frecuencia de muestreo |
| `WINDOW_MODE` | '3s' | Duración de ventana |
| `EXPECTED_8` | 8 canales | C3, Cz, C4, FC3, FC4, CP3, CPz, CP4 |

#### Entrenamiento Global
| Parámetro | Valor | Descripción |
|------------|--------|-------------|
| `N_FOLDS` | 5 | CV por sujeto |
| `BATCH_SIZE` | 16 | Tamaño de lote |
| `EPOCHS_GLOBAL` | 100 | Máximo de épocas |
| `LR` | 1e-3 | Tasa de aprendizaje |
| `GLOBAL_VAL_SPLIT` | 0.15 | Validación por sujeto |
| `GLOBAL_PATIENCE` | 10 | Early stopping |
| `LOG_EVERY` | 5 | Log cada 5 épocas |

#### Fine-Tuning
| Parámetro | Valor | Descripción |
|------------|--------|-------------|
| `CALIB_CV_FOLDS` | 4 | CV interno del sujeto |
| `FT_EPOCHS` | 30 | Máx. épocas por etapa |
| `FT_BASE_LR` | 5e-5 | LR para capas base |
| `FT_HEAD_LR` | 1e-3 | LR para cabeza |
| `FT_L2SP` | 1e-4 | Penalización L2-SP |
| `FT_PATIENCE` | 5 | Early stopping FT |
| `FT_VAL_RATIO` | 0.2 | Validación interna del sujeto |

**Regla práctica:**  
- Si tienes **pocos ensayos**, usa `out` o `head`.  
- Si tienes **más datos**, prueba `spatial+head`.  
- Si ves sobreajuste → sube `FT_L2SP` o baja `LR`.

---

### Métricas y Salidas

- **Global accuracy por fold** (inter-sujeto puro).  
- **Fine-tuning accuracy** y **Δ(FT - Global)** (mejora por personalización).  
- **Classification reports:** precisión, recall, F1 por clase.  
- **Matriz de confusión global** acumulada (todos los folds).

---

## Evaluación INTRA-Sujeto (Fine-Tuning Progresivo dentro del mismo sujeto)

En el modo **INTRA**, evaluamos el rendimiento **dentro del mismo sujeto**.  
Para cada persona, realizamos una **validación cruzada k-fold** con sus propias épocas.

En cada fold:
- Partimos de un **modelo global** (pre-entrenado con todos los sujetos).
- Lo **ajustamos ligeramente** a ese sujeto mediante **fine-tuning progresivo**.
- Elegimos la etapa (`out`, `head` o `spatial+head`) que mejor rinde en validación.
- Probamos esa etapa en el **test del fold**.

---

### Flujo de INTRA (paso a paso)

#### Pre-entrenamiento global (pretrain)

- Mezclamos **todas las épocas de todos los sujetos** (sin limitar a 21 por clase).  
- Entrenamos un modelo **EEGNet global** durante hasta **100 épocas** (por defecto).  
- Este modelo sirve como **base inicial** para todos los fine-tuning posteriores.

---

#### k-Fold por sujeto

Para cada sujeto `Sxyz`:

1. **División interna (k-fold estratificado):**  
   - Dividimos solo las épocas de ese sujeto en `k` folds (por ejemplo, `k=5`),  
     manteniendo las clases balanceadas en cada fold.

2. **En cada fold:**

   **a) Calibración / Validación**  
   - Dentro del conjunto de *train* del fold, hacemos un split adicional (p. ej. 20% para validación).

   **b) Fine-Tuning Progresivo**  
   - Clonamos el **modelo global pre-entrenado**.  
   - Entrenamos tres configuraciones diferentes:

     | Modo | Capas entrenadas | Descripción |
     |------|------------------|-------------|
     | `out` | Solo la capa de salida | Rápido y seguro con pocos datos |
     | `head` | FC + salida | Más capacidad de adaptación |
     | `spatial+head` | Convoluciones espaciales + separables + cabeza | Congela los filtros temporales globales |

   - En todos los modos se aplica **L2-SP**, una regularización que penaliza desviarse del modelo global → reduce el sobreajuste.

   **c) Selección de modelo**
   - Elegimos la etapa con **mayor `val_acc`** (accuracy en validación dentro del fold).

   **d) Prueba**
   - Evaluamos esa etapa elegida en el **test del fold** (épocas del mismo sujeto, nunca vistas en ese fold).

---

#### Agregación de resultados

- **Por sujeto:**  
  Promediamos `accuracy` y `F1-macro` sobre sus `k` folds.

- **Global INTRA:**  
  Promedio de las métricas de todos los sujetos → mide el rendimiento medio de personalización dentro del sujeto.

---

> 🧠 **Resumen:**  
> El protocolo INTRA evalúa la capacidad del modelo global de adaptarse a cada sujeto con pocos datos.  
> Combina transferencia de aprendizaje, regularización L2-SP y validación cruzada interna para medir un rendimiento realista de calibración personalizada.


> 🔍 **Resumen:**  
> EEGNet + Fine-Tuning Progresivo permite transferir un modelo global a nuevos sujetos con mínima calibración, manteniendo estabilidad y mejorando la personalización en escenarios de BCI realistas.
