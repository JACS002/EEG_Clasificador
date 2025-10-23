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


---

# 🧠 EEGNet (4 clases, 8 canales) con Augmentaciones, TTA y Fine-Tuning Progresivo

Este proyecto implementa un clasificador EEGNet en PyTorch para señales EEG de imaginación motora (MI-EEG), con:
- Entrenamiento global (5 folds) sobre datos sin balancear por sujeto/clase.
- **Augmentaciones (jitter, ruido, channel-drop)** inspiradas en CNN+Transformer.
- **SGDR (CosineAnnealingWarmRestarts)** para ajustar la tasa de aprendizaje.
- **Label smoothing**, **pesos por clase** y **max-norm** como regularizaciones.
- **Test-Time Augmentation (TTA)** por desplazamientos temporales.
- **Fine-Tuning progresivo** por sujeto con penalización L2SP.

---

## 📦 Estructura general

| Bloque | Descripción |
|:--|:--|
| **Carga de datos** | Lectura de archivos EDF, extracción de epochs `[-1,5]s`, selección de 8 canales, z-score por época. |
| **Normalización** | Estandarización por canal usando estadísticas del conjunto de entrenamiento. |
| **Modelo** | EEGNet con filtros temporales y espaciales separables, salida lineal para 4 clases. |
| **Entrenamiento global** | Augments + SGDR + label smoothing + pesos por clase + max-norm. |
| **Evaluación (Test)** | TTA por *time-shifts* y métricas (ACC, F1, matriz de confusión). |
| **Fine-tuning progresivo** | Adaptación por sujeto con L2SP y congelamiento progresivo de capas. |

---

## ⚙️ Configuración global (hiperparámetros)

| Categoría | Parámetro | Valor | Explicación |
|:--|:--|:--|:--|
| Datos | FS | 160 Hz | Frecuencia de muestreo objetivo |
| | TMIN/TMAX | -1.0 / 5.0 s | Ventana temporal de cada ensayo |
| | EXPECTED_8 | C3,C4,Cz,CP3,CP4,FC3,FC4,FCz | Canales usados |
| | NORM_EPOCH_ZSCORE | True | Z-score por época y canal |
| Split | VAL_SUBJECT_FRAC | 0.18 | % de sujetos usados como validación |
| | VAL_STRAT_SUBJECT | True | Validación estratificada por etiqueta dominante |
| Train | EPOCHS_GLOBAL | 100 | Épocas máximas |
| | BATCH_SIZE | 64 | Tamaño de batch |
| | LR_INIT | 1e-2 | Tasa inicial de aprendizaje |
| | SGDR_T0 / SGDR_Tmult | 6 / 2 | Ciclos coseno: 6, 12, 24… |
| | GLOBAL_PATIENCE | 10 | Early stopping |
| EEGNet | F1 / D | 24 / 2 | Filtros temporales y multiplicador depthwise |
| | kernel_t / k_sep | 64 / 16 | Kernels temporal y separable |
| | pool1_t / pool2_t | 4 / 6 | Reducción temporal por bloque |
| | drop1_p / drop2_p | 0.35 / 0.6 | Dropout |
| | chdrop_p | 0.10 | Channel dropout |
| Loss | label_smoothing | 0.05 | Suavizado de etiquetas |
| | boost (clase 2/3) | 1.25 / 1.05 | Pesos extra para clases raras |
| Augments | p_jitter / p_noise / p_chdrop | 0.35 / 0.35 / 0.15 | Probabilidad de aplicar cada tipo |
| | max_jitter_frac / noise_std | 0.03 / 0.03 | Magnitud del jitter y ruido |
| TTA | shifts_s | ±0.075,…,0 s | Desplazamientos en inferencia |
| FT | CALIB_CV_FOLDS | 4 | Folds internos por sujeto |
| | FT_EPOCHS | 30 | Épocas por modo |
| | FT_HEAD_LR / FT_BASE_LR | 1e-3 / 5e-5 | LR para cabeza y base |
| | FT_L2SP | 1e-4 | Penalización de alejamiento de pesos globales |
| | FT_PATIENCE / FT_VAL_RATIO | 5 / 0.2 | Early stopping y validación interna |

---

## 🧠 Arquitectura EEGNet

### Estructura general
Entrada: `(B, 1, T, C)` → Salida: `(B, 4)`  
(`B`: batch, `T`: tiempo ≈ 960, `C`: canales=8)

1. **ChannelDropout (p=0.10)**  
   Apaga canales completos aleatoriamente (simula fallos de electrodos).

2. **Bloque temporal**
   ```python
   Conv2d(1 → F1, kernel=(64,1)) → BN → ELU
   ```
   Extrae patrones de oscilación y filtros pasa-banda temporales.

3. **Bloque espacial (depthwise)**
   ```python
   Conv2d(F1 → F1*D, kernel=(1, n_ch), groups=F1)
   AvgPool2d(4,1) → Dropout(0.35)
   ```
   Aprende combinaciones espaciales (proyecciones de canales) por cada filtro temporal.

4. **Bloque separable temporal**

   ```python
   Conv2d(F2 → F2, kernel=(16,1), groups=F2)
   Conv2d(F2 → F2, kernel=(1,1))
   AvgPool2d(6,1) → Dropout(0.6)
   ```
   Refina dinámicas temporales específicas con pocos parámetros.

5. **Head**

   ```python
   Flatten → Linear(1920→128) → ELU → Linear(128→4)
   ```
   Proyección final a clases.

## 🧩 Entrenamiento global

- **Optimizador:** Adam (`LR inicial = 1e-2`)
- **Scheduler:** CosineAnnealingWarmRestarts (SGDR)  
  - Ciclos: 6, 12, 24, 48 épocas…  
  - Cada reinicio restablece el LR alto para explorar nuevos mínimos.
- **Augmentaciones:** jitter temporal, ruido gaussiano, channel-drop
- **Pérdida:** Weighted Soft CrossEntropy con *label smoothing* (0.05)
- **Regularizaciones:**
  - Max-norm = 2.0 (filtros espaciales y FC)
  - Dropout + ChannelDropout
  - Label smoothing + pesos por clase

## 🔄 CosineAnnealingWarmRestarts (SGDR)

Controla la **tasa de aprendizaje (LR)** de forma cíclica, alternando fases de exploración y refinamiento.

### 🧠 Intuición
- **Grandes saltos** → explora nuevos mínimos.  
- **LR pequeño** → refina soluciones locales.  
- **Reinicios** → permite escapar de mínimos subóptimos.

### ⚙️ Parámetros
- **T₀ = 6** → duración del primer ciclo (épocas).  
- **T_mult = 2** → cada ciclo siguiente dura el doble: 6, 12, 24, 48, …

El LR se reinicia al valor inicial en cada ciclo, generando una curva coseno decreciente dentro de cada fase.

### 💡 Consejos prácticos
| Situación | Ajuste recomendado |
|:--|:--|
| `val_acc` oscila mucho | Aumenta **T₀** o reduce **LR_INIT** |
| Entrenamiento estancado | Sube **LR_INIT** o acorta **T₀** |

✨ **Idea clave:** equilibrar exploración (LR alto) y refinamiento (LR bajo) para alcanzar mejores mínimos sin sobreentrenar.

## 🌈 Augmentaciones (entrenamiento)

Tres transformaciones principales se aplican por batch (B, 1, T, C):

### 1️⃣ Jitter temporal
Desplaza la señal unos milisegundos (roll en el eje temporal).  
**max_jitter_frac = 0.03**  → ±180 ms para 6 s (T≈960)  

**Motivo:** simula pequeñas desincronizaciones del inicio del ensayo (onset).  
**Rango típico:** 0.02–0.05 (±120–300 ms).  

---

### 2️⃣ Ruido gaussiano
**noise_std = 0.03**  

**Motivo:** mejora la robustez frente a ruido fisiológico y electrónico.  
**Rango recomendado:** σ = 0.01–0.05 para señales EEG normalizadas (z-scoreadas).  

---

### 3️⃣ Channel-drop
**p_chdrop = 0.15**, **max_chdrop = 1**  

**Motivo:** incrementa la robustez espacial ante la pérdida o mal contacto de electrodos.  
**Recomendación:** apagar 1–2 canales cuando se trabaja con 8 canales totales.  

## 🧱 Regularizaciones

### 🧩 Label Smoothing (ε=0.05)
Suaviza las etiquetas para evitar sobreconfianza:

**Fórmula:**
\[
\tilde{y} = (1 - \varepsilon) \cdot \text{one\_hot} + \frac{\varepsilon}{K}
\]

Esto reduce la probabilidad de que el modelo se vuelva demasiado confiado en una sola clase, mejorando la **calibración** de las predicciones.

---

### ⚖️ Pesos por clase
Para corregir el desbalance de clases en el entrenamiento se asignan pesos distintos a las clases minoritarias:

- **Clase 2 (Both Fists):** ×1.25  
- **Clase 3 (Both Feet):** ×1.05  

Esto aumenta la contribución de las clases menos frecuentes en la función de pérdida y ayuda a equilibrar el rendimiento entre categorías.

---

### 🧮 Max-Norm
Se aplica un límite **L2 máximo (2.0)** sobre los pesos de los filtros.  
Este método evita la explosión de pesos y mejora la **estabilidad del entrenamiento**.  
Si el modelo sobreajusta, se puede reducir el límite a 1.8; si no aprende bien, puede aumentarse a 2.5–3.0.

---

## 🔮 Inferencia: Test-Time Augmentation (TTA)

Durante la fase de inferencia, cada señal EEG se **desplaza ligeramente en el tiempo** y se promedian las predicciones (logits) para mejorar la robustez.

**Desplazamientos usados (en segundos):**  
[-0.075, -0.05, -0.025, 0, +0.025, +0.05, +0.075]

**Propósito:**  
Aumentar la resistencia a pequeñas desalineaciones temporales entre ensayos.

**Costo:**  
Proporcional al número de desplazamientos (más TTA = más tiempo de inferencia).

**Recomendaciones:**
- Máximo rendimiento: usar 5–7 desplazamientos.  
- Cuando el tiempo sea crítico: usar 3 desplazamientos ([-0.05, 0, +0.05]).

---

## 🔧 Fine-Tuning progresivo por sujeto

Cada sujeto de test se adapta mediante **fine-tuning** interno (4 folds por sujeto), ajustando partes específicas del modelo sin olvidar el conocimiento global.

**Modos de ajuste:**
- **out:** solo la capa de salida  
- **head:** capa totalmente conectada + salida  
- **spatial+head:** bloque espacial + cabeza del modelo  

**Parámetros clave:**
- LR base: 5e-5  
- LR cabeza: 1e-3  
- L2SP: 1e-4  
- Patience (early stopping): 5  
- Validación interna: 20%  

El objetivo es **personalizar el modelo a cada sujeto** preservando las representaciones generales aprendidas.

---

## 📊 Métricas y salidas

- **Accuracy global** y **F1 macro** promedio.  
- Curvas de entrenamiento: `training_curve_foldX.png`  
- Matrices de confusión: `confusion_global_foldX.png`  
- Resultados del fine-tuning: `acc_global`, `acc_ft` y diferencia `Δ(FT-Global)`.

---

## 💡 Cheatsheet de ajuste

| Problema | Causa posible | Solución sugerida |
|:--|:--|:--|
| **Overfitting (Train↑, Val↓)** | Modelo muy complejo o augmentaciones suaves | Subir dropout, aumentar ε de Label Smoothing, reducir F1 |
| **Underfitting (Train↓, Val↓)** | LR demasiado bajo o augmentaciones muy fuertes | Aumentar LR, reducir ruido/jitter |
| **Oscilaciones en validación** | LR alto o ciclos SGDR muy cortos | Aumentar T₀ o reducir LR inicial |
| **Mala precisión en clases 2/3** | Pocas muestras o sin boost | Subir pesos (1.4 / 1.2) |
| **Modelo inestable** | Max-norm demasiado alto | Bajar a 1.8–2.0 |

---


## 🧮 Flujo resumido del pipeline
```bash
raw EDF ─► selección de 8 canales
        └► epoch [-1, 5] s @160 Hz
            └► z-score por época
                └► split Kfold5 (train/val/test)
                    └► estandarización canal-fit(train)
                        └► EEGNet + Augments
                            └► Train con SGDR
                                └► Test con TTA
                                    └► Fine-tuning por sujeto (L2SP)
```


# 🧠 CNN+Transformer para MI-EEG (4 clases, 8 canales)

Con sampler balanceado, Focal Loss, Warmup+Cosine, EMA, TTA y Fine-Tuning Progresivo.

Este modelo combina **CNNs** (para aprender patrones locales) y **Transformers** (para dependencias globales) en la clasificación de señales EEG de imaginación motora. Usa 8 canales seleccionados y una ventana de 6 s, con técnicas modernas de regularización y calibración por sujeto.

---

## 📦 Datos y preprocesamiento

- **Ventana temporal:** [-1, 5] s (6 s totales)  
- **Canales usados (8):** C3, C4, Cz, CP3, CP4, FC3, FC4, FCz  
- **Runs utilizados:**
  - R04, R08, R12 → T1/T2 = left/right  
  - R06, R10, R14 → T1/T2 = both fists/both feet

**Preprocesamiento configurable:**
- Filtro notch 60 Hz (`DO_NOTCH=True`)  
- Band-pass opcional 4–38 Hz (`DO_BANDPASS=False`)  
- Referencia promedio (CAR) opcional  
- Resample opcional (`RESAMPLE_HZ=None`)  
- `ZSCORE_PER_EPOCH=False` → usa estandarización por canal  
- **Normalización:** por canal (fit en train, aplicado a val/test)

---

## 🔀 Partición por sujetos

- **K-fold (5)** definido en `models/folds/Kfold5.json`  
- Cada fold usa ~18 % de los sujetos de entrenamiento como validación  
- **Estratificación:** por etiqueta dominante de cada sujeto para asegurar balance

---

## ⚖️ Sampler balanceado (templado)

Se usa un `WeightedRandomSampler` con pesos:

\[
w = (1 / w_s)^{0.8} \cdot (1 / w_{(s,y)})^{1.0}
\]

donde:
- \(w_s\): número de ensayos del sujeto *s*  
- \(w_{(s,y)}\): número de ensayos de la clase *y* dentro del sujeto *s*

Esto equilibra tanto el número de sujetos como el de clases, evitando dominancia por participantes o categorías más frecuentes.

---

## 🏗️ Arquitectura: CNN + Transformer

### CNN (bloque temporal)

La **CNN** extrae patrones locales de la señal temporal (oscilaciones, desincronizaciones o transientes) y los combina jerárquicamente.

| Capa | Tipo | Propósito | Salida (T=960) |
|:--|:--|:--|:--|
| 1 | Conv 1D (8→32, k=129, stride=2) + GN + ELU + Drop 0.2 | Detecta patrones largos (~0.8 s) | (B, 32, 480) |
| 2 | Depthwise Sep Conv (32→64, k=31, stride=2) | Refina patrones / mezcla espacial | (B, 64, 240) |
| 3 | Depthwise Sep Conv (64→128, k=15, stride=2) | Captura interacciones amplias | (B, 128, 120) |

**Características clave:**  
- **Depthwise separable convolutions:** separan “qué patrón por canal” de “cómo combinar canales”.  
- **GroupNorm:** estabilidad con batches pequeños.  
- **ELU:** activación suave sin saturación.  
- **Dropout:** regularización temporal y espacial.  

La CNN produce una secuencia comprimida (B, 128, ≈120) que resume la dinámica temporal.

---

### Transformer (bloque global)

El **Transformer Encoder** aprende dependencias globales entre las características producidas por la CNN.  
Cada posición “observa” todas las demás mediante **auto-atención**.

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V
$$

- **Q (Query):** qué busca  
- **K (Key):** dónde buscar  
- **V (Value):** información aportada

Así, cada instante combina información de otros momentos relevantes, capturando **relaciones a larga distancia** (p. ej., desincronización temprana y rebote tardío).

**Estructura del Transformer:**
- Proyección conv 1×1 (128→128)  
- Positional encoding (senos/cosenos fijos)  
- Token [CLS] entrenable que resume la secuencia  
- Encoder: 2 capas, 4 cabezas de atención, GELU, dropout 0.1  
- Head final: LayerNorm → Linear → 4 clases

**Qué aprende:**
- Dependencias temporales no locales  
- Relaciones entre fases del ensayo  
- Evidencias dispersas → representación global

---

## 🔗 Sinergia CNN + Transformer

| Componente | Aporta | Ejemplo EEG |
|:--|:--|:--|
| **CNN** | Patrones locales robustos, filtrado y reducción temporal | Ritmos μ, β, transientes |
| **Transformer** | Relaciones globales, sincronías, dependencias largas | Conexión entre desincronización y rebote |
| **Combinados** | Robustez + contexto global | Decisión estable por ensayo |

---

## 🎯 Pérdida y optimización

- **Focal Loss (γ = 1.5):** enfatiza ejemplos difíciles  
- **α por clase:** inversa de frecuencia con boosts  
  - both fists × 1.25  
  - both feet × 1.05  
- **Optimizador:** AdamW (lr = 1e-3, weight_decay = 1e-2)  
- **Scheduler:** Warmup (8 épocas) + Cosine decay (0.1×lr mínimo)  
- **EMA:** promedio exponencial de pesos (decay = 0.9995)

---

## 🌈 Augmentaciones

| Tipo | Parámetros globales | Propósito |
|:--|:--|:--|
| **Jitter temporal** | ±3 % T (~±180 ms) | Robustez a desalineación del onset |
| **Ruido gaussiano** | σ = 0.03 | Simula ruido fisiológico/electrónico |
| **Channel dropout** | 1 canal (p = 0.15) | Robustez ante fallos de electrodos |

En **fine-tuning**, se suavizan: jitter 2 %, σ 0.02, p 0.10.

---

## 🔮 Inferencia (TTA / Subwindows)

- **TTA:** desplaza la señal ±75 ms y promedia los logits  
- **Subwindows:** evalúa tramos de 4.5 s cada 1.5 s → promedio  
- **Modo combinado:** mezcla ambos promedios  

→ Invarianza ante errores de tiempo y mayor estabilidad de predicción.

---

## 🔧 Fine-Tuning progresivo por sujeto

Ajuste personalizado en dos etapas:

| Etapa | Capas entrenadas | Épocas | Propósito |
|:--|:--|:--|:--|
| 1 | Solo head | 8 | Calibrar salida sin alterar features |
| 2 | Todo modelo (unfreeze) | 8 | Adaptar representación al sujeto |

**CV interna:** 4 folds  
**Optimizador FT:** AdamW (backbone lr 2e-4, head lr 1e-3)  
**Pérdida:** Focal Loss re-estimada por sujeto (con boosts)  
**Augmentaciones:** suaves  

---

## 📊 Métricas y resultados

- Early stopping por F1 macro (validación de sujetos)  
- Curvas: `training_curve_foldX.png`  
- Matrices: `confusion_global_foldX.png`, `confusion_ft_foldX.png`  
- Resumen: Accuracy global, F1 macro, FT accuracy y Δ (FT − Global)

---

## ⚙️ Diagnóstico y ajuste rápido

| Problema | Posible causa | Ajuste sugerido |
|:--|:--|:--|
| **F1 inestable** | LR alto / warmup corto | Bajar LR (7e-4 – 1e-3), subir warmup (10–12) |
| **Underfit** | Augment fuerte / modelo pequeño | Reducir ruido/jitter, aumentar d_model a 160–192 |
| **Overfit** | Regularización débil | Aumentar dropout (0.25–0.4), WD 2e-2, γ 1.7–2.0 |
| **Recall bajo clase 2/3** | Desbalance residual | Subir α[2]/α[3], boost FT, ajustar sampler (b > 1.0) |
| **FT no mejora** | Ajuste agresivo | Reducir augment FT, usar solo etapa 1, bajar lr head |

---

## 🧩 Flujo de datos

EEG (B, 8, T = 960)  
→ Conv1d 8→32 (k129, s2) → GN → ELU → Drop  
→ SepConv 32→64 (k31, s2)  
→ SepConv 64→128 (k15, s2)  
→ Conv 1×1 128→128 → Drop  
→ Transpose (B, L ≈ 120, D = 128)  
→ + PosEnc → concat CLS  
→ Transformer Encoder × 2  
→ CLS → LayerNorm → Linear → logits (4)

---

## 🧠 Cómo funciona el CNN y el Transformer

### CNN — Extracción local de patrones

La CNN aprende filtros que responden a **patrones temporales específicos** (oscilaciones, desincronizaciones, transientes).  
Las convoluciones con *stride* reducen resolución temporal y amplían el contexto.  
Las **depthwise separable convolutions** separan la detección temporal (por canal) de la mezcla espacial, generando mapas de activación compactos y expresivos.

### Transformer — Aprendizaje global de dependencias

El Transformer usa **auto-atención** para que cada instante observe todos los demás, asignando pesos según relevancia.  
Así capta **relaciones de largo alcance** sin necesidad de más capas convolucionales.  
El token [CLS] sintetiza toda la secuencia en un vector representativo.  
Cada cabeza de atención analiza las relaciones en un subespacio distinto.

### Sinergia

- **CNN:** aprende *qué* patrones existen  
- **Transformer:** aprende *cómo* se combinan en el tiempo  
- **Combinación:** robustez al ruido + comprensión de contextos largos y complejos
