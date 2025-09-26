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

## 4. Primer modelo: FBCSP + LDA

Tras el preprocesamiento, se implementó un **modelo base** siguiendo metodologías clásicas en BCI:

### FBCSP (Filter Bank Common Spatial Patterns)
- Banco de filtros **denso 8–30 Hz** en pasos de 2 Hz.  
- Extracción de **6–8 componentes CSP** por sub-banda.  
- Cálculo de **log-varianza** en cada componente como feature.  
- Concatenación de todas las features en un único vector por época.  

### LDA (Linear Discriminant Analysis)
- Clasificador lineal con **shrinkage automático** (`lsqr`, `shrinkage='auto'`).  
- Elegido por su simplicidad, robustez frente a alta dimensionalidad y efectividad comprobada en EEG MI.  

### Evaluación intra-sujeto
- Validación cruzada (K-fold).  
- Resultados ejemplo:
  - **S001**: ACC ≈ 93% | F1 ≈ 92%  
  - **S002**: ACC ≈ 78% | F1 ≈ 77%  
  - **S003**: ACC ≈ 50% | F1 ≈ 48%  
  - **S004**: ACC ≈ 76% | F1 ≈ 70%  

### Evaluación inter-sujeto (LOSO)
- Esquema Leave-One-Subject-Out.  
- Con calibración mínima de **k=5 épocas/clase** para ajustar al sujeto test.  
- Resultado global: **ACC ≈ 63% | F1 ≈ 59%**.  

### Conclusión del modelo base
- Intra-sujeto: rendimientos altos (70–93%), comparables con literatura.  
- Inter-sujeto (LOSO): rendimiento más bajo (≈60%), consistente con estudios previos.  
- Este modelo sirve como baseline sólido para comparar en siguientes etapas (SVM, Riemannianos, CNNs, Transformers).

---

## 5. Próximos pasos
- Incorporar **flags de QA** para sujetos problemáticos.  
- Ajustar ventanas temporales (crop adaptativo).  
- Explorar clasificadores alternativos y modelos profundos.  
- Comparar FBCSP+LDA con modelos Riemannianos y CNNs.

