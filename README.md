# Classification of Motor Imagery EEG Signals Using CNN–Transformer Hybrid Models
### Autor: Joel Cuascota

Clasificación de señales EEG de imaginación motora (motor imagery) utilizando un modelo híbrido **CNN+Transformer** para el reconocimiento de movimientos imaginados de manos (izquierda/derecha).

## Descripción del Proyecto

Este proyecto implementa un sistema de clasificación de señales electroencefalográficas (EEG) basado en imaginación motora, utilizando un modelo híbrido que combina redes neuronales convolucionales (CNN) con arquitectura Transformer. El objetivo principal es clasificar señales EEG correspondientes a la imaginación de movimientos de la mano izquierda o derecha.

### Características Principales

- **Modelo Híbrido CNN+Transformer**: Combina extracción de características espaciotemporales (CNN) con modelado de dependencias temporales largas (Transformer)
- **Clasificación Binaria**: Distinción entre imaginación motora de mano izquierda vs derecha
- **Dataset**: PhysioNet EEG Motor Movement/Imagery Dataset
- **Canales Motores**: 8 canales específicos del área motora (C3, C4, Cz, CP3, CP4, FC3, FC4, FCz)
- **Validación Robusta**: K-Fold cross-validation por sujeto (5 folds)
- **Técnicas Avanzadas**:
  - Test-Time Augmentation (TTA)
  - Exponential Moving Average (EMA) de pesos
  - Focal Loss para manejo de desbalance de clases
  - Fine-tuning por sujeto
  - Interpretabilidad mediante mapas de atención

## Arquitectura del Modelo

### EEGCNNTransformer

El modelo se compone de tres etapas principales:

1. **Backbone Convolucional**:
   - Stem inicial con conv1d (129 kernels)
   - Bloques depthwise separable para reducción de parámetros
   - GroupNorm y ELU como activación

2. **Encoder Transformer**:
   - Multi-head self-attention para modelar dependencias temporales
   - Codificación posicional sinusoidal
   - Token CLS para clasificación

3. **Head de Clasificación**:
   - LayerNorm + Linear layer
   - Salida de 2 clases (left/right)

### Hiperparámetros Principales

```python
D_MODEL = 144              # Dimensión del embedding
N_HEADS = 4                # Cabezas de atención
N_LAYERS = 1               # Capas del Transformer
BATCH_SIZE = 64            # Tamaño de batch
BASE_LR = 5e-4            # Learning rate
EPOCHS = 60                # Épocas de entrenamiento
TMIN, TMAX = -1.0, 5.0    # Ventana temporal (6s)
```

## Instalación

### Requisitos

- Python >= 3.8
- CUDA compatible GPU (recomendado)

### Instalación de Dependencias

```bash
git clone https://github.com/JACS002/EEG_Clasificador.git
cd EEG_Clasificador
pip install -r requirements.txt
```

### Dependencias Principales

- **PyTorch** >= 2.0 - Framework de deep learning
- **MNE** >= 1.5 - Procesamiento de señales EEG
- **NumPy** >= 1.24 - Computación numérica
- **scikit-learn** >= 1.3 - Métricas y validación
- **matplotlib** >= 3.7 - Visualización

## Dataset

El proyecto utiliza el **PhysioNet EEG Motor Movement/Imagery Dataset**, que contiene:

- 109 sujetos
- 64 canales de EEG
- Tareas de imaginación motora de manos y pies
- Frecuencia de muestreo: 160 Hz

### Preprocesamiento

1. **Selección de canales**: 8 canales del área motora
2. **Filtrado**: Notch filter en 60 Hz (ruido eléctrico)
3. **Epoching**: Ventanas de 6 segundos (-1s a 5s respecto al evento)
4. **Normalización**: Z-score por canal usando estadísticas del train set
5. **Runs utilizados**: 4, 8, 12 (imaginación motora L/R)

## Uso

### Entrenamiento del Modelo

El notebook principal se encuentra en:

```
models/04_hybrid/cnntransformer2c.ipynb
```

**Flujo de entrenamiento:**

1. **Entrenamiento Global** (5-fold cross-validation por sujeto)
   - Entrenamiento en múltiples sujetos
   - Validación por sujetos no vistos
   - EMA de pesos activado
   - TTA en evaluación

2. **Fine-tuning por Sujeto** (opcional)
   - Adaptación a características individuales
   - Dos etapas: congelado + descongelado del backbone
   - Data augmentation (jitter, ruido, channel dropout)

### Pipeline de Evaluación

```python
# El notebook implementa:
1. Carga de datos por sujeto
2. División en folds según Kfold5.json
3. Entrenamiento con early stopping
4. Evaluación con TTA
5. Métricas: Accuracy, F1, Precision, Recall
6. Visualizaciones: matrices de confusión, curvas de aprendizaje
```

## Resultados

El modelo logra:

- **Validación Cruzada**: Accuracy y Macro F1 ~82% en evaluación inter-sujeto (5-fold CV)

### Métricas Reportadas

- Accuracy por fold y promedio
- F1-score macro/weighted
- Matrices de confusión


## Modelos Adicionales

### Extensión a 4 Clases (cnntransformer4c.ipynb)

El notebook `cnntransformer4c.ipynb` extiende el modelo para clasificar **4 tipos de imaginación motora**:
- Mano izquierda
- Mano derecha
- Ambos puños
- Ambos pies

**Nota**: Este es un problema más desafiante con menor accuracy (~54%), útil para evaluar la capacidad del modelo en escenarios multi-clase más complejos.

## Características Técnicas

### Reproducibilidad

Configuración completa de semillas para garantizar experimentos reproducibles:
```python
RANDOM_STATE = 42
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
```

## Citación

Este proyecto es parte del trabajo de tesis para el grado de **Ingeniería en Ciencias de la Computación** en la **Universidad San Francisco de Quito**.

Si utilizas este código, modelos o metodología en tu investigación, por favor cita este repositorio:

```bibtex
@misc{cuascota2025eeg_cnn_transformer,
  author = {Cuascota, Joel},
  title = {Classification of Motor Imagery EEG Signals Using CNN–Transformer Hybrid Models},
  year = {2025},
  month = {12},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/JACS002/EEG_Clasificador}},
  note = {Trabajo de tesis, Universidad San Francisco de Quito}
}
```

**Formato APA:**
> Cuascota, J. (2025). *Classification of Motor Imagery EEG Signals Using CNN–Transformer Hybrid Models* [Repositorio de GitHub]. Universidad San Francisco de Quito. https://github.com/JACS002/EEG_Clasificador

## 📝 Licencia

Este proyecto es de código abierto y está disponible bajo la licencia MIT.

