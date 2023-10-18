# Técnicas de Explicabilidad para Redes Neuronales Profundas en el Contexto de Aprendizaje Multivista

Este repositorio contiene la investigación y recursos asociados con el Trabajo de Fin de Grado (TFG) del Grado de Ingeniería Informática de la Universidad de Granada.

## 🧐 Acerca del Proyecto

**Autor**: Pablo García Sánchez

**Título**: 
Técnicas de explicabilidad para redes neuronales profundas en el contexto de aprendizaje multivista

### Resumen

Investigamos técnicas avanzadas de explicabilidad en inteligencia artificial (xIA) con el propósito de potenciar la comprensión y transparencia de las redes neuronales convolucionales (CNN) dentro del ámbito del aprendizaje multivista. Esta investigación emplea una CNN multivista diseñada para estimar la edad a partir de proyecciones de modelos tridimensionales de la sínfisis púbica. El objetivo principal es extender y mejorar su capacidad explicativa, proporcionando intuiciones más profundas sobre su funcionamiento y decisiones.

### Abstract

We investigate advanced explainable artificial intelligence (xAI) techniques with the aim of enhancing the understanding and transparency of convolutional neural networks (CNNs) within the multi-view learning context. This research employs a multi-view CNN designed to estimate age from projections of three-dimensional models of the pubic symphysis. The primary goal is to expand and enhance its explanatory capacity, offering deeper insights into its operation and decision-making

## 🛠 Herramientas y Tecnologías Utilizadas

- Python 3.10.12
- Pytorch 2.0.1
 
## 📂 Estructura del Repositorio

Este repositorio consta de varios ficheros principales, cada uno con funciones específicas:
  
### `data_normalization.py`
- **Función**: Calcula y registra la media y desviación estándar del conjunto de entrenamiento para cada canal RGB.
- **Resultado**: Genera un archivo `.txt` que facilita la normalización del conjunto de datos a una media 0 y una desviación estándar 1.
  
### `ranges_of_ages.py`
- **Función**: Establece los rangos de edad para calcular métricas como el Erro Medio Absoluto (MAE) y el Error Cuadrático Medio (MSE).
  
### `training.py`
- **Función**: Entrena las arquitecturas propuestas.
- **Arquitecturas**:
  - Panorama-CNN
  - Resnet-CNN
  - Panorama-CNNEx
  - Resnet-CNNEx
  
  **Uso**:
  ```bash
  training.py [PANORAMACNN:True/False] [Explainable:True/False]

### `utils.py`
- **Función**:
  - Define el Dataset: Permite cargar las imágenes en memoria normalizadas para que la media sea 0 y la desviación estándar 1
  - Define las capas PS y VS: Define las capas VS y PS, basado en: R. Song, Y. Liu, and P. L. Rosin. "[Mesh Saliency via Weakly Supervised Classification-for-Saliency CNN](https://doi.org/10.1109/TVCG.2019.2928794)." *IEEE Transactions on Visualization and Computer Graphics*, vol. 27, no. 1, pp. 151-164, 1 Jan. 2021.
  - Define las arquitecturas Panorama-CNN, Panorama-CNNEx, Resnet-CNN y Resnet-CNNEx.
