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

El repositorio está organizado en varios directorios principales:

- `data_normalization/`: Obtiene la media y la desviación estándar del conjunto de entrenamiento, por canal RGB. Elabora un .txt para su posteriormente, normalizar el conjunto de datos para tener media 0 y desviación estándar 1.
