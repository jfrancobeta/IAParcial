# Título del Proyecto
Taller Evaluable

## Descripción General
Este repositorio contiene la solución al taller evaluable del segundo corte de la asignatura **Inteligencia Artificial, 2025A**. El objetivo principal del taller es desarrollar un chatbot conversacional utilizando modelos de lenguaje de gran tamaño (LLMs) con PyTorch y Hugging Face. A través de cinco ejercicios prácticos, se implementan técnicas fundamentales para la creación, personalización y despliegue de asistentes virtuales basados en tecnologías de procesamiento de lenguaje natural (NLP).

## Autor
**Juan David Franco**  
Estudiante de Ingeniería de Sistemas

## Objetivos de Aprendizaje
1. Comprender la arquitectura básica de los modelos transformer utilizados en chatbots.
2. Aprender a cargar y utilizar modelos pre-entrenados con Hugging Face y PyTorch.
3. Implementar técnicas para optimizar el rendimiento de modelos LLM en dispositivos con recursos limitados.
4. Desarrollar un sistema de diálogo completo con manejo de contexto conversacional.
5. Personalizar la "personalidad" y respuestas del chatbot mediante técnicas de fine-tuning.

## Estructura del Proyecto
El proyecto está dividido en cinco ejercicios, cada uno abordando un aspecto clave del desarrollo de chatbots:

### Ejercicio 1: Configuración del Entorno y Carga de Modelo Base
- **Descripción:** Configuración del entorno de desarrollo y carga de un modelo pre-entrenado.
- **Archivo relacionado:** `Ejercicio1.py`

### Ejercicio 2: Procesamiento de Entrada y Generación de Respuestas
- **Descripción:** Implementación de funciones para preprocesar la entrada del usuario y generar respuestas coherentes.
- **Archivo relacionado:** `Ejercicio2.py`

### Ejercicio 3: Manejo de Contexto Conversacional
- **Descripción:** Implementación de un sistema para mantener el contexto de la conversación.
- **Archivo relacionado:** `Ejercicio3.py`

### Ejercicio 4: Optimización del Modelo para Recursos Limitados
- **Descripción:** Implementación de técnicas de optimización para mejorar la velocidad de inferencia y reducir el consumo de memoria.
- **Archivo relacionado:** `Ejercicio4.py`

### Ejercicio 5: Personalización del Chatbot y Despliegue
- **Descripción:** Personalización del comportamiento del chatbot y creación de una interfaz web interactiva.
- **Archivo relacionado:** `Ejercicio5.py`


Diferencias entre modelos encoder-only, decoder-only y encoder-decoder
Los modelos encoder-only son ideales para tareas de clasificación y análisis de texto. Los modelos decoder-only son más adecuados para generación de texto, como chatbots. Los modelos encoder-decoder son útiles para tareas de traducción y resumen.

Concepto de "temperatura" en generación de texto
La temperatura controla la aleatoriedad en la generación de texto. Valores bajos generan respuestas más conservadoras, mientras que valores altos producen respuestas más creativas.

Técnicas para reducir "alucinaciones" en chatbots

A nivel de inferencia: Ajustar la temperatura y usar técnicas como top-p sampling.
A nivel de prompt engineering: Proporcionar instrucciones claras y específicas al modelo.

1. Clona este repositorio:
   ```bash
   git clone https://github.com/jfrancobeta/IAParcial.git
   cd IAParcial
