# Análisis de Sentimiento en Reseñas de Amazon

## Descripción del Proyecto
Este proyecto implementa un análisis de sentimiento en reseñas de productos de Amazon.
El objetivo es clasificar automáticamente las reseñas en **Positivas** o **Negativas**
mediante diferentes modelos de Machine Learning y Deep Learning, y comparar sus métricas de rendimiento.

## Dataset
- **Fuente oficial:** [Amazon Review Data (2018)](https://nijianmo.github.io/amazon/index.html)
- **Archivo utilizado:** `Industrial_and_Scientific_5.json.gz`
- **Procesamiento realizado:**
  - Eliminación de reseñas neutrales (rating = 3).
  - Clasificación binaria: `1 = Positivo` y `0 = Negativo`.

## Estructura del Notebook
1. **Preprocesamiento de texto:** limpieza, lematización y eliminación de stopwords.
2. **Análisis exploratorio de datos (EDA):** visualización de la longitud de reseñas y nube de palabras.
3. **Modelos clásicos (TF-IDF):** Logistic Regression, Naive Bayes, Random Forest, Linear SVC.
4. **Modelos de Deep Learning:** GRU y Conv1D entrenados sobre secuencias tokenizadas.
5. **Comparación de resultados:** métricas de Accuracy, F1-score y tiempo de entrenamiento.

## Resultados Principales
El mejor modelo encontrado fue:
- **Linear SVC** con:
  - Accuracy: 0.9588
  - F1-score: 0.9783
  - Tiempo de entrenamiento: 0.16 s

## Tecnologías Utilizadas
- Python 3
- Pandas, NumPy, Matplotlib
- Scikit-learn
- TensorFlow / Keras
- HuggingFace Transformers

## Cómo Ejecutar el Notebook
1. Clonar este repositorio.
2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Ejecutar el notebook en Jupyter o VSCode:
   ```bash
   jupyter notebook nlp_amazon_sentiment_comparacion_final.ipynb
   ```

## Autor
Proyecto desarrollado para portafolio profesional con el objetivo de mostrar habilidades en:
- Procesamiento de lenguaje natural (NLP)
- Evaluación y comparación de modelos de ML y DL

✉️ Cecilia Ledesma (ceciledes32@gmail.com)

---
