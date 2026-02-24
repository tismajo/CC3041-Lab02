# README

## Clasificador Binario de Spam / No Spam (Dataset en Español)

---

## 1. Descripción del Problema

El objetivo de este proyecto es construir un **clasificador binario** capaz de identificar mensajes de texto como:

* **Spam (1)**
* **No Spam / Ham (0)**

Se utilizó el dataset:

> *Spam / Ham Dataset para detección de Spam en Español*
> [https://www.kaggle.com/datasets/alexandercarreo/es-spamham-dataset-para-deteccin-de-spam](https://www.kaggle.com/datasets/alexandercarreo/es-spamham-dataset-para-deteccin-de-spam)

El enfoque busca seleccionar un modelo con **el menor error posible**, validado mediante métricas estándar de clasificación binaria.

---

## 2. Metodología General

El flujo de trabajo implementado fue el siguiente:

1. Carga y exploración del dataset
2. Preprocesamiento de texto (NLP)
3. Vectorización del texto (TF-IDF)
4. División en conjuntos de entrenamiento, validación y prueba
5. Entrenamiento del modelo
6. Validación cruzada
7. Evaluación con múltiples métricas

---

## 3. Preprocesamiento de Texto (NLP)

Dado que el dataset contiene texto en lenguaje natural, fue necesario aplicar técnicas básicas de procesamiento de lenguaje natural:

### 3.1 Normalización

* Conversión a minúsculas.
* Eliminación de caracteres no alfabéticos mediante expresiones regulares.

### 3.2 Remoción de Stopwords

Se eliminaron palabras comunes en español que no aportan información discriminativa (ej. “el”, “la”, “de”, “y”), utilizando `nltk.corpus.stopwords`.

### 3.3 Lematización

Se aplicó lematización para reducir las palabras a su forma base. Esto ayuda a reducir la dimensionalidad del vocabulario.

El objetivo del preprocesamiento fue:

* Reducir ruido.
* Disminuir dimensionalidad.
* Mejorar la capacidad del modelo para capturar patrones relevantes.

---

## 4. Representación Numérica del Texto (Embedding)

Los modelos de Machine Learning requieren entradas numéricas. Por ello se utilizó:

### TF-IDF (Term Frequency – Inverse Document Frequency)

Configuración utilizada:

* `max_features=5000`
* `ngram_range=(1,2)`

El uso de:

* **Unigramas** permite capturar palabras individuales.
* **Bigramas** permite capturar patrones frecuentes en spam como:

  * “haz clic”
  * “dinero rápido”
  * “oferta limitada”

TF-IDF pondera términos frecuentes en un documento pero raros en el corpus, lo cual es particularmente efectivo en detección de spam.

---

## 5. División del Dataset

Se realizó una división estratificada para mantener la proporción de clases:

* 80% → Entrenamiento + Validación
* 20% → Prueba

Posteriormente:

* 80% → Entrenamiento
* 20% → Validación

La división estratificada asegura que ambas clases estén balanceadas en todos los subconjuntos.

---

## 6. Modelo Utilizado

Se utilizó:

### Logistic Regression

Justificación:

* Modelo lineal robusto en espacios de alta dimensión.
* Excelente desempeño en clasificación de texto con TF-IDF.
* Permite obtener probabilidades (necesarias para ROC AUC y log loss).
* Bajo riesgo de sobreajuste comparado con modelos más complejos.

Parámetros principales:

* `max_iter=1000`

---

## 7. Validación Cruzada

Se implementó validación cruzada de 5 folds utilizando `Pipeline`, integrando:

* TF-IDF
* Logistic Regression

Esto garantiza que:

* Cada fold aprende su propio vocabulario.
* Se evita fuga de información.
* La medición de desempeño sea estadísticamente más robusta.

Métrica utilizada en validación cruzada:

* **F1 Score**

---

## 8. Métricas de Evaluación

Para evaluar el desempeño del modelo se utilizaron las siguientes métricas:

### Accuracy

Proporción total de predicciones correctas.

### Precision

Proporción de mensajes clasificados como spam que realmente eran spam.

Importante cuando se quiere evitar falsos positivos.

### Recall

Proporción de spam reales que fueron detectados correctamente.

Importante cuando se quiere minimizar falsos negativos.

### F1 Score

Media armónica entre precision y recall.
Métrica balanceada y adecuada cuando las clases pueden estar desbalanceadas.

### ROC AUC

Mide la capacidad del modelo para separar ambas clases en distintos umbrales.

### Binary Crossentropy (Log Loss)

Evalúa la calidad de las probabilidades predichas.

---

## 9. Evaluación Final

Se evaluó el modelo tanto en:

* Conjunto de validación
* Conjunto de prueba (no visto durante el entrenamiento)

Además se generaron:

* Matriz de confusión
* Curva ROC

Esto permite analizar:

* Errores tipo I (falsos positivos)
* Errores tipo II (falsos negativos)
* Capacidad discriminativa del modelo

---

## 10. Justificación de Decisiones

| Decisión                   | Justificación                                                    |
| -------------------------- | ---------------------------------------------------------------- |
| Uso de TF-IDF              | Representación efectiva para texto corto y clasificación de spam |
| Uso de Logistic Regression | Modelo lineal adecuado para datos de alta dimensión              |
| Validación cruzada         | Medición robusta del desempeño                                   |
| División estratificada     | Mantener balance de clases                                       |
| Métricas múltiples         | Evaluación completa del desempeño                                |

---

## 11. Conclusión

Se construyó un clasificador binario para detección de spam en español utilizando técnicas estándar de NLP y Machine Learning.

La combinación de:

* Preprocesamiento de texto
* Vectorización TF-IDF
* Regresión Logística
* Validación cruzada
* Evaluación con múltiples métricas

permite obtener un modelo sólido, con bajo error y capacidad de generalización adecuada.

El enfoque implementado cumple con los requisitos establecidos en las instrucciones del problema, utilizando buenas prácticas en:

* Separación de datos
* Representación textual
* Evaluación estadística
* Validación del desempeño

---

## 12. ¿Cómo ejecutarlo?

Para ejecutar el proyecto:

1. Descargar el dataset desde Kaggle.
2. Instalar dependencias:

   ```
   pandas
   numpy
   nltk
   scikit-learn
   matplotlib
   seaborn
   ```
3. Ejecutar el script principal.