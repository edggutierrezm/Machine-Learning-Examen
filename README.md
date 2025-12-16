Predicción de Riesgo de Incumplimiento de Crédito (Home Credit Default Risk)

Resumen del Proyecto
Este proyecto tiene como objetivo construir un modelo de Machine Learning para predecir la probabilidad de que un solicitante de crédito incumpla su pago a una institución financiera. Se aborda como un problema de Clasificación Binaria utilizando el complejo dataset de la competición Home Credit Default Risk de Kaggle.

La implementación cumple con estrictos requisitos de ingeniería de software, incluyendo la modularización del código bajo la metodología CRISP-DM y el despliegue del modelo final a través de una API REST.


Objetivos Técnicos y Desafíos
Construir un modelo robusto capaz de predecir la variable binaria TARGET (incumplimiento).

Integración de Datos: Fusionar la tabla principal (application_train/test.csv) con datos de burós de crédito, solicitudes previas y comportamiento de pago (como se detalla en el diagrama).

Ingeniería de Características: Generar features agregadas (ej. promedios, sumas, ratios) a partir de las tablas secundarias antes de la unión.

Manejo de Desbalance: Abordar el desafío de un dataset desbalanceado y de alta dimensionalidad.


Estructura del Proyecto (Microservicios)
El proyecto está organizado siguiendo una estructura modular basada en CRISP-DM, donde cada carpeta representa una fase independiente y ejecutable.

01_data_understanding/
Análisis Exploratorio de Datos (EDA) inicial.

02_data_preparation/
Limpieza, Ingeniería de Características y Preprocesamiento.

03_modeling/
Entrenamiento, validación cruzada y ajuste de hiperparámetros.

04_evaluation/
Evaluación final del modelo campeón y generación de reportes (AUC, matriz de confusión).

05_deployment/
Código de la API REST para servir el modelo.

artifacts/
Almacenamiento de salidas: modelo entrenado (.pkl) y scalers utilizados.

requirements.txt
Listado de dependencias de Python.

