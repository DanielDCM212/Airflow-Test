doc = """
# Prueba Data Engineering 

DAG presentado como resultado de realizar la prueba para Data Enginnering. 

Este DAG es basado en el Kernel https://www.kaggle.com/code/kabure/predicting-credit-risk-model-pipeline/notebook en 
el cual se entrenan modelos para la predicción de riesgo en créditos. 

La cantidad de datos ocupados para esta prueba no presenta un riesgo o una complicación para XCOM, sin embargo, 
pensando esto como un flujo en el que pudiese haber mucha información, decidí ir más por una estrategia Intermediary 
Data Storage, para limitarnos a algo local, no se ocupó algún bucket en la nube, por lo que se opte por generar una 
carpeta que fungira como bucket. 

Los Datos Finales se encuentran en bucket/Test/results

Dependencias:

* scikit-learn >= 1.0.2
* pandas >= 1.3.5
* numpy >= 1.21.6
* joblib >= 1.1.0

"""
