FROM apache/airflow:2.6.0 as airflow-mlops
RUN pip install scikit-learn pandas requests clearml beautifulsoup4 joblib pickle5
COPY datasets/ /datasets/