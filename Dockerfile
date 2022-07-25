FROM apache/airflow:2.3.0
USER airflow
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --user -r requirements.txt