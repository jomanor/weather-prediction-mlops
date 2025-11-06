FROM apache/spark:3.5.5-python3

USER root
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY spark-jobs/ /opt/spark-jobs/
COPY config/ /opt/config/

USER spark
WORKDIR /opt/spark-jobs