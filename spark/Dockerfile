FROM apache/spark:3.5.5-python3

USER root

COPY requirements.txt .
COPY spark-scheduler/requirements.txt scheduler-requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r scheduler-requirements.txt

RUN mkdir -p /home/spark/.ivy2/cache && \
    chown -R spark:spark /home/spark/.ivy2

RUN mkdir -p /opt/spark-tmp && \
    chown -R spark:spark /opt/spark-tmp

RUN mkdir -p /opt/spark/spark-events && \
    chown -R spark:spark /opt/spark/spark-events

COPY spark-jobs/ /opt/spark-jobs/
COPY config/ /opt/config/
COPY spark-scheduler/scheduler.py /opt/scheduler.py

USER spark

# WORKDIR /opt/spark-jobs
# ENTRYPOINT []
CMD ["python3", "/opt/scheduler.py"]

