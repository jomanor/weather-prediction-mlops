FROM apache/spark:3.5.5-python3

USER root
COPY spark-scheduler/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /home/spark/.ivy2/cache && \
    chown -R spark:spark /home/spark/.ivy2

RUN mkdir -p /opt/spark-tmp && \
    chown -R spark:spark /opt/spark-tmp

RUN mkdir -p /opt/spark/spark-events && \
    chown -R spark:spark /opt/spark/spark-events

COPY spark-scheduler/scheduler.py /opt/scheduler.py
COPY spark-base/spark-jobs/ /opt/spark-jobs/
COPY spark-base/config/ /opt/config/

USER spark

ENTRYPOINT []
CMD ["python3", "/opt/scheduler.py"]
