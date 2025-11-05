FROM apache/spark:3.5.5-python3

USER root

RUN pip install --no-cache-dir schedule pymongo numpy pandas scikit-learn

RUN mkdir -p /home/spark/.ivy2/cache && \
    chown -R spark:spark /home/spark/.ivy2

RUN mkdir -p /opt/spark-tmp && \
    chown -R spark:spark /opt/spark-tmp

RUN mkdir -p /opt/spark/spark-events && \
    chown -R spark:spark /opt/spark/spark-events

COPY scheduler.py /opt/scheduler.py

USER spark

ENTRYPOINT []
CMD ["python3", "/opt/scheduler.py"]
