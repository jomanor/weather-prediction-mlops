FROM apache/spark:3.5.5-python3

USER root

RUN pip install --no-cache-dir schedule

RUN mkdir -p /home/spark/.ivy2/cache && \
    chown -R spark:spark /home/spark/.ivy2

COPY scheduler.py /opt/scheduler.py

USER spark

ENTRYPOINT []
CMD ["python3", "/opt/scheduler.py"]
