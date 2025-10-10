FROM apache/spark:3.5.5-python3

USER root

# Install the schedule package
RUN pip install --no-cache-dir schedule

# Create Ivy cache directory with proper permissions
RUN mkdir -p /home/spark/.ivy2/cache && \
    chown -R spark:spark /home/spark/.ivy2

# Copy scheduler script
COPY scheduler.py /opt/scheduler.py

# Switch back to spark user
USER spark

CMD ["python3", "/opt/scheduler.py"]
