import schedule
import time
import subprocess
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_spark_job(job_file, job_name):
    logger.info(f"Starting {job_name}...")

    cmd = [
        "/opt/spark/bin/spark-submit",
        "--packages", "org.mongodb.spark:mongo-spark-connector_2.12:10.5.0",
        "--master", "spark://spark-master:7077",
        f"/opt/spark-jobs/{job_file}"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"{job_name} completed successfully")
        else:
            logger.error(f"{job_name} failed: {result.stderr}")
    except Exception as e:
        logger.error(f"Error running {job_name}: {e}")

def run_feature_engineering():
    run_spark_job("batch_processing.py", "Feature Engineering")

def run_ml_training():
    run_spark_job("ml_training.py", "ML Training")

# TODO create an inference script
def run_quick_predictions():
    logger.info("Running quick predictions...")
    pass

logger.info("Setting up job schedule...")

schedule.every(6).hours.do(run_feature_engineering)
schedule.every().day.at("02:00").do(run_ml_training)
schedule.every().hour.do(run_quick_predictions)

logger.info("Running initial feature engineering...")
run_feature_engineering()

logger.info("Scheduler started. Jobs will run automatically.")
logger.info("Scheduled jobs:")
logger.info("  - Feature Engineering: Every 6 hours")
logger.info("  - ML Training: Daily at 02:00")
logger.info("  - Quick Predictions: Every hour")

while True:
    schedule.run_pending()
    time.sleep(60)