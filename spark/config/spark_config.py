import os

from pyspark.sql import SparkSession


def create_spark_session(app_name="WeatherMLOps"):
    mongo_uri = os.getenv("MONGO_URI") or os.getenv(
        "MONGO_URL",
        "mongodb://admin:admin123@mongodb:27017/weather_db?authSource=admin",
    )

    spark = (
        SparkSession.builder.appName(app_name)
        .config("spark.mongodb.input.uri", mongo_uri)
        .config("spark.mongodb.output.uri", mongo_uri)
        .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:10.5.0")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        # Local runs default to 200 shuffle partitions, which is mostly task
        # scheduling overhead on a 4-core runner.
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    return spark


FEATURES_CONFIG = {
    "window_sizes": [6, 12, 24],  # Hours rolling windows
    "lag_periods": [1, 2, 3, 6, 12],  # Hours lag features
    "target_horizon": 1,  # Hours prediction
}


ML_CONFIG = {
    "data_split": {"train": 0.6, "validation": 0.2, "test": 0.2, "seed": 42},
    # Two folds, not three: the third fold bought little and every fold
    # multiplies the number of fits.
    "cross_validation": {"num_folds": 2, "seed": 42},
    # Grids are deliberately small: a full sweep across three algorithms did not
    # finish inside the GitHub runner (the nightly train step was cancelled at
    # 30, then at 60 minutes). These combinations keep the candidate algorithms
    # while cutting fits from ~200 to ~36, and the trainers cache the split
    # frames so each fit no longer re-reads Atlas.
    "gradient_boosted_trees": {
        "maxDepth": [5, 7],
        "maxIter": [50],
        "stepSize": [0.1],
    },
    "random_forest_regressor": {
        "numTrees": [40, 80],
        "maxDepth": [5, 8],
        "minInstancesPerNode": [1],
    },
    "linear_regression": {"elasticNetParam": [0.0, 0.5, 1.0], "regParam": [0.01, 0.1]},
    "random_forest_classifier": {
        "numTrees": [40, 80],
        "maxDepth": [5, 8],
        "minInstancesPerNode": [1],
    },
    "gradient_boosted_trees_classifier": {
        "maxDepth": [5, 7],
        "maxIter": [50],
        "stepSize": [0.1],
    },
}
