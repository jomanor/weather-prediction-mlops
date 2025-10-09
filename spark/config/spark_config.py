from pyspark.sql import SparkSession
import os

def create_spark_session(app_name="WeatherMLOps"):

    mongo_uri = os.getenv(
        'MONGO_URL', 
        'mongodb://admin:admin123@mongodb:27017/weather_db?authSource=admin'
    )
    
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.mongodb.input.uri", mongo_uri) \
        .config("spark.mongodb.output.uri", mongo_uri) \
        .config("spark.jars.packages", 
                "org.mongodb.spark:mongo-spark-connector_2.12:10.2.0") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    return spark

FEATURES_CONFIG = {
    'window_sizes': [6, 12, 24],  # Hours rolling windows
    'lag_periods': [1, 2, 3, 6, 12],  # Hours lag features
    'target_horizon': 1,  # Hours prediction
}


ML_CONFIG = {
    'data_split': {
        'train': 0.6,
        'validation': 0.2,
        'test': 0.2,
        'seed': 42
    },
    'cross_validation': {
        'num_folds': 3,
        'seed': 42
    },
    'gradient_boosted_trees': {
        'maxDepth': [5, 7, 10],
        'maxIter': [50, 100],
        'stepSize': [0.1, 0.05]
    },
    'random_forest_regressor': {
        'numTrees': [50, 100, 150],
        'maxDepth': [5, 10, 15],
        'minInstancesPerNode': [1, 5]
    },
    'linear_regression': {
        'elasticNetParam': [0.0, 0.5, 1.0],
        'regParam': [0.01, 0.1]
    },
    'random_forest_classifier': {
        'numTrees': [50, 100, 150],
        'maxDepth': [5, 10, 15],
        'minInstancesPerNode': [1, 5]
    },
    'gradient_boosted_trees_classifier': {
        'maxDepth': [5, 7, 10],
        'maxIter': [50, 100],
        'stepSize': [0.1, 0.05]
    }
}