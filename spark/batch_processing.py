from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import *
from datetime import datetime, timedelta
import sys
sys.path.append('/opt/spark-jobs')
from config.spark_config import create_spark_session, FEATURES_CONFIG

def extract_weather_data(spark):

    df = spark.read \
        .format("mongodb") \
        .option("collection", "weather_data") \
        .load()

    df_flat = df.select(
        F.col("city"),
        F.col("timestamp"),
        F.col("data.main.temp").alias("temperature"),
        F.col("data.main.feels_like").alias("feels_like"),
        F.col("data.main.humidity").alias("humidity"),
        F.col("data.main.pressure").alias("pressure"),
        F.col("data.wind.speed").alias("wind_speed"),
        F.col("data.wind.deg").alias("wind_direction"),
        F.col("data.clouds.all").alias("cloud_coverage"),
        F.col("data.weather")[0]["main"].alias("weather_main"),
        F.col("data.weather")[0]["description"].alias("weather_description")
    )
    
    df_flat = df_flat.withColumn("timestamp", F.to_timestamp("timestamp"))
    
    return df_flat

def create_time_features(df):

    df = df.withColumn("hour", F.hour("timestamp"))
    df = df.withColumn("day_of_week", F.dayofweek("timestamp"))
    df = df.withColumn("day_of_month", F.dayofmonth("timestamp"))
    df = df.withColumn("month", F.month("timestamp"))
    df = df.withColumn("quarter", F.quarter("timestamp"))
    
    # Cyclical features for hour
    df = df.withColumn("hour_sin", F.sin(F.col("hour") * 2 * 3.14159 / 24))
    df = df.withColumn("hour_cos", F.cos(F.col("hour") * 2 * 3.14159 / 24))
    
    # Cyclical features for month
    df = df.withColumn("month_sin", F.sin(F.col("month") * 2 * 3.14159 / 12))
    df = df.withColumn("month_cos", F.cos(F.col("month") * 2 * 3.14159 / 12))
    
    return df

def create_lag_features(df, lag_periods):

    window_spec = Window.partitionBy("city").orderBy("timestamp")

    lag_columns = ["temperature", "humidity", "pressure", "wind_speed"]
    
    for col_name in lag_columns:
        for lag in lag_periods:
            df = df.withColumn(
                f"{col_name}_lag_{lag}h",
                F.lag(col_name, lag).over(window_spec)
            )
    
    return df

def create_rolling_features(df, window_sizes):
    
    rolling_columns = ["temperature", "humidity", "pressure", "wind_speed"]
    
    for hours in window_sizes:
        window_spec = Window.partitionBy("city") \
                           .orderBy(F.col("timestamp").cast("long")) \
                           .rangeBetween(-hours * 3600, 0)
        
        for col_name in rolling_columns:
            df = df.withColumn(
                f"{col_name}_mean_{hours}h",
                F.mean(col_name).over(window_spec)
            )

            df = df.withColumn(
                f"{col_name}_std_{hours}h",
                F.stddev(col_name).over(window_spec)
            )

            df = df.withColumn(
                f"{col_name}_min_{hours}h",
                F.min(col_name).over(window_spec)
            )
            df = df.withColumn(
                f"{col_name}_max_{hours}h",
                F.max(col_name).over(window_spec)
            )
    
    return df

def create_rate_of_change_features(df):

    window_spec = Window.partitionBy("city").orderBy("timestamp")

    df = df.withColumn("temp_change_1h", 
                      F.col("temperature") - F.lag("temperature", 1).over(window_spec))
    df = df.withColumn("pressure_change_1h",
                      F.col("pressure") - F.lag("pressure", 1).over(window_spec))
    df = df.withColumn("humidity_change_1h",
                      F.col("humidity") - F.lag("humidity", 1).over(window_spec))
    
    return df

def create_target_variable(df, horizon=1):

    window_spec = Window.partitionBy("city").orderBy("timestamp")

    df = df.withColumn(
        f"target_temp_{horizon}h",
        F.lead("temperature", horizon).over(window_spec)
    )

    df = df.withColumn(
        f"target_will_rain_{horizon}h",
        F.when(F.lead("weather_main", horizon).over(window_spec) == "Rain", 1).otherwise(0)
    )
    
    return df

def save_features_to_mongodb(df, collection_name="weather_features", horizon=1):

    df_clean = df.dropna(subset=[f"target_temp_{horizon}h"])

    df_clean.write \
        .format("mongodb") \
        .option("collection", collection_name) \
        .mode("overwrite") \
        .save()

    return df_clean

def main():

    spark = create_spark_session("WeatherFeatureEngineering")

    try:
        df = extract_weather_data(spark)

        df = create_time_features(df)
        df = create_lag_features(df, FEATURES_CONFIG['lag_periods'])
        df = create_rolling_features(df, FEATURES_CONFIG['window_sizes'])
        df = create_rate_of_change_features(df)
        df = create_target_variable(df, FEATURES_CONFIG['target_horizon'])

        horizon = FEATURES_CONFIG['target_horizon']

        df.select("city", "timestamp", "temperature",
                 "temp_change_1h", "temperature_mean_6h",
                 f"target_temp_{horizon}h").show(10, truncate=False)

        df_final = save_features_to_mongodb(df, horizon=horizon)

        df_final.select(
            F.mean("temperature").alias("avg_temp"),
            F.stddev("temperature").alias("std_temp"),
            F.mean("humidity").alias("avg_humidity"),
            F.mean("wind_speed").alias("avg_wind_speed")
        ).show()

    except Exception as e:
        print(f"Error during feature engineering: {e}")
        raise
    finally:
        spark.stop()

if __name__ == "__main__":
    main()