"""
batch_processing.py
====================
Spark feature-engineering job.

Reads from raw_weather (both live consumer records and historical backfill),
builds atmospheric + time-series features, and writes the result
*incrementally* (append mode) to weather_features.

Key changes vs the previous version
-------------------------------------
- Source collection: raw_weather  (was weather_data)
- Save mode: append               (was overwrite — data was lost on every run)
- Atmospheric pressure-level features extracted from upper-air payload
- Specific humidity computed from relative humidity, temperature, and pressure
- 6-hour rolling precipitation sum
"""

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType
import sys
import os

sys.path.append("/opt/config")
from spark_config import create_spark_session, FEATURES_CONFIG

# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract_weather_data(spark):
    """
    Read from raw_weather and flatten into a tidy per-row DataFrame.

    The raw_weather documents written by the consumer / backfill script
    follow two slightly different shapes (live vs historical), but both
    expose the relevant fields under payload.current.
    """
    mongo_url = os.getenv("MONGO_URL")

    df = (
        spark.read.format("mongodb")
        .option("connection.uri", mongo_url)
        .option("database", "weather_db")
        .option("collection", "raw_weather")
        .load()
    )

    df_flat = df.select(
        F.col("city"),
        F.col("timestamp"),
        # current / hourly-point scalars
        F.col("payload.current.temperature_2m").alias("temperature"),
        F.col("payload.current.apparent_temperature").alias("feels_like"),
        F.col("payload.current.relative_humidity_2m").alias("humidity"),
        F.col("payload.current.surface_pressure").alias("pressure"),
        F.col("payload.current.pressure_msl").alias("pressure_msl"),
        F.col("payload.current.dew_point_2m").alias("dew_point"),
        F.col("payload.current.wind_speed_10m").alias("wind_speed"),
        F.col("payload.current.wind_direction_10m").alias("wind_direction"),
        F.col("payload.current.wind_gusts_10m").alias("wind_gusts"),
        F.col("payload.current.wind_speed_80m").alias("wind_speed_80m"),
        F.col("payload.current.wind_direction_80m").alias("wind_direction_80m"),
        F.col("payload.current.cloud_cover").alias("cloud_cover"),
        F.col("payload.current.precipitation").alias("precipitation"),
        F.col("payload.current.rain").alias("rain"),
        F.col("payload.current.snowfall").alias("snowfall"),
        F.col("payload.current.visibility").alias("visibility"),
        F.col("payload.current.shortwave_radiation").alias("shortwave_radiation"),
        F.col("payload.current.cape").alias("cape"),
        F.col("payload.current.weather_code").alias("weather_code"),
        F.col("payload.latitude").alias("latitude"),
        F.col("payload.longitude").alias("longitude"),
    )

    df_flat = df_flat.withColumn("timestamp", F.to_timestamp("timestamp"))

    return df_flat


# ---------------------------------------------------------------------------
# Derived atmospheric features
# ---------------------------------------------------------------------------


def add_atmospheric_features(df):
    """
    Add physically-derived atmospheric features:
      - specific_humidity  : converted from relative humidity, temperature, pressure
      - wind_u / wind_v    : Cartesian wind components (useful for GNN later)
    """
    import math

    # Specific humidity (Bolton 1980 approximation):
    #   e_s = 6.112 * exp(17.67 * T / (T + 243.5))   [hPa]
    #   e   = (RH/100) * e_s
    #   q   = 0.622 * e / (p - 0.378 * e)             [kg/kg]
    T = F.col("temperature")
    RH = F.col("humidity")
    p = F.col("pressure")  # surface pressure in hPa

    e_s = 6.112 * F.exp(17.67 * T / (T + 243.5))
    e = (RH / 100.0) * e_s
    q = 0.622 * e / (p - 0.378 * e)

    df = df.withColumn("specific_humidity", q.cast(DoubleType()))

    # Wind components (wind_direction is meteorological: 0°=N, 90°=E)
    #   u = -|speed| * sin(dir_rad)
    #   v = -|speed| * cos(dir_rad)
    dir_rad = F.col("wind_direction") * (math.pi / 180.0)
    df = df.withColumn(
        "wind_u", (-F.col("wind_speed") * F.sin(dir_rad)).cast(DoubleType())
    )
    df = df.withColumn(
        "wind_v", (-F.col("wind_speed") * F.cos(dir_rad)).cast(DoubleType())
    )

    return df


# ---------------------------------------------------------------------------
# Time features
# ---------------------------------------------------------------------------


def create_time_features(df):
    df = df.withColumn("hour", F.hour("timestamp"))
    df = df.withColumn("day_of_week", F.dayofweek("timestamp"))
    df = df.withColumn("day_of_month", F.dayofmonth("timestamp"))
    df = df.withColumn("month", F.month("timestamp"))
    df = df.withColumn("quarter", F.quarter("timestamp"))

    # Cyclical encoding
    df = df.withColumn("hour_sin", F.sin(F.col("hour") * 2 * 3.14159 / 24))
    df = df.withColumn("hour_cos", F.cos(F.col("hour") * 2 * 3.14159 / 24))
    df = df.withColumn("month_sin", F.sin(F.col("month") * 2 * 3.14159 / 12))
    df = df.withColumn("month_cos", F.cos(F.col("month") * 2 * 3.14159 / 12))

    return df


# ---------------------------------------------------------------------------
# Lag features
# ---------------------------------------------------------------------------


def create_lag_features(df, lag_periods):
    window_spec = Window.partitionBy("city").orderBy("timestamp")

    lag_columns = [
        "temperature",
        "humidity",
        "pressure",
        "wind_speed",
        "specific_humidity",
        "cape",
        "precipitation",
    ]

    for col_name in lag_columns:
        if col_name not in df.columns:
            continue
        for lag in lag_periods:
            df = df.withColumn(
                f"{col_name}_lag_{lag}h", F.lag(col_name, lag).over(window_spec)
            )

    return df


# ---------------------------------------------------------------------------
# Rolling / window features
# ---------------------------------------------------------------------------


def create_rolling_features(df, window_sizes):
    rolling_columns = [
        "temperature",
        "humidity",
        "pressure",
        "wind_speed",
        "specific_humidity",
        "shortwave_radiation",
    ]

    for hours in window_sizes:
        window_spec = (
            Window.partitionBy("city")
            .orderBy(F.col("timestamp").cast("long"))
            .rangeBetween(-hours * 3600, 0)
        )

        for col_name in rolling_columns:
            if col_name not in df.columns:
                continue
            df = df.withColumn(
                f"{col_name}_mean_{hours}h", F.mean(col_name).over(window_spec)
            )
            df = df.withColumn(
                f"{col_name}_std_{hours}h", F.stddev(col_name).over(window_spec)
            )
            df = df.withColumn(
                f"{col_name}_min_{hours}h", F.min(col_name).over(window_spec)
            )
            df = df.withColumn(
                f"{col_name}_max_{hours}h", F.max(col_name).over(window_spec)
            )

    return df


def create_precipitation_rolling_sum(df):
    """6-hour accumulated precipitation (useful rain-risk feature)."""
    window_spec = (
        Window.partitionBy("city")
        .orderBy(F.col("timestamp").cast("long"))
        .rangeBetween(-6 * 3600, 0)
    )
    df = df.withColumn("precip_sum_6h", F.sum("precipitation").over(window_spec))
    return df


# ---------------------------------------------------------------------------
# Rate-of-change features
# ---------------------------------------------------------------------------


def create_rate_of_change_features(df):
    window_spec = Window.partitionBy("city").orderBy("timestamp")

    df = df.withColumn(
        "temp_change_1h",
        F.col("temperature") - F.lag("temperature", 1).over(window_spec),
    )
    df = df.withColumn(
        "pressure_change_1h", F.col("pressure") - F.lag("pressure", 1).over(window_spec)
    )
    df = df.withColumn(
        "humidity_change_1h", F.col("humidity") - F.lag("humidity", 1).over(window_spec)
    )

    return df


# ---------------------------------------------------------------------------
# Target variables
# ---------------------------------------------------------------------------


def create_target_variable(df, horizon=1):
    window_spec = Window.partitionBy("city").orderBy("timestamp")

    df = df.withColumn(
        f"target_temp_{horizon}h", F.lead("temperature", horizon).over(window_spec)
    )
    df = df.withColumn(
        f"target_will_rain_{horizon}h",
        F.when(F.lead("rain", horizon).over(window_spec) > 0, 1).otherwise(0),
    )

    return df


# ---------------------------------------------------------------------------
# Incremental save
# ---------------------------------------------------------------------------


def save_features_to_mongodb(df, collection_name="weather_features", horizon=1):
    """
    Save the feature DataFrame to MongoDB in *append* mode so that previously
    processed rows are not overwritten.

    The job is designed to be run periodically (every 6 hours by the scheduler).
    Deduplication is left to downstream consumers / the training job which
    should select the latest record per city+hour when needed.
    """
    mongo_url = os.getenv("MONGO_URL")

    print(f"BEFORE dropna: {df.count()} rows")
    df_clean = df.dropna(
        subset=[
            "temperature",
            "humidity",
            "pressure",
            "wind_speed",
            f"target_temp_{horizon}h",
        ]
    )
    print(f"AFTER dropna (essential cols): {df_clean.count()} rows")

    df_clean.write.format("mongodb").option("connection.uri", mongo_url).option(
        "database", "weather_db"
    ).option("collection", collection_name).mode("append").save()

    return df_clean


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    spark = create_spark_session("WeatherFeatureEngineering")

    try:
        df = extract_weather_data(spark)

        df = add_atmospheric_features(df)
        df = create_time_features(df)
        df = create_lag_features(df, FEATURES_CONFIG["lag_periods"])
        df = create_rolling_features(df, FEATURES_CONFIG["window_sizes"])
        df = create_precipitation_rolling_sum(df)
        df = create_rate_of_change_features(df)
        df = create_target_variable(df, FEATURES_CONFIG["target_horizon"])

        horizon = FEATURES_CONFIG["target_horizon"]

        df.select(
            "city",
            "timestamp",
            "temperature",
            "specific_humidity",
            "cape",
            "temp_change_1h",
            "temperature_mean_6h",
            "precip_sum_6h",
            f"target_temp_{horizon}h",
        ).show(10, truncate=False)

        df_final = save_features_to_mongodb(df, horizon=horizon)

        df_final.select(
            F.mean("temperature").alias("avg_temp"),
            F.stddev("temperature").alias("std_temp"),
            F.mean("humidity").alias("avg_humidity"),
            F.mean("wind_speed").alias("avg_wind_speed"),
            F.mean("specific_humidity").alias("avg_specific_humidity"),
        ).show()

    except Exception as e:
        print(f"Error during feature engineering: {e}")
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
