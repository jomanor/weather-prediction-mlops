from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import GBTRegressor, LinearRegression, RandomForestRegressor
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import RegressionEvaluator, BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import sys
sys.path.append('/opt/config')
from spark_config import create_spark_session, ML_CONFIG, FEATURES_CONFIG
from datetime import datetime
import os
import json
from pymongo import MongoClient
from gridfs import GridFS
import shutil
import mlflow
import mlflow.spark

def load_features(spark):

    mongo_url = os.getenv("MONGO_URL")

    df = spark.read \
        .format("mongodb") \
        .option("connection.uri", mongo_url) \
        .option("database", "weather_db") \
        .option("collection", "weather_features") \
        .load()
    
    # TODO this should be done in other part
    df = df.drop("_id")
    df_clean = df.dropna()

    return df_clean

def prepare_features_for_ml(df, target_col, horizon=1):

    target_temp_col = f"target_temp_{horizon}h"
    target_rain_col = f"target_will_rain_{horizon}h"

    exclude_cols = ["city", "timestamp", "weather_main", "weather_description",
                   target_col, target_temp_col, target_rain_col]

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features_raw"
    )

    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features"
    )

    return assembler, scaler, feature_cols

def train_temperature_prediction_model(df, horizon=1):

    target_col = f"target_temp_{horizon}h"

    assembler, scaler, feature_cols = prepare_features_for_ml(df, target_col, horizon)

    train_ratio = ML_CONFIG['data_split']['train']
    val_ratio = ML_CONFIG['data_split']['validation']
    test_ratio = ML_CONFIG['data_split']['test']
    seed = ML_CONFIG['data_split']['seed']

    train_df, val_df, test_df = df.randomSplit([train_ratio, val_ratio, test_ratio], seed=seed)

    print(f"Train size: {train_df.count()}, Validation size: {val_df.count()}, Test size: {test_df.count()}")

    models = {
        "GradientBoostedTrees": GBTRegressor(
            featuresCol="features",
            labelCol=target_col
        ),
        "RandomForest": RandomForestRegressor(
            featuresCol="features",
            labelCol=target_col
        ),
        "LinearRegression": LinearRegression(
            featuresCol="features",
            labelCol=target_col
        )
    }

    param_grids = {
        "GradientBoostedTrees": ParamGridBuilder() \
            .addGrid(models["GradientBoostedTrees"].maxDepth, ML_CONFIG['gradient_boosted_trees']['maxDepth']) \
            .addGrid(models["GradientBoostedTrees"].maxIter, ML_CONFIG['gradient_boosted_trees']['maxIter']) \
            .addGrid(models["GradientBoostedTrees"].stepSize, ML_CONFIG['gradient_boosted_trees']['stepSize']) \
            .build(),
        "RandomForest": ParamGridBuilder() \
            .addGrid(models["RandomForest"].numTrees, ML_CONFIG['random_forest_regressor']['numTrees']) \
            .addGrid(models["RandomForest"].maxDepth, ML_CONFIG['random_forest_regressor']['maxDepth']) \
            .addGrid(models["RandomForest"].minInstancesPerNode, ML_CONFIG['random_forest_regressor']['minInstancesPerNode']) \
            .build(),
        "LinearRegression": ParamGridBuilder() \
            .addGrid(models["LinearRegression"].elasticNetParam, ML_CONFIG['linear_regression']['elasticNetParam']) \
            .addGrid(models["LinearRegression"].regParam, ML_CONFIG['linear_regression']['regParam']) \
            .build()
    }

    evaluator = RegressionEvaluator(
        labelCol=target_col,
        predictionCol="prediction",
        metricName="rmse"
    )
    mae_evaluator = RegressionEvaluator(
        labelCol=target_col,
        predictionCol="prediction",
        metricName="mae"
    )
    r2_evaluator = RegressionEvaluator(
        labelCol=target_col,
        predictionCol="prediction",
        metricName="r2"
    )

    best_model = None
    best_rmse = float('inf')
    best_model_name = None

    for model_name, model in models.items():

        pipeline = Pipeline(stages=[assembler, scaler, model])

        cv = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=param_grids[model_name],
            evaluator=evaluator,
            numFolds=ML_CONFIG['cross_validation']['num_folds'],
            seed=ML_CONFIG['cross_validation']['seed']
        )

        cv_model = cv.fit(train_df)

        best_pipeline = cv_model.bestModel

        val_predictions = best_pipeline.transform(val_df)
        val_rmse = evaluator.evaluate(val_predictions)

        print(f"\n{model_name} Validation RMSE: {val_rmse:.4f}")

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_model = best_pipeline
            best_model_name = model_name

    print(f"\n=== BEST MODEL: {best_model_name} ===")
    print(f"Validation RMSE: {best_rmse:.4f}")

    test_predictions = best_model.transform(test_df)
    test_rmse = evaluator.evaluate(test_predictions)
    test_mae = mae_evaluator.evaluate(test_predictions)
    test_r2 = r2_evaluator.evaluate(test_predictions)

    print(f"\n=== FINAL TEST SET PERFORMANCE FOR TEMPERATURE MODEL ===")
    print(f"\nTest RMSE: {test_rmse:.4f}")
    print(f"Test MAE : {test_mae:.4f}")
    print(f"Test R²  : {test_r2:.4f}")

    test_predictions.select(
        "city", "temperature", target_col, "prediction",
        F.abs(F.col(target_col) - F.col("prediction")).alias("error")
    ).show(5)

    if best_model_name in ["GradientBoostedTrees", "RandomForest"]:
        model_stage = best_model.stages[-1]
        feature_importance = model_stage.featureImportances

        important_features = sorted(
            [(feature_cols[i], float(importance))
             for i, importance in enumerate(feature_importance)],
            key=lambda x: x[1],
            reverse=True
        )[:10]

        print("\nTop 10 Important Features:")
        for feature, importance in important_features:
            print(f"  {feature}: {importance:.4f}")

    elif best_model_name == "LinearRegression":
        model_stage = best_model.stages[-1]
        coefficients = model_stage.coefficients.toArray()
        
        important_features = sorted(
            [(feature_cols[i], abs(float(coef)))
             for i, coef in enumerate(coefficients)],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        print("\nTop 10 Important Features (by absolute coefficient):")
        for feature, coef in important_features:
            print(f"  {feature}: {coef:.4f}")

    return best_model, best_model_name, important_features, {
        "rmse": test_rmse, "mae": test_mae, "r2": test_r2,
    }

def train_rain_prediction_model(df, horizon=1):

    target_col = f"target_will_rain_{horizon}h"

    assembler, scaler, feature_cols = prepare_features_for_ml(df, target_col, horizon)

    train_ratio = ML_CONFIG['data_split']['train']
    val_ratio = ML_CONFIG['data_split']['validation']
    test_ratio = ML_CONFIG['data_split']['test']
    seed = ML_CONFIG['data_split']['seed']

    train_df, val_df, test_df = df.randomSplit([train_ratio, val_ratio, test_ratio], seed=seed)

    print(f"Train size: {train_df.count()}, Validation size: {val_df.count()}, Test size: {test_df.count()}")

    models = {
        "GradientBoostedTrees": GBTClassifier(
            featuresCol="features",
            labelCol=target_col
        ),
        "RandomForest": RandomForestClassifier(
            featuresCol="features",
            labelCol=target_col
        )
    }

    param_grids = {
        "GradientBoostedTrees": ParamGridBuilder() \
            .addGrid(models["GradientBoostedTrees"].maxDepth, ML_CONFIG['gradient_boosted_trees_classifier']['maxDepth']) \
            .addGrid(models["GradientBoostedTrees"].maxIter, ML_CONFIG['gradient_boosted_trees_classifier']['maxIter']) \
            .addGrid(models["GradientBoostedTrees"].stepSize, ML_CONFIG['gradient_boosted_trees_classifier']['stepSize']) \
            .build(),
        "RandomForest": ParamGridBuilder() \
            .addGrid(models["RandomForest"].numTrees, ML_CONFIG['random_forest_classifier']['numTrees']) \
            .addGrid(models["RandomForest"].maxDepth, ML_CONFIG['random_forest_classifier']['maxDepth']) \
            .addGrid(models["RandomForest"].minInstancesPerNode, ML_CONFIG['random_forest_classifier']['minInstancesPerNode']) \
            .build()
    }

    evaluator = BinaryClassificationEvaluator(
        labelCol=target_col,
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )

    best_model = None
    best_auc = 0.0
    best_model_name = None

    for model_name, model in models.items():

        pipeline = Pipeline(stages=[assembler, scaler, model])

        cv = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=param_grids[model_name],
            evaluator=evaluator,
            numFolds=ML_CONFIG['cross_validation']['num_folds'],
            seed=ML_CONFIG['cross_validation']['seed']
        )

        cv_model = cv.fit(train_df)

        best_pipeline = cv_model.bestModel

        val_predictions = best_pipeline.transform(val_df)
        val_auc = evaluator.evaluate(val_predictions)

        if val_auc > best_auc:
            best_auc = val_auc
            best_model = best_pipeline
            best_model_name = model_name

    evaluator.setMetricName("areaUnderROC")
    test_predictions = best_model.transform(test_df)
    test_auc = evaluator.evaluate(test_predictions)
    evaluator.setMetricName("areaUnderPR")
    test_pr_auc = evaluator.evaluate(test_predictions)

    print(f"\n=== FINAL TEST SET PERFORMANCE FOR RAIN MODEL ===")
    print(f"\nTest AUC-ROC: {test_auc:.4f}")
    print(f"\nTest AUC-PR: {test_pr_auc:.4f}")

    print("\nConfusion matrix:")
    test_predictions.groupBy(target_col, "prediction").count().show()

    model_stage = best_model.stages[-1]
    feature_importance = model_stage.featureImportances

    important_features = sorted(
        [(feature_cols[i], float(importance))
         for i, importance in enumerate(feature_importance)],
        key=lambda x: x[1],
        reverse=True
    )[:10]

    return best_model, best_model_name, important_features, {
        "auc_roc": test_auc, "auc_pr": test_pr_auc,
    }

# ---------------------------------------------------------------------------
# MLflow logging helper
# ---------------------------------------------------------------------------

def _log_to_mlflow(
    run_name: str,
    model_type: str,
    model_name: str,
    params: dict,
    metrics: dict,
    important_features: list,
    model,
    horizon: int,
):
    """Log a single training run (temp or rain) to MLflow."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(f"weather-{model_type}-prediction")

    with mlflow.start_run(run_name=run_name):
        # --- params ---
        mlflow.log_param("model_algorithm", model_name)
        mlflow.log_param("horizon_hours", horizon)
        mlflow.log_param("train_ratio", ML_CONFIG['data_split']['train'])
        mlflow.log_param("cv_folds", ML_CONFIG['cross_validation']['num_folds'])
        for k, v in params.items():
            mlflow.log_param(k, v)

        # --- metrics ---
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # --- feature importance as JSON artifact ---
        fi_dict = {f: imp for f, imp in important_features}
        fi_path = f"/tmp/{run_name}_feature_importance.json"
        with open(fi_path, "w") as fh:
            json.dump(fi_dict, fh, indent=2)
        mlflow.log_artifact(fi_path, artifact_path="feature_importance")

        # --- log the Spark PipelineModel ---
        mlflow.spark.log_model(
            spark_model=model,
            artifact_path="model",
            registered_model_name=f"weather_{model_type}_{horizon}h",
        )

        print(f"MLflow run '{run_name}' logged successfully.")


def save_model(model, model_name, db_name="weather_db", metadata_collection="model_registry"):
    
    mongo_url = os.getenv("MONGO_URL")
    temp_dir = "/opt/spark-tmp"
    os.makedirs(temp_dir, exist_ok=True)
    
    model_dir_path = f"{temp_dir}/{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_zip_path = f"{model_dir_path}.zip"

    try:
        model.write().overwrite().save(model_dir_path)
        shutil.make_archive(base_name=model_dir_path, format='zip', root_dir=model_dir_path)

        client = MongoClient(mongo_url)
        db = client[db_name]
        fs = GridFS(db)

        with open(model_zip_path, 'rb') as model_zip_file:
            file_id = fs.put(model_zip_file, filename=model_zip_path.split('/')[-1])

        print(f"Successfully uploaded model zip to GridFS with file_id: {file_id}")

        metadata = {
            "model_name": model_name,
            "model_type": model.__class__.__name__,
            "gridfs_file_id": file_id,
            "timestamp": datetime.now(),
            "version": datetime.now().strftime('%Y%m%d_%H%M%S')
        }

        db[metadata_collection].insert_one(metadata)

    except Exception as e:
        print(f"Error during model and metadata save: {e}")
        raise
    
    finally:
        if os.path.exists(model_dir_path):
            print(f"Cleaning up temporary directory: {model_dir_path}")
            shutil.rmtree(model_dir_path)
        if os.path.exists(model_zip_path):
            print(f"Cleaning up temporary zip file: {model_zip_path}")
            os.remove(model_zip_path)
        if 'client' in locals():
            client.close()

def create_prediction_batch(spark, model, collection="weather_predictions"):

    mongo_url = os.getenv("MONGO_URL")

    df = spark.read \
        .format("mongodb") \
        .option("connection.uri", mongo_url) \
        .option("database", "weather_db") \
        .option("collection", "weather_features") \
        .load()

    latest_df = df.groupBy("city").agg(
        F.max("timestamp").alias("timestamp")
    ).join(df, ["city", "timestamp"], "inner")

    predictions = model.transform(latest_df)

    predictions_to_save = predictions.select(
        "city",
        "timestamp",
        F.current_timestamp().alias("prediction_timestamp"),
        "prediction",
        "temperature"
    )

    predictions_to_save.write \
        .format("mongodb") \
        .option("connection.uri", mongo_url) \
        .option("database", "weather_db") \
        .option("collection", collection) \
        .mode("append") \
        .save()

    predictions_to_save.show()

def main():

    spark = create_spark_session("WeatherMLTraining")

    try:
        df = load_features(spark)

        horizon = FEATURES_CONFIG['target_horizon']

        print(f"\n=== Training models with {horizon}h prediction horizon ===\n")

        temp_model, temp_model_name, temp_features, temp_metrics = train_temperature_prediction_model(df, horizon)
        rain_model, rain_model_name, rain_features, rain_metrics = train_rain_prediction_model(df, horizon)

        # --- persist to GridFS (used by inference.py) ---
        save_model(temp_model, f"temp_prediction_{horizon}h_{temp_model_name}")
        save_model(rain_model, f"rain_prediction_{horizon}h_{rain_model_name}")

        # --- log to MLflow ---
        _log_to_mlflow(
            run_name=f"temp_{temp_model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            model_type="temperature",
            model_name=temp_model_name,
            params={},
            metrics=temp_metrics,
            important_features=temp_features,
            model=temp_model,
            horizon=horizon,
        )
        _log_to_mlflow(
            run_name=f"rain_{rain_model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            model_type="rain",
            model_name=rain_model_name,
            params={},
            metrics=rain_metrics,
            important_features=rain_features,
            model=rain_model,
            horizon=horizon,
        )

        # TODO: Add also the rain model.
        create_prediction_batch(spark, temp_model)

    except Exception as e:
        print(f"Error during ML training: {e}")
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()