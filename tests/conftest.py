"""
conftest.py
===========
Shared fixtures for the weather-prediction-mlops test suite.

All Spark tests use master("local[1]") — no cluster needed in CI.
Mongo tests use mongomock to avoid a real connection.
"""

import ast
from pathlib import Path
from unittest.mock import MagicMock

import pytest

#: Repository root, for guardrail tests that read the declared schema/constants
#: from source instead of importing modules that need FastAPI/Motor/Spark.
REPO_ROOT = Path(__file__).resolve().parents[1]


def _eval_constant(node: ast.AST):
    """Evaluate a literal/arithmetic AST node (Constants, containers, ``*``/``+``/``-``)."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Tuple):
        return tuple(_eval_constant(element) for element in node.elts)
    if isinstance(node, ast.List):
        return [_eval_constant(element) for element in node.elts]
    if isinstance(node, ast.Dict):
        return {
            _eval_constant(key): _eval_constant(value) for key, value in zip(node.keys, node.values)
        }
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _eval_constant(node.operand)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Add, ast.Sub)):
        left = _eval_constant(node.left)
        right = _eval_constant(node.right)
        if isinstance(node.op, ast.Mult):
            return left * right
        return left + right if isinstance(node.op, ast.Add) else left - right
    raise ValueError(f"unsupported constant node: {type(node).__name__}")


def eval_python_constant(source_path, name: str):
    """Read a module-level constant from source, without importing the module.

    The guardrail tests need the *declared* TTLs and indexes from
    ``backend/app/db/mongo.py``, but that module imports FastAPI and Motor, which
    the ``pipeline-test`` environment does not install. Parsing the source keeps
    the tests deriving the values instead of restating them.
    """
    tree = ast.parse(Path(source_path).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return _eval_constant(node.value)
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            if isinstance(target, ast.Name) and target.id == name:
                return _eval_constant(node.value)
    raise KeyError(f"{name} not found in {source_path}")


# ---------------------------------------------------------------------------
# Spark
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def spark():
    """Local SparkSession for unit tests (no cluster, no MongoDB connector)."""
    from pyspark.sql import SparkSession

    session = (
        SparkSession.builder.master("local[1]")
        .appName("weather-unit-tests")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


# ---------------------------------------------------------------------------
# MongoDB mock
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_mongo_client():
    """Return a MagicMock that mimics MongoClient well enough for unit tests."""
    client = MagicMock()
    db = MagicMock()
    client.__getitem__ = MagicMock(return_value=db)
    return client, db


# ---------------------------------------------------------------------------
# Sample document factories
# ---------------------------------------------------------------------------


def make_open_meteo_message(city: str = "Madrid", unix_ts: int = 1_750_000_000) -> dict:
    """Minimal Open-Meteo-shaped Kafka message value."""
    return {
        "city": city,
        "latitude": 40.4168,
        "longitude": -3.7038,
        "utc_offset_seconds": 3600,
        "timezone": "Europe/Madrid",
        "current": {
            "time": unix_ts,
            "temperature_2m": 22.5,
            "apparent_temperature": 21.0,
            "relative_humidity_2m": 55,
            "surface_pressure": 1013.0,
            "pressure_msl": 1015.0,
            "dew_point_2m": 13.5,
            "wind_speed_10m": 5.2,
            "wind_direction_10m": 270,
            "wind_gusts_10m": 8.1,
            "wind_speed_80m": 7.3,
            "wind_direction_80m": 265,
            "cloud_cover": 20,
            "precipitation": 0.0,
            "rain": 0.0,
            "showers": 0.0,
            "snowfall": 0.0,
            "visibility": 24140.0,
            "shortwave_radiation": 350.0,
            "cape": 0.0,
            "weather_code": 1,
            "uv_index": 4.0,
            "is_day": 1,
        },
    }
