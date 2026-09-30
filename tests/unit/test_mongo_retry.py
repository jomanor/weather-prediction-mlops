"""
tests/unit/test_mongo_retry.py
===============================
Unit tests for spark/spark-jobs/mongo_retry.py

Pure classification + retry-loop logic only: no live Mongo, no Spark session.
"""

import logging
import os
import sys

import pytest
from pymongo.errors import (
    AutoReconnect,
    DuplicateKeyError,
    NotPrimaryError,
    OperationFailure,
)

# ---------------------------------------------------------------------------
# Path setup — mongo_retry.py is stdlib-only, so no spark_config stub is needed.
# ---------------------------------------------------------------------------
SPARK_JOBS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "spark", "spark-jobs")
sys.path.insert(0, SPARK_JOBS_DIR)

import mongo_retry as mr  # noqa: E402

# ---------------------------------------------------------------------------
# _error_code
# ---------------------------------------------------------------------------


def test_error_code_returns_int_code():
    class _FakeOpFailure(Exception):
        code = 91

    assert mr._error_code(_FakeOpFailure()) == 91


def test_error_code_none_for_missing_or_non_int():
    class _NoCode(Exception):
        pass

    class _StrCode(Exception):
        code = "91"

    assert mr._error_code(_NoCode()) is None
    assert mr._error_code(_StrCode()) is None
    assert mr._error_code(None) is None
    assert mr._error_code("boom") is None
    assert mr._error_code(123) is None


# ---------------------------------------------------------------------------
# is_transient_mongo_error — classification
# ---------------------------------------------------------------------------


def test_pymongo_autoreconnect_is_transient():
    assert mr.is_transient_mongo_error(AutoReconnect("connection closed")) is True


def test_pymongo_not_primary_error_is_transient():
    assert mr.is_transient_mongo_error(NotPrimaryError("not primary")) is True


def test_operation_failure_transient_codes_are_transient():
    assert mr.is_transient_mongo_error(OperationFailure("cmd failed", 91)) is True
    assert mr.is_transient_mongo_error(OperationFailure("interrupted", 116)) is True


def test_duplicate_key_and_code_11000_are_not_transient():
    assert mr.is_transient_mongo_error(OperationFailure("dup key", 11000)) is False
    assert mr.is_transient_mongo_error(DuplicateKeyError("E11000 dup", code=11000)) is False


def test_value_error_is_not_transient():
    assert mr.is_transient_mongo_error(ValueError("schema mismatch")) is False


def test_fake_class_named_transient_is_detected():
    MongoNodeIsRecoveringException = type("MongoNodeIsRecoveringException", (Exception,), {})
    assert mr.is_transient_mongo_error(MongoNodeIsRecoveringException("quiesce mode")) is True


def test_py4jjavaerror_transient_message_is_detected():
    Py4JJavaError = type("Py4JJavaError", (Exception,), {})
    transient = Py4JJavaError(
        "org.apache.spark.SparkException: com.mongodb.MongoSocketReadException: "
        "Prematurely reached end of stream"
    )
    assert mr.is_transient_mongo_error(transient) is True


def test_py4jjavaerror_schema_error_is_not_transient():
    Py4JJavaError = type("Py4JJavaError", (Exception,), {})
    schema_err = Py4JJavaError(
        "org.apache.spark.SparkException: AnalysisException: cannot resolve column 'city'"
    )
    assert mr.is_transient_mongo_error(schema_err) is False


def test_cause_chain_is_detected():
    cause = type("MongoSocketReadException", (Exception,), {})("Prematurely reached end of stream")
    outer = RuntimeError("wrapped failure")
    outer.__cause__ = cause
    assert mr.is_transient_mongo_error(outer) is True


def test_cause_cycle_does_not_hang():
    loop = RuntimeError("loop")
    loop.__cause__ = loop  # self-referential cause
    assert mr.is_transient_mongo_error(loop) is False


def test_non_exception_inputs_are_false_without_raising():
    assert mr.is_transient_mongo_error(None) is False
    assert mr.is_transient_mongo_error("just some text") is False
    assert mr.is_transient_mongo_error(12345) is False


# ---------------------------------------------------------------------------
# is_transient_mongo_error — unstringifiable nodes (classification is total)
# ---------------------------------------------------------------------------


def test_unstringifiable_node_returns_false_without_raising():
    class _BadStr(Exception):
        def __str__(self):
            raise RuntimeError("str exploded")

    assert mr.is_transient_mongo_error(_BadStr()) is False


def test_unstringifiable_node_wrapping_transient_cause_is_true():
    class _BadStr(Exception):
        def __str__(self):
            raise RuntimeError("str exploded")

    cause = type("MongoSocketReadException", (Exception,), {})("Prematurely reached end of stream")
    outer = _BadStr()
    outer.__cause__ = cause
    assert mr.is_transient_mongo_error(outer) is True


# ---------------------------------------------------------------------------
# is_transient_mongo_error — pinned classification entries
# ---------------------------------------------------------------------------


def test_mongo_connection_pool_cleared_name_is_transient():
    Py4JJavaError = type("Py4JJavaError", (Exception,), {})
    exc = Py4JJavaError(
        "org.apache.spark.SparkException: Job aborted due to stage failure: "
        "com.mongodb.MongoConnectionPoolClearedException: Connection pool cleared "
        "for ac-ikelm1f-shard-00-02"
    )
    assert mr.is_transient_mongo_error(exc) is True


def test_mongo_connection_pool_cleared_as_nested_cause_is_transient():
    Py4JJavaError = type("Py4JJavaError", (Exception,), {})
    cause = Py4JJavaError(
        "com.mongodb.MongoConnectionPoolClearedException: Connection pool cleared "
        "for ac-ikelm1f-shard-00-02"
    )
    outer = RuntimeError("stage failure")
    outer.__cause__ = cause
    assert mr.is_transient_mongo_error(outer) is True


def test_connection_pool_cleared_fragment_is_transient():
    Py4JJavaError = type("Py4JJavaError", (Exception,), {})
    exc = Py4JJavaError("driver reports connection pool cleared for this host")
    assert mr.is_transient_mongo_error(exc) is True


# ---------------------------------------------------------------------------
# is_transient_mongo_error — chain-walk semantics
#
# Depth convention pinned here: `_iter_exception_chain` yields the root node
# (depth 0) plus up to `_MAX_CAUSE_DEPTH` further causes/contexts, so the
# deepest reachable level is depth `_MAX_CAUSE_DEPTH` and depth
# `_MAX_CAUSE_DEPTH + 1` is out of range.
# ---------------------------------------------------------------------------


def test_implicit_context_chain_is_detected():
    Transient = type("MongoSocketReadException", (Exception,), {})

    def build():
        try:
            raise Transient("Prematurely reached end of stream")
        except Exception:
            raise RuntimeError("wrapper")  # implicit chaining -> __context__ only

    with pytest.raises(RuntimeError) as excinfo:
        build()

    outer = excinfo.value
    assert outer.__cause__ is None
    assert isinstance(outer.__context__, Transient)
    assert mr.is_transient_mongo_error(outer) is True


def test_max_cause_depth_reached_is_transient():
    transient = type("MongoSocketReadException", (Exception,), {})(
        "Prematurely reached end of stream"
    )
    # root(0) -> c1(1) -> c2(2) -> c3(3 = _MAX_CAUSE_DEPTH, transient)
    c2 = RuntimeError("c2")
    c2.__cause__ = transient
    c1 = RuntimeError("c1")
    c1.__cause__ = c2
    root = RuntimeError("root")
    root.__cause__ = c1
    assert mr.is_transient_mongo_error(root) is True


def test_max_cause_depth_exceeded_is_not_transient():
    transient = type("MongoSocketReadException", (Exception,), {})(
        "Prematurely reached end of stream"
    )
    # root(0) -> c1(1) -> c2(2) -> c3(3) -> c4(4 = _MAX_CAUSE_DEPTH + 1, transient)
    c3 = RuntimeError("c3")
    c3.__cause__ = transient
    c2 = RuntimeError("c2")
    c2.__cause__ = c3
    c1 = RuntimeError("c1")
    c1.__cause__ = c2
    root = RuntimeError("root")
    root.__cause__ = c1
    assert mr.is_transient_mongo_error(root) is False


# ---------------------------------------------------------------------------
# retry_delay_seconds
# ---------------------------------------------------------------------------


def test_retry_delay_exponential_backoff():
    assert mr.retry_delay_seconds(1) == 5.0
    assert mr.retry_delay_seconds(2) == 10.0
    assert mr.retry_delay_seconds(3) == 20.0
    assert mr.retry_delay_seconds(4) == 40.0
    assert mr.retry_delay_seconds(5) == 60.0  # capped
    assert mr.retry_delay_seconds(6) == 60.0  # capped


def test_retry_delay_nonpositive_attempt_is_treated_as_one():
    assert mr.retry_delay_seconds(0) == 5.0
    assert mr.retry_delay_seconds(-3) == 5.0


def test_retry_delay_custom_parameters():
    assert mr.retry_delay_seconds(3, base=1.0, factor=3.0, cap=100.0) == 9.0


# ---------------------------------------------------------------------------
# run_with_mongo_retries
# ---------------------------------------------------------------------------


def test_run_retries_then_succeeds(caplog):
    calls = {"n": 0}
    sleeps = []

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise AutoReconnect("connection closed")
        return "ok"

    with caplog.at_level(logging.INFO):
        result = mr.run_with_mongo_retries(flaky, label="read weather", sleep=sleeps.append)

    assert result == "ok"
    assert calls["n"] == 3
    assert sleeps == [5.0, 10.0]

    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2
    assert all("read weather" in w for w in warnings)
    assert "retrying in 5.0s" in warnings[0]
    assert "retrying in 10.0s" in warnings[1]

    infos = [r.message for r in caplog.records if r.levelno == logging.INFO]
    assert any("read weather succeeded on attempt 3" in m for m in infos)


def test_run_non_transient_raises_immediately(caplog):
    calls = {"n": 0}
    sleeps = []

    def boom():
        calls["n"] += 1
        raise ValueError("schema mismatch")

    with pytest.raises(ValueError, match="schema mismatch"):
        mr.run_with_mongo_retries(boom, sleep=sleeps.append)

    assert calls["n"] == 1
    assert sleeps == []


def test_run_exhausts_attempts_and_reraises_unchanged(caplog):
    calls = {"n": 0}
    sleeps = []

    def always_fail():
        calls["n"] += 1
        raise AutoReconnect("Prematurely reached end of stream")

    with caplog.at_level(logging.ERROR):
        with pytest.raises(AutoReconnect, match="Prematurely reached end of stream"):
            mr.run_with_mongo_retries(always_fail, attempts=2, sleep=sleeps.append)

    assert calls["n"] == 2
    assert sleeps == [5.0]
    errors = [r.message for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    assert "giving up" in errors[0]


def test_run_single_attempt_calls_once_without_sleep():
    calls = {"n": 0}
    sleeps = []

    def fail_once():
        calls["n"] += 1
        raise AutoReconnect("down")

    with pytest.raises(AutoReconnect):
        mr.run_with_mongo_retries(fail_once, attempts=1, sleep=sleeps.append)

    assert calls["n"] == 1
    assert sleeps == []


def test_run_attempts_zero_raises_value_error():
    with pytest.raises(ValueError, match="attempts"):
        mr.run_with_mongo_retries(lambda: None, attempts=0)


def test_run_default_attempts_is_four():
    calls = {"n": 0}
    sleeps = []

    def always_fail():
        calls["n"] += 1
        raise AutoReconnect("down")

    with pytest.raises(AutoReconnect):
        mr.run_with_mongo_retries(always_fail, sleep=sleeps.append)

    assert calls["n"] == 4
    assert sleeps == [5.0, 10.0, 20.0]


def test_run_succeeds_first_try_without_logging(caplog):
    calls = {"n": 0}

    def ok():
        calls["n"] += 1
        return "value"

    with caplog.at_level(logging.INFO):
        result = mr.run_with_mongo_retries(ok)

    assert result == "value"
    assert calls["n"] == 1
    assert caplog.records == []
