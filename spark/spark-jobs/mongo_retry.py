"""Transient-Mongo retry helper for the Spark jobs.

Atlas M0 (shared tier) intermittently fails with transient errors that kill the
Spark jobs because they run in local mode (``spark/config/spark_config.py``) and
abort the whole stage on the first task failure.  Those failures reach Python as
``pyspark.errors.Py4JJavaError`` whose message embeds the Java exception class
name + text (e.g. ``com.mongodb.MongoSocketReadException: Prematurely reached
end of stream``), so classification must fall back to matching names/messages
inside ``str(exc)`` — ``isinstance`` checks on pymongo classes are not enough.

Pure stdlib + typing only: no pyspark, no pymongo, no gridfs, no network.
"""

import logging
import time
from typing import Any, Callable, Iterator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Classification data
# ---------------------------------------------------------------------------

# Retryable error codes, from the MongoDB retryable-writes spec as implemented
# in pymongo 4.17.0 (``pymongo/helpers_shared.py::_RETRYABLE_ERROR_CODES``,
# which is the union of the retryable-writes codes and the SDAM "not primary" /
# "node recovering" codes), plus the pre-3.6 legacy InterruptedAtShutdown code
# (116) which we observed in our Atlas logs alongside 91 (ShutdownInProgress).
TRANSIENT_MONGO_ERROR_CODES: frozenset[int] = frozenset(
    {
        6,  # HostUnreachable
        7,  # HostNotFound
        89,  # NetworkTimeout
        262,  # ExceededTimeLimit (retryable-writes spec)
        9001,  # SocketException
        134,  # ReadConcernMajorityNotAvailableYet (retryable-writes spec)
        # SDAM "node recovering / not primary" codes.
        91,  # ShutdownInProgress (verified in our Atlas logs)
        189,  # PrimarySteppedDown
        10058,  # LegacyNotPrimary (server <=3.2)
        10107,  # NotWritablePrimary
        11600,  # InterruptedAtShutdown (server >=3.6)
        11602,  # InterruptedDueToReplStateChange
        13435,  # NotPrimaryNoSecondaryOk
        13436,  # NotPrimaryOrSecondary
        # Legacy pre-3.6 InterruptedAtShutdown code, verified in our logs.
        116,  # InterruptedAtShutdown (legacy)
    }
)

# Exception class names matched exactly against ``type(exc).__name__`` and as
# substrings of ``str(exc)`` (for Java driver names embedded in Py4JJavaError).
# Java driver names are the mongodb-driver-sync classes seen in our logs; the
# pymongo names are the real classes in pymongo 4.17.0 (``pymongo.errors``).
# Pruned: pymongo ``WaitQueueTimeoutError`` and ``ExecutionTimeout`` — pymongo
# itself treats the former as non-retryable (pool exhaustion = overload) and the
# latter is MaxTimeMSExceeded (a query problem, not a transient fault).
TRANSIENT_MONGO_EXCEPTION_NAMES: frozenset[str] = frozenset(
    {
        # Java driver (mongodb-driver-sync) exceptions, wrapped in Py4JJavaError.
        "MongoSocketReadException",
        "MongoSocketWriteException",
        "MongoSocketException",
        "MongoSocketClosedException",
        "MongoSocketReadTimeoutException",
        "MongoNetworkException",
        "MongoNetworkError",  # legacy 2.x driver name
        "MongoNotPrimaryException",
        "MongoNodeIsRecoveringException",
        "MongoConnectionPoolClearedException",  # legacy driver class
        # pymongo 4.17.0 error classes (pymongo.errors).
        "AutoReconnect",
        "NetworkTimeout",
        "NotPrimaryError",
        "ServerSelectionTimeoutError",
        "ConnectionFailure",
    }
)

# Message fragments matched as substrings of ``str(exc)`` (and the cause chain).
TRANSIENT_MONGO_MESSAGE_FRAGMENTS: tuple[str, ...] = (
    "Prematurely reached end of stream",  # MongoSocketReadException detail (our logs)
    "quiesce mode",  # ShutdownInProgress server text
    "ShutdownInProgress",  # error 91 name
    "connection pool cleared",  # MongoConnectionPoolClearedException text
    "connection reset",  # connection reset by peer
    "No server chosen",  # ServerSelectionTimeoutError detail
    "not primary",  # stepdown / stale primary
    "SocketException",  # Java socket exception name
)

# Retry loop defaults.
MONGO_RETRY_ATTEMPTS: int = 4
MONGO_RETRY_BASE_SECONDS: float = 5.0
MONGO_RETRY_FACTOR: float = 2.0
MONGO_RETRY_MAX_SECONDS: float = 60.0

#: How many levels of ``__cause__`` / ``__context__`` to inspect (guard vs cycles).
_MAX_CAUSE_DEPTH = 3


def _error_code(exc: Any) -> int | None:
    """Return ``exc.code`` when it is an int, else None — never raises on odd objects."""
    code = getattr(exc, "code", None)
    return code if isinstance(code, int) else None


def _contains_any(text: str, needles: frozenset[str] | tuple[str, ...]) -> bool:
    """True if ``text`` contains any of ``needles`` as a substring."""
    return any(needle in text for needle in needles)


def _iter_exception_chain(
    exc: object, _depth: int = 0, _seen: set[int] | None = None
) -> Iterator[object]:
    """Yield ``exc`` then its ``__cause__``/``__context__`` chain, cycle- and depth-guarded."""
    if exc is None or _depth > _MAX_CAUSE_DEPTH:
        return
    if _seen is None:
        _seen = set()
    ident = id(exc)
    if ident in _seen:
        return
    _seen.add(ident)
    yield exc
    for attr in ("__cause__", "__context__"):
        nxt = getattr(exc, attr, None)
        if nxt is not None and nxt is not exc:
            yield from _iter_exception_chain(nxt, _depth + 1, _seen)


def is_transient_mongo_error(exc: object) -> bool:
    """True when ``exc`` (or its cause chain) denotes a transient Mongo fault.

    Matches: (1) the exception type name against the transient name set,
    (2) the Mongo error code against the transient code set, or (3) the string
    form of the exception (and its ``__cause__``/``__context__`` chain) against
    the transient names/message fragments.  Anything else returns False so real
    bugs (``ValueError``, duplicate-key 11000, schema-mismatch Py4JJavaError)
    fail fast instead of being retried.
    """
    for node in _iter_exception_chain(exc):
        if type(node).__name__ in TRANSIENT_MONGO_EXCEPTION_NAMES:
            return True
        if _error_code(node) in TRANSIENT_MONGO_ERROR_CODES:
            return True
        # ``str(node)`` can itself raise: a real Py4JJavaError.__str__ performs a
        # JVM round-trip and raises Py4JNetworkError when the gateway is gone.
        # Classification must be total (never raise, for any input): an
        # unstringifiable node simply contributes no text match and the walk
        # continues to the next node in the chain.
        try:
            text = str(node)
        except Exception:
            text = ""
        if _contains_any(text, TRANSIENT_MONGO_EXCEPTION_NAMES):
            return True
        if _contains_any(text, TRANSIENT_MONGO_MESSAGE_FRAGMENTS):
            return True
    return False


def retry_delay_seconds(
    attempt: int,
    base: float = MONGO_RETRY_BASE_SECONDS,
    factor: float = MONGO_RETRY_FACTOR,
    cap: float = MONGO_RETRY_MAX_SECONDS,
) -> float:
    """Exponential backoff: attempt 1 -> ``base``, 2 -> ``base*factor``, capped at ``cap``.

    Pure.  Non-positive ``attempt`` is treated as 1.
    """
    if attempt < 1:
        attempt = 1
    return min(base * (factor ** (attempt - 1)), cap)


def run_with_mongo_retries(
    fn: Callable[[], Any],
    *,
    label: str = "mongo operation",
    attempts: int = MONGO_RETRY_ATTEMPTS,
    sleep: Callable[[float], Any] = time.sleep,
) -> Any:
    """Call ``fn()`` and return its value, retrying transient Mongo errors.

    Non-transient exceptions re-raise immediately (no sleep).  Transient
    exceptions retry with exponential backoff until ``attempts`` is exhausted,
    then re-raise the last exception unchanged.  ``sleep`` is injectable for
    tests.
    """
    if attempts < 1:
        raise ValueError(f"attempts must be >= 1, got {attempts}")
    for attempt in range(1, attempts + 1):
        try:
            result = fn()
        except Exception as exc:
            if not is_transient_mongo_error(exc):
                raise
            attempts_left = attempts - attempt
            if attempts_left <= 0:
                logger.error(
                    "%s failed on attempt %d/%d (%s: %s); giving up",
                    label,
                    attempt,
                    attempts,
                    type(exc).__name__,
                    exc,
                )
                raise
            delay = retry_delay_seconds(attempt)
            logger.warning(
                "%s failed on attempt %d/%d (%s: %s); %d attempt(s) left, retrying in %.1fs",
                label,
                attempt,
                attempts,
                type(exc).__name__,
                exc,
                attempts_left,
                delay,
            )
            sleep(delay)
            continue
        if attempt > 1:
            logger.info("%s succeeded on attempt %d", label, attempt)
        return result
    raise AssertionError("unreachable")  # pragma: no cover
