"""``app.core.thresholds``: one authority, plain constants, no imports.

The module is imported by the API, ``scripts/`` and the Spark jobs, so its
surface is frozen by name and it must stay import-free: a settings object or a
web framework in here would drag its dependencies into every consumer.
"""

import ast
from pathlib import Path

from app.core import thresholds
from app.core.config import Settings
from app.repositories import weather_repo
from app.routers import predictions as predictions_router

EXPECTED = {
    "QUALITY_OK_AGE_HOURS": 12.0,
    "QUALITY_WARN_AGE_HOURS": 18.0,
    "QUALITY_BAD_COMPLETENESS_BELOW": 0.80,
    "PREDICTION_STALE_AFTER_HOURS": 12.0,
    "PREDICTION_OK_AGE_HOURS": 12.0,
    "PREDICTION_WARN_AGE_HOURS": 18.0,
    "PREDICTION_COVERAGE_WARN_RATIO": 0.90,
    "SKILL_SCORE_ADVISORY_AT_OR_BELOW": 0.0,
}


def test_the_frozen_constants_and_values():
    for name, value in EXPECTED.items():
        assert getattr(thresholds, name) == value, name
    exported = {name for name in vars(thresholds) if name.isupper()}
    assert exported == set(EXPECTED)  # names are frozen: no extras, none missing


def test_the_module_imports_nothing_and_defines_only_constants():
    """It is imported by scripts and Spark: no ``app.*``, no third-party, no code."""
    source = Path(thresholds.__file__).read_text()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        # ``from __future__`` is a compiler directive, not a dependency.
        if isinstance(node, ast.ImportFrom):
            assert node.module == "__future__", f"unexpected import from {node.module}"
        assert not isinstance(node, ast.Import), "thresholds must not import anything"

    statements = [
        node
        for node in tree.body
        if not isinstance(node, (ast.Expr, ast.AnnAssign))  # docstring, comments
    ]
    assert statements, "expected the constants"
    for statement in statements:
        assert isinstance(statement, ast.Assign), type(statement).__name__
        for target in statement.targets:
            assert isinstance(target, ast.Name) and target.id.isupper()


def test_consumers_read_the_shared_values_not_copies():
    # Identity, not equality: a copied literal would drift silently.
    assert weather_repo.QUALITY_OK_AGE_HOURS is thresholds.QUALITY_OK_AGE_HOURS
    assert weather_repo.QUALITY_WARN_AGE_HOURS is thresholds.QUALITY_WARN_AGE_HOURS
    assert predictions_router.PREDICTION_STALE_AFTER_HOURS is (
        thresholds.PREDICTION_STALE_AFTER_HOURS
    )


def test_there_is_no_env_override_for_the_stale_threshold(monkeypatch):
    monkeypatch.setenv("PREDICTION_STALE_AFTER_HOURS", "24")
    assert not hasattr(Settings(_env_file=None), "prediction_stale_after_hours")
    # 12 h, not the old 24 h: the badge cannot be retuned out of band any more.
    assert thresholds.PREDICTION_STALE_AFTER_HOURS == 12.0
    assert thresholds.PREDICTION_OK_AGE_HOURS == thresholds.PREDICTION_STALE_AFTER_HOURS
