"""
tests/unit/test_watchdog.py
===========================
Unit tests for scripts/watchdog.py — the pipeline dead-man's switch.

Pure logic plus an injected fake transport: no network, no GitHub, no token.

**Hermetic by construction.** ``build_parser()`` reads ``GITHUB_REPOSITORY`` and
``GITHUB_TOKEN`` from the environment *at call time* (that is the documented CLI
behaviour), and GitHub Actions exports both into every job. The autouse
``_hermetic_env`` fixture below clears them, so each test states its inputs
explicitly instead of inheriting whatever the runner happens to set — without it
``test_missing_repo_exits_1`` passed locally and failed on CI, where
``GITHUB_REPOSITORY`` is ambient.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
from datetime import datetime, timedelta, timezone

import pytest

from tests.conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "scripts"))

import watchdog as wd  # noqa: E402

NOW = datetime(2026, 10, 31, 12, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

#: Every environment variable ``watchdog.build_parser`` reads a default from.
#: Keep this list equal to the ``os.environ`` reads in scripts/watchdog.py.
ENV_DERIVED_DEFAULTS = ("GITHUB_REPOSITORY", "GITHUB_TOKEN")


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch):
    """Strip the ambient GitHub env so no test depends on the runner's environment."""
    for name in ENV_DERIVED_DEFAULTS:
        monkeypatch.delenv(name, raising=False)
    yield


def iso(hours_ago: float, *, now: datetime = NOW) -> str:
    """ISO-8601 UTC string ``hours_ago`` before ``now`` (the API's format)."""
    return (now - timedelta(hours=hours_ago)).isoformat().replace("+00:00", "Z")


def current_payload(station_ages: dict[str, float | None]) -> dict:
    stations = []
    for city, age in station_ages.items():
        station = {"city": city, "temperature": 20.0}
        if age is not None:
            station["observed_at"] = iso(age)
        stations.append(station)
    return {"count": len(stations), "stations": stations}


def quality_payload(completeness_by_city: dict[str, float | None]) -> dict:
    cities = []
    for city, completeness in completeness_by_city.items():
        entry = {"city": city, "observed_hours": 100, "status": "ok"}
        if completeness is not None:
            entry["completeness"] = completeness
        cities.append(entry)
    return {"generated_at": iso(0), "days": 7, "cities": cities}


def predictions_payload(city_ages: dict[str, float | None]) -> dict:
    rows = []
    for city, age in city_ages.items():
        row = {"city": city, "horizon_hours": 1, "prediction_timestamp": iso(0)}
        if age is not None:
            row["source_timestamp"] = iso(age)
        rows.append(row)
    return {"count": len(rows), "generated_at": iso(0), "predictions": rows}


def models_payload(entries: list[dict]) -> dict:
    return {"count": len(entries), "models": entries}


def healthy_payloads(**overrides) -> dict[str, object]:
    """All four endpoint payloads, healthy by default; ``overrides`` win by path."""
    payloads: dict[str, object] = {
        wd.PATH_CURRENT: current_payload({"Madrid": 1.0, "Barcelona": 2.0}),
        wd.PATH_QUALITY: quality_payload({"Madrid": 0.99, "Barcelona": 0.98}),
        wd.PATH_PREDICTIONS: predictions_payload({"Madrid": 1.0, "Barcelona": 2.0}),
        wd.PATH_MODELS: models_payload(
            [
                {
                    "name": "temp_prediction_1h_GBT",
                    "target": "temperature",
                    "horizon_hours": 1,
                    "stage": "production",
                    "metrics": {"skill_score": 0.53},
                }
            ]
        ),
    }
    payloads.update(overrides)
    return payloads


def fake_fetch(payloads: dict[str, object]):
    """Transport stub: full URL -> payload, keyed by API path.

    A payload that is an ``Exception`` instance is raised instead of returned, so
    a single endpoint can be made to fail (F2/F3) without touching the others.
    """

    def fetch(url: str):
        for path, payload in payloads.items():
            if url.endswith(path):
                if isinstance(payload, BaseException):
                    raise payload
                return payload
        raise AssertionError(f"unexpected url: {url}")

    return fetch


def verdict_for(**overrides) -> wd.Verdict:
    return wd.gather_verdict(
        api_base=wd.DEFAULT_API_BASE, fetch=fake_fetch(healthy_payloads(**overrides)), now=NOW
    )


def status_of(verdict: wd.Verdict, key: str) -> str:
    return verdict.statuses[key]


class FakeGitHub:
    """Records every call so tests can assert reads vs writes."""

    def __init__(self, issue: dict | None = None, last_verdict: dict | None = None) -> None:
        self.issue = issue
        self.last_verdict = last_verdict
        self.calls: list[tuple] = []

    def find_open_issue(self):
        self.calls.append(("find_open_issue",))
        return self.issue

    def last_recorded_verdict(self, issue):
        self.calls.append(("last_recorded_verdict",))
        return self.last_verdict

    def create_issue(self, body):
        self.calls.append(("create_issue", body))
        return {"number": 11}

    def comment(self, number, body):
        self.calls.append(("comment", number, body))

    def close(self, number):
        self.calls.append(("close", number))

    @property
    def writes(self) -> list[tuple]:
        return [
            call
            for call in self.calls
            if call[0] != "find_open_issue" and call[0] != "last_recorded_verdict"
        ]


def parse_args(*argv: str):
    return wd.build_parser().parse_args(list(argv))


def run_with(payloads, *, github: FakeGitHub, argv: list[str], now: datetime = NOW, token="t0ken"):
    """Run ``watchdog.run`` with a fake transport and fake GitHub client."""
    args = parse_args(*argv)
    if token is not None and "--token" not in argv:
        args.token = token
    lines: list[str] = []
    errors: list[str] = []
    code = wd.run(
        args,
        fetch=fake_fetch(payloads),
        client_factory=lambda **_: github,
        now=now,
        out=lines.append,
        err=errors.append,
    )
    return code, lines, errors


# ---------------------------------------------------------------------------
# Threshold boundaries — exactly 12 h / 18 h / 0.80 / 0.90
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("hours", "expected"),
    [
        (11.9, "ok"),
        (12.0, "ok"),  # exactly 12 h is ok
        (12.0001, "warn"),
        (18.0, "warn"),  # exactly 18 h is still warn
        (18.0001, "bad"),
        (30.0, "bad"),
    ],
)
def test_ingest_age_boundaries(hours, expected):
    verdict = verdict_for(**{wd.PATH_CURRENT: current_payload({"Madrid": hours})})
    assert status_of(verdict, "ingest_age") == expected


@pytest.mark.parametrize(
    ("hours", "expected"),
    [
        (12.0, "ok"),
        (12.0001, "ok"),  # prediction alert only starts past 18 h
        (18.0, "ok"),
        (18.0001, "bad"),
    ],
)
def test_prediction_age_boundaries(hours, expected):
    verdict = verdict_for(
        **{
            wd.PATH_CURRENT: current_payload({"Madrid": 1.0}),
            wd.PATH_PREDICTIONS: predictions_payload({"Madrid": hours}),
        }
    )
    assert status_of(verdict, "prediction_age") == expected


@pytest.mark.parametrize(
    ("completeness", "expected"),
    [(0.95, "ok"), (0.80, "ok"), (0.8001, "ok"), (0.7999, "bad"), (0.23, "bad")],
)
def test_completeness_boundary(completeness, expected):
    verdict = verdict_for(**{wd.PATH_QUALITY: quality_payload({"Madrid": completeness})})
    assert status_of(verdict, "ingest_completeness") == expected


def _cities(count: int) -> list[str]:
    return [f"City{i}" for i in range(count)]


@pytest.mark.parametrize(
    ("predicted", "total", "expected"),
    [
        (15, 15, "ok"),
        (14, 15, "ok"),  # a single missing city is 93 % and must NOT alert
        (900, 1000, "ok"),  # coverage exactly 0.90 is ok
        (899, 1000, "bad"),  # 0.899 is bad
        (10, 1000, "bad"),
    ],
)
def test_prediction_coverage_boundary(predicted, total, expected):
    cities = _cities(total)
    stations = {city: 1.0 for city in cities}
    predicted_cities = {city: 1.0 for city in cities[:predicted]}
    verdict = verdict_for(
        **{
            wd.PATH_CURRENT: current_payload(stations),
            wd.PATH_PREDICTIONS: predictions_payload(predicted_cities),
        }
    )
    assert status_of(verdict, "prediction_coverage") == expected


def test_coverage_14_of_15_reports_ratio():
    cities = _cities(15)
    verdict = verdict_for(
        **{
            wd.PATH_CURRENT: current_payload({city: 1.0 for city in cities}),
            wd.PATH_PREDICTIONS: predictions_payload({city: 1.0 for city in cities[:14]}),
        }
    )
    coverage = next(signal for signal in verdict.signals if signal.key == "prediction_coverage")
    assert coverage.value.startswith("0.933")
    assert coverage.is_bad is False
    assert verdict.triggered is False


# ---------------------------------------------------------------------------
# Newest vs oldest, missing timestamps
# ---------------------------------------------------------------------------


def test_ingest_age_uses_newest_station_not_oldest():
    verdict = verdict_for(
        **{
            wd.PATH_CURRENT: current_payload({"Madrid": 1.0, "Barcelona": 40.0, "Bilbao": 20.0}),
            wd.PATH_PREDICTIONS: predictions_payload(
                {"Madrid": 1.0, "Barcelona": 40.0, "Bilbao": 20.0}
            ),
        }
    )
    age = next(signal for signal in verdict.signals if signal.key == "ingest_age")
    assert age.status == "ok"
    assert age.value == "1.0 h"
    assert verdict.triggered is False


def test_prediction_age_uses_newest_source_timestamp():
    verdict = verdict_for(
        **{wd.PATH_PREDICTIONS: predictions_payload({"Madrid": 2.0, "Barcelona": 30.0})}
    )
    age = next(signal for signal in verdict.signals if signal.key == "prediction_age")
    assert age.status == "ok"
    assert age.value == "2.0 h"


def test_missing_observed_at_is_bad():
    verdict = verdict_for(
        **{
            wd.PATH_CURRENT: current_payload({"Madrid": None, "Barcelona": None}),
            wd.PATH_PREDICTIONS: predictions_payload({"Madrid": 1.0, "Barcelona": 1.0}),
        }
    )
    age = next(signal for signal in verdict.signals if signal.key == "ingest_age")
    assert age.status == "bad"
    assert age.value == "sin observed_at"
    assert verdict.triggered is True


def test_missing_source_timestamp_is_bad():
    verdict = verdict_for(**{wd.PATH_PREDICTIONS: predictions_payload({"Madrid": None})})
    age = next(signal for signal in verdict.signals if signal.key == "prediction_age")
    assert age.status == "bad"
    assert age.value == "sin source_timestamp"
    assert verdict.triggered is True


def test_empty_observed_at_payload_is_not_a_check_failure():
    # A valid payload without usable timestamps is a *pipeline* problem (bad),
    # never a broken watchdog (exit 1): only the transport/parse path raises.
    verdict = verdict_for(**{wd.PATH_CURRENT: {"count": 0, "stations": []}})
    assert status_of(verdict, "ingest_age") == "bad"


# ---------------------------------------------------------------------------
# Model degradation — advisory only
# ---------------------------------------------------------------------------


def _model(name="m", stage="production", diagnostics=None, metrics=None, horizon=1):
    entry = {"name": name, "target": "temperature", "horizon_hours": horizon, "stage": stage}
    if diagnostics is not None:
        entry["diagnostics"] = diagnostics
    if metrics is not None:
        entry["metrics"] = metrics
    return entry


def test_advisory_from_metrics_skill_score():
    advisories = wd.check_models(
        models_payload(
            [_model("temp_prediction_24h_LR", metrics={"skill_score": -0.117}, horizon=24)]
        )
    )
    assert len(advisories) == 1
    assert "temp_prediction_24h_LR" in advisories[0]
    assert "-0.117" in advisories[0]


def test_advisory_from_diagnostics_skill_score():
    advisories = wd.check_models(
        models_payload([_model("rain_prediction_1h_G", diagnostics={"skill_score": -0.044})])
    )
    assert len(advisories) == 1


@pytest.mark.parametrize(
    "entry",
    [
        _model("a", metrics={"skill_score": None}),
        _model("b", metrics=None),
        _model("c"),  # no diagnostics and no metrics at all
        _model("d", diagnostics={}, metrics={}),
        _model("e", stage="staging", metrics={"skill_score": -0.5}),
        _model("f", stage=None, metrics={"skill_score": -0.5}),
        _model("g", metrics={"skill_score": 0.01}),
    ],
)
def test_no_advisory_when_score_unknown_or_model_not_production(entry):
    assert wd.check_models(models_payload([entry])) == []


def test_skill_score_zero_is_advisory():
    assert len(wd.check_models(models_payload([_model("z", metrics={"skill_score": 0.0})]))) == 1


def test_advisory_never_triggers_an_alert():
    verdict = verdict_for(
        **{
            wd.PATH_MODELS: models_payload(
                [
                    _model("temp_prediction_24h_LR", metrics={"skill_score": -0.117}, horizon=24),
                    _model("rain_prediction_1h_G", metrics={"skill_score": -0.044}),
                ]
            )
        }
    )
    assert verdict.triggered is False
    assert verdict.status == "ok"
    assert len(verdict.advisories) == 2


def test_advisory_only_appears_when_an_issue_is_opened():
    assert "Aviso" not in wd.render_issue_body(verdict_for())
    alerting = verdict_for(**{wd.PATH_CURRENT: current_payload({"Madrid": 40.0})})
    assert "Aviso" not in wd.render_issue_body(alerting)  # healthy models -> no block
    with_advisory = verdict_for(
        **{
            wd.PATH_CURRENT: current_payload({"Madrid": 40.0}),
            wd.PATH_MODELS: models_payload([_model("m", metrics={"skill_score": -0.2})]),
        }
    )
    assert "skill_score -0.200" in wd.render_issue_body(with_advisory)


# ---------------------------------------------------------------------------
# Decision table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("triggered", "issue_number", "last_verdict", "kind", "comment"),
    [
        (True, None, None, "create", False),
        (True, 5, {"ingest_age": "bad", "prediction_age": "ok"}, "comment", True),
        (True, 5, {"ingest_age": "ok", "prediction_age": "ok"}, "none", False),  # unchanged
        (True, 5, None, "comment", True),
        (False, 5, {"ingest_age": "bad"}, "close", True),  # recovery, new verdict
        (
            False,
            5,
            {"ingest_age": "ok", "prediction_age": "ok"},
            "close",
            False,
        ),  # already recorded
        (False, None, None, "none", False),
    ],
)
def test_plan_action_table(triggered, issue_number, last_verdict, kind, comment):
    action = wd.plan_action(
        triggered=triggered,
        issue_number=issue_number,
        last_verdict=last_verdict,
        current_verdict={"ingest_age": "ok", "prediction_age": "ok"},
    )
    assert (action.kind, action.comment) == (kind, comment)


def test_embedded_verdict_block_round_trips():
    verdict = verdict_for()
    for body in (wd.render_issue_body(verdict), wd.render_comment_body(verdict)):
        assert wd.extract_verdict(body) == verdict.statuses


def test_extract_verdict_handles_garbage():
    assert wd.extract_verdict(None) is None
    assert wd.extract_verdict("no marker here") is None
    assert wd.extract_verdict("<!-- watchdog-verdict {not json} -->") is None


def test_verdict_block_is_stable_and_compact():
    block = verdict_for().statuses_json()
    assert block == json.dumps(json.loads(block), sort_keys=True, separators=(",", ":"))
    assert verdict_for().statuses_json() == block  # same evidence -> identical bytes


# ---------------------------------------------------------------------------
# run(): create / comment / close / no-op
# ---------------------------------------------------------------------------


ALERT = {
    "ingest_age": "bad",
    "ingest_completeness": "ok",
    "prediction_age": "ok",
    "prediction_coverage": "ok",
}


def test_run_creates_issue_with_title_and_label():
    github = FakeGitHub()
    code, lines, errors = run_with(
        healthy_payloads(**{wd.PATH_CURRENT: current_payload({"Madrid": 40.0})}),
        github=github,
        argv=["--repo", "jomanor/weather-prediction-mlops"],
    )
    assert code == 0, errors
    assert [call[0] for call in github.writes] == ["create_issue"]
    body = github.writes[0][1]
    assert wd.ISSUE_TITLE == "[ops] Pipeline de datos desactualizado"
    assert wd.ISSUE_LABEL == "ops-alert"
    assert "| Observaciones" in body
    assert "40.0 h" in body


def test_run_comments_when_verdict_changed():
    github = FakeGitHub(
        issue={"number": 9}, last_verdict={"ingest_age": "ok", "prediction_age": "ok"}
    )
    code, lines, _ = run_with(
        healthy_payloads(**{wd.PATH_CURRENT: current_payload({"Madrid": 40.0})}),
        github=github,
        argv=["--repo", "jomanor/weather-prediction-mlops"],
    )
    assert code == 0
    assert [call[0] for call in github.writes] == ["comment"]
    assert github.writes[0][1] == 9


def test_run_does_not_comment_on_unchanged_verdict():
    github = FakeGitHub(issue={"number": 9}, last_verdict=ALERT)
    code, lines, _ = run_with(
        healthy_payloads(
            **{
                wd.PATH_CURRENT: current_payload({"Madrid": 40.0}),
                wd.PATH_PREDICTIONS: predictions_payload({"Madrid": 1.0, "Barcelona": 1.0}),
            }
        ),
        github=github,
        argv=["--repo", "jomanor/weather-prediction-mlops"],
    )
    assert code == 0
    assert github.writes == []
    assert any("sin cambios" in line for line in lines)


def test_run_closes_issue_on_recovery():
    github = FakeGitHub(
        issue={"number": 9}, last_verdict={"ingest_age": "bad", "prediction_age": "bad"}
    )
    code, _, _ = run_with(
        healthy_payloads(), github=github, argv=["--repo", "jomanor/weather-prediction-mlops"]
    )
    assert code == 0
    assert [call[0] for call in github.writes] == ["comment", "close"]
    assert github.writes[1][1] == 9


def test_run_does_not_recomment_recovery():
    github = FakeGitHub(
        issue={"number": 9},
        last_verdict={
            "ingest_age": "ok",
            "ingest_completeness": "ok",
            "prediction_age": "ok",
            "prediction_coverage": "ok",
        },
    )
    code, _, _ = run_with(
        healthy_payloads(), github=github, argv=["--repo", "jomanor/weather-prediction-mlops"]
    )
    assert code == 0
    assert [call[0] for call in github.writes] == ["close"]


def test_at_most_one_comment_per_run():
    github = FakeGitHub(issue={"number": 9}, last_verdict=None)
    run_with(
        healthy_payloads(
            **{
                wd.PATH_CURRENT: current_payload({"Madrid": 40.0}),
                wd.PATH_PREDICTIONS: predictions_payload({"Madrid": 40.0, "Barcelona": 40.0}),
            }
        ),
        github=github,
        argv=["--repo", "jomanor/weather-prediction-mlops"],
    )
    assert len([call for call in github.writes if call[0] == "comment"]) <= 1


# ---------------------------------------------------------------------------
# dry-run / exit codes
# ---------------------------------------------------------------------------


def test_dry_run_never_writes_and_needs_no_token():
    github = FakeGitHub(issue={"number": 9}, last_verdict=None)
    code, lines, errors = run_with(
        healthy_payloads(**{wd.PATH_CURRENT: current_payload({"Madrid": 40.0})}),
        github=github,
        argv=["--repo", "jomanor/weather-prediction-mlops", "--dry-run"],
        token=None,
    )
    assert code == 0, errors
    assert github.writes == []
    assert all(call[0] in {"find_open_issue", "last_recorded_verdict"} for call in github.calls)
    assert any("40.0 h" in line for line in lines)
    assert any(line.startswith("plan: comment") for line in lines)
    assert any("no se ha escrito nada" in line for line in lines)


def test_dry_run_without_repo_prints_verdict():
    lines: list[str] = []
    code = wd.run(
        parse_args("--dry-run"),
        fetch=fake_fetch(healthy_payloads()),
        now=NOW,
        out=lines.append,
        err=lines.append,
    )
    assert code == 0
    assert any(line.startswith("veredicto: ok") for line in lines)


def test_dry_run_json_output():
    lines: list[str] = []
    code = wd.run(
        parse_args("--dry-run", "--json"),
        fetch=fake_fetch(healthy_payloads()),
        now=NOW,
        out=lines.append,
    )
    payload = json.loads(lines[-1])
    assert code == 0
    assert payload["triggered"] is False
    assert {signal["signal"] for signal in payload["signals"]} == {
        "ingest_age",
        "ingest_completeness",
        "prediction_age",
        "prediction_coverage",
    }
    assert payload["action"] == "none"


def test_missing_token_exits_1_without_traceback():
    github = FakeGitHub()
    code, lines, errors = run_with(
        healthy_payloads(),
        github=github,
        argv=["--repo", "jomanor/weather-prediction-mlops"],
        token=None,
    )
    assert code == 1
    assert github.calls == []
    assert any("GITHUB_TOKEN" in line for line in errors)
    assert not any("Traceback" in line for line in errors + lines)


def test_missing_repo_exits_1():
    lines: list[str] = []
    code = wd.run(
        parse_args("--token", "t"),
        fetch=fake_fetch(healthy_payloads()),
        now=NOW,
        out=lines.append,
        err=lines.append,
    )
    assert code == 1
    assert any("GITHUB_REPOSITORY" in line for line in lines)


def test_env_defaults_are_hermetic(monkeypatch):
    """Regression: the suite must not inherit GITHUB_REPOSITORY / GITHUB_TOKEN.

    CI exports both, so an un-cleared environment makes ``--repo``/``--token``
    resolve to the runner's values and turns the documented exit-1 paths green
    (or worse, lets a test reach the real GitHub API). This asserts both
    directions: cleared -> the documented outcomes, set -> the defaults really do
    come from the environment, which is why the fixture must clear them.
    """
    seen_clients: list[dict] = []

    def client_factory(**kwargs):
        seen_clients.append(kwargs)
        return FakeGitHub()

    def run_path(*argv: str):
        lines: list[str] = []
        code = wd.run(
            parse_args(*argv),
            fetch=fake_fetch(healthy_payloads()),
            client_factory=client_factory,
            now=NOW,
            out=lines.append,
            err=lines.append,
        )
        return code, lines

    # 1. Cleared environment (the fixture's state): no repo -> exit 1, no client.
    assert "GITHUB_REPOSITORY" not in os.environ
    code, lines = run_path("--token", "t")
    assert code == 1
    assert any("GITHUB_REPOSITORY" in line for line in lines)
    assert seen_clients == []

    # 2. Cleared environment: no token -> exit 1 even with an explicit repo.
    code, lines = run_path("--repo", "o/r")
    assert code == 1
    assert any("GITHUB_TOKEN" in line for line in lines)
    assert seen_clients == []

    # 3. Ambient env as GitHub Actions sets it: the defaults are picked up at
    #    parse time, so only the fixture keeps the suite deterministic.
    monkeypatch.setenv("GITHUB_REPOSITORY", "jomanor/weather-prediction-mlops")
    monkeypatch.setenv("GITHUB_TOKEN", "ambient-token")
    code, lines = run_path()
    assert code == 0
    assert seen_clients == [{"repo": "jomanor/weather-prediction-mlops", "token": "ambient-token"}]


def test_api_failure_after_retries_is_api_down_not_a_check_failure():
    """F2b: an unreachable API is a verdict (exit 0), never a red watchdog run."""
    calls = {"n": 0}

    def fetch(url):
        calls["n"] += 1
        raise wd.WatchdogError("HTTP 502 Bad Gateway")

    lines: list[str] = []
    code = wd.run(parse_args("--dry-run"), fetch=fetch, now=NOW, out=lines.append, err=lines.append)
    assert code == 0
    assert calls["n"] == 1  # the retry loop lives inside fetch_json, not in run()
    assert any(line.startswith("veredicto: api_down") for line in lines)
    assert any("HTTP 502 Bad Gateway" in line for line in lines)  # never swallowed


def test_api_down_verdict_uses_its_own_marker_key():
    verdict = verdict_for(**{wd.PATH_CURRENT: wd.WatchdogError("connection reset")})
    assert verdict.status == "api_down"
    assert verdict.triggered is True
    assert verdict.statuses == {"api": "api_down"}
    assert "api_down" in wd.render_issue_body(verdict)


def test_unparseable_payload_is_a_check_failure():
    def fetch(url):
        return ["not", "a", "dict"]

    lines: list[str] = []
    code = wd.run(parse_args("--dry-run"), fetch=fetch, now=NOW, out=lines.append, err=lines.append)
    assert code == 1
    assert any("payload inesperado" in line for line in lines)


def test_github_read_failure_in_dry_run_still_prints_plan():
    class Broken(FakeGitHub):
        def find_open_issue(self):
            raise wd.GitHubError("boom")

    lines: list[str] = []
    code = wd.run(
        parse_args("--repo", "o/r", "--dry-run"),
        fetch=fake_fetch(healthy_payloads()),
        client_factory=lambda **_: Broken(),
        now=NOW,
        out=lines.append,
        err=lines.append,
    )
    assert code == 0
    assert any("plan:" in line for line in lines)


def test_github_write_failure_exits_1():
    class Broken(FakeGitHub):
        def create_issue(self, body):
            raise wd.GitHubError("HTTP 403")

    lines: list[str] = []
    code = wd.run(
        parse_args("--repo", "o/r", "--token", "t"),
        fetch=fake_fetch(healthy_payloads(**{wd.PATH_CURRENT: current_payload({"Madrid": 40.0})})),
        client_factory=lambda **_: Broken(),
        now=NOW,
        out=lines.append,
        err=lines.append,
    )
    assert code == 1
    assert any("HTTP 403" in line for line in lines)


# ---------------------------------------------------------------------------
# fetch_json retry policy (5 s / 10 s, 3 attempts)
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, body: dict) -> None:
        self._body = json.dumps(body).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def test_fetch_json_retries_then_succeeds():
    attempts: list[int] = []
    sleeps: list[float] = []

    def open_url(request, timeout=None):
        attempts.append(1)
        if len(attempts) < 3:
            raise OSError("connection reset")
        return _FakeResponse({"ok": True})

    payload = wd.fetch_json(
        "https://example.test/x", open_url=open_url, sleep=sleeps.append, timeout=1.0
    )
    assert payload == {"ok": True}
    assert len(attempts) == 3
    assert sleeps == [5.0, 10.0]


def test_fetch_json_gives_up_after_three_attempts():
    sleeps: list[float] = []

    def open_url(request, timeout=None):
        raise OSError("prematurely reached end of stream")

    with pytest.raises(wd.WatchdogError) as excinfo:
        wd.fetch_json("https://example.test/x", open_url=open_url, sleep=sleeps.append)
    assert "3 intentos" in str(excinfo.value)
    assert sleeps == [5.0, 10.0]


def test_fetch_json_retries_unparseable_body():
    sleeps: list[float] = []

    class Bad:
        def read(self):
            return b"<html>502</html>"

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

    with pytest.raises(wd.WatchdogError):
        wd.fetch_json("https://example.test/x", open_url=lambda *a, **k: Bad(), sleep=sleeps.append)


# ---------------------------------------------------------------------------
# F2: cold-start timeout + bounded total
# ---------------------------------------------------------------------------


def test_http_timeout_covers_a_render_cold_start():
    assert wd.HTTP_TIMEOUT_SECONDS >= 120.0
    assert wd.API_ATTEMPTS == 3  # 3 attempts and 5/10 s backoff are frozen
    assert wd.API_BACKOFF_SECONDS == (5.0, 10.0)


def test_budget_bounds_the_run_well_inside_the_workflow_timeout():
    workflow = (REPO_ROOT / ".github" / "workflows" / "watchdog.yml").read_text()
    match = re.search(r"timeout-minutes:\s*(\d+)", workflow)
    assert match is not None, "watchdog.yml must declare an explicit job timeout"
    workflow_budget = int(match.group(1)) * 60
    worst_api_phase = wd.API_TOTAL_BUDGET_SECONDS + wd.HTTP_TIMEOUT_SECONDS
    # 4 GitHub calls (find issue, comments, write, label) x (3 attempts x 30 s + 15 s).
    worst_github_phase = 4 * (wd.API_ATTEMPTS * wd.GITHUB_TIMEOUT_SECONDS + 15.0)
    assert worst_api_phase + worst_github_phase <= workflow_budget


def test_fetch_json_stops_when_the_budget_is_exhausted():
    clock = iter([0.0, 0.0, 999.0]).__next__

    def open_url(request, timeout=None):
        raise OSError("timed out")

    with pytest.raises(wd.WatchdogError) as excinfo:
        wd.fetch_json(
            "https://example.test/x",
            open_url=open_url,
            sleep=lambda _s: None,
            deadline=10.0,
            clock=clock,
        )
    assert "presupuesto" in str(excinfo.value)


def test_502_on_every_attempt_produces_api_down_and_opens_the_issue():
    def open_url(request, timeout=None):
        raise urllib.error.HTTPError("https://api.test/x", 502, "Bad Gateway", {}, None)

    def fetch(url):
        return wd.fetch_json(url, open_url=open_url, sleep=lambda _s: None)

    github = FakeGitHub()
    lines: list[str] = []
    errors: list[str] = []
    code = wd.run(
        parse_args("--repo", "jomanor/weather-prediction-mlops", "--token", "t0ken"),
        fetch=fetch,
        client_factory=lambda **_: github,
        now=NOW,
        out=lines.append,
        err=errors.append,
    )
    assert code == 0, errors
    assert [call[0] for call in github.writes] == ["create_issue"]
    body = github.writes[0][1]
    assert "api_down" in body
    assert "502" in body
    assert any("502" in line for line in errors)


# ---------------------------------------------------------------------------
# F3: advisory gathering is fault-tolerant
# ---------------------------------------------------------------------------


def test_models_500_still_alerts_on_staleness_and_records_advisory_error():
    verdict = verdict_for(
        **{
            wd.PATH_CURRENT: current_payload({"Madrid": 40.0}),
            wd.PATH_MODELS: wd.WatchdogError("GET /api/v1/models 502 Bad Gateway"),
        }
    )
    assert verdict.triggered is True
    assert status_of(verdict, "ingest_age") == "bad"
    assert verdict.api_error is None
    assert verdict.advisories == ()
    assert verdict.advisory_error is not None
    assert "502" in verdict.advisory_error
    assert "502" in verdict.as_dict()["advisory_error"]
    body = wd.render_issue_body(verdict)
    assert "Observaciones" in body  # the graded evidence is still there
    assert "Aviso no disponible" in body
    assert "502" in body


def test_broken_models_payload_is_an_advisory_error_not_a_crash():
    verdict = verdict_for(**{wd.PATH_MODELS: ["not", "a", "dict"]})
    assert verdict.triggered is False
    assert verdict.status == "ok"
    assert verdict.advisory_error is not None
    assert "payload inesperado" in verdict.advisory_error


def test_advisory_error_is_absent_when_models_answers():
    verdict = verdict_for()
    assert verdict.advisory_error is None
    assert "Aviso no disponible" not in wd.render_issue_body(verdict)


# ---------------------------------------------------------------------------
# F4: parse_timestamp normalisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # naive -> treated as UTC
        ("2026-10-31T12:00:00", datetime(2026, 10, 31, 12, 0, tzinfo=timezone.utc)),
        ("2026-10-31T12:00:00.500", datetime(2026, 10, 31, 12, 0, 0, 500000, timezone.utc)),
        # Z / z / +00:00
        ("2026-10-31T12:00:00Z", datetime(2026, 10, 31, 12, 0, tzinfo=timezone.utc)),
        ("2026-10-31T12:00:00z", datetime(2026, 10, 31, 12, 0, tzinfo=timezone.utc)),
        ("2026-10-31T12:00:00+00:00", datetime(2026, 10, 31, 12, 0, tzinfo=timezone.utc)),
        # fractional seconds, with the Z suffix
        ("2026-10-31T12:00:00.123456Z", datetime(2026, 10, 31, 12, 0, 0, 123456, timezone.utc)),
        # non-UTC offset -> converted to UTC
        ("2026-10-31T05:00:00-07:00", datetime(2026, 10, 31, 12, 0, tzinfo=timezone.utc)),
        ("2026-10-31T14:30:00+02:30", datetime(2026, 10, 31, 12, 0, tzinfo=timezone.utc)),
    ],
)
def test_parse_timestamp_normalises_to_utc(raw, expected):
    parsed = wd.parse_timestamp(raw)
    assert parsed == expected
    assert parsed.tzinfo is timezone.utc


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "garbage",
        "2026-13-45T99:99:99Z",
        None,
        42,
        {"when": "2026-10-31T12:00:00Z"},
    ],
)
def test_parse_timestamp_returns_none(raw):
    assert wd.parse_timestamp(raw) is None


def test_parse_timestamp_accepts_surrounding_whitespace():
    assert wd.parse_timestamp("  2026-10-31T12:00:00Z ") == datetime(
        2026, 10, 31, 12, 0, tzinfo=timezone.utc
    )


# ---------------------------------------------------------------------------
# GitHub client (urllib stubbed) — 422 on the label
# ---------------------------------------------------------------------------


class _JsonResponse:
    """Minimal stand-in for ``http.client.HTTPResponse`` (context manager + status)."""

    def __init__(self, status: int, body: object) -> None:
        self.status = status
        self._body = json.dumps(body)

    def read(self) -> bytes:
        return self._body.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def test_create_issue_creates_missing_label_and_retries():
    calls: list[tuple[str, str]] = []
    responses = [
        (422, "Validation Failed"),
        (201, {"name": "ops-alert"}),
        (201, {"number": 3}),
    ]

    def open_url(request, timeout=None):
        calls.append((request.get_method(), request.full_url))
        status, body = responses.pop(0)
        return _JsonResponse(status, body)

    client = wd.GitHubClient(repo="o/r", token="t", open_url=open_url)
    issue = client.create_issue("body")
    assert issue == {"number": 3}
    assert [method for method, _ in calls] == ["POST", "POST", "POST"]
    assert calls[1][1].endswith("/labels")
    assert calls[2][1].endswith("/issues")


def test_find_open_issue_matches_title_only():
    def open_url(request, timeout=None):
        return _JsonResponse(
            200,
            [
                {"number": 1, "title": "otra cosa"},
                {"number": 2, "title": wd.ISSUE_TITLE},
            ],
        )

    client = wd.GitHubClient(repo="o/r", token="t", open_url=open_url)
    assert client.find_open_issue()["number"] == 2


def test_last_recorded_verdict_prefers_newest_comment():
    bodies = [
        'old comment\n<!-- watchdog-verdict {"ingest_age":"bad"} -->',
        'new comment\n<!-- watchdog-verdict {"ingest_age":"ok"} -->',
    ]

    def open_url(request, timeout=None):
        return _JsonResponse(200, [{"body": body} for body in bodies])

    client = wd.GitHubClient(repo="o/r", token="t", open_url=open_url)
    assert client.last_recorded_verdict({"number": 4, "body": "sin bloque"}) == {"ingest_age": "ok"}
