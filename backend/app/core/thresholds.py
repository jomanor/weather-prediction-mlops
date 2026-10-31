"""One authority for every freshness/quality age threshold in the repo.

Plain constants and nothing else: no imports from ``app.*``, no third-party
imports, so the API, the scripts and the Spark jobs can all import this exact
file without dragging in a web framework or a settings object. There is no env
override — a single number that four surfaces read is the whole point (audit
finding 5: "fresh" was 24 h in the API, 12 h in the watchdog and warned about at
12 h by inference).

Age is measured on the *source observation* timestamp, in hours, against a
single ``now`` per response.
"""

#: Data-quality meter: at or below this source age a station grades ``ok``.
QUALITY_OK_AGE_HOURS = 12.0
#: Data-quality meter: above this source age a station grades ``bad`` (``warn`` in between).
QUALITY_WARN_AGE_HOURS = 18.0
#: Data-quality meter: completeness below this ratio grades ``bad``.
QUALITY_BAD_COMPLETENESS_BELOW = 0.80

#: A served prediction whose source observation is older than this is ``stale``.
PREDICTION_STALE_AFTER_HOURS = 12.0
#: Prediction freshness badge: at or below this source age the badge is ``ok``.
PREDICTION_OK_AGE_HOURS = 12.0
#: Prediction freshness badge: above this source age the badge is ``bad`` (``warn`` in between).
PREDICTION_WARN_AGE_HOURS = 18.0
#: Prediction interval coverage below this ratio is flagged for retraining.
PREDICTION_COVERAGE_WARN_RATIO = 0.90
#: A model skill score at or below this value is advisory only (no better than persistence).
SKILL_SCORE_ADVISORY_AT_OR_BELOW = 0.0
