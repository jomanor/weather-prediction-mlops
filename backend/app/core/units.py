"""One authority for everything that is a unit, not a threshold.

Plain constants and nothing else: no imports, so any layer (repositories,
routers, scripts) can import this file without dragging in a framework. Two
repositories read wind speeds out of the stored documents, so the conversion
lives here once instead of being re-declared per repository (audit finding 20).
"""

#: Stored m/s -> contract km/h. Both writers request ``wind_speed_unit=ms`` (see
#: ``scripts/ingest_weather.py`` and ``scripts/backfill_historical_data.py``), so
#: a stored wind speed is m/s while ``CurrentWeather.wind_speed`` promises km/h.
MS_TO_KMH = 3.6
