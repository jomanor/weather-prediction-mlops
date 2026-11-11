"""Seed data for the station registry.

These are the fourteen Spanish stations the ingest covers. On startup the API
inserts them into the ``cities`` collection when it is empty — that is all this
tuple is for. Once seeded, the Mongo registry is the only runtime source of the
station set: the ingest, the backfill, the quality meter and the analytics
aggregates all read their cities from it, so adding or deleting a station
through ``/cities`` immediately changes what they ingest and report. The tuple
is also the fallback when the registry is empty or unreadable. Names are kept
byte-for-byte as the ingest writer uses them, including the unaccented
``"Malaga"``.

Every entry is a Spanish municipality with an AEMET forecast, so the AEMET
benchmark covers the whole registry. Pasadena, the only US station, was removed
on 2026-11-11 (owner decision, audit finding 16): the product is Iberia-only and
the map's camera cannot show it, so it only made "15 estaciones" disagree with
the pins on screen.
"""

from typing import Final

#: ``(name, latitude, longitude)`` for the default stations.
DEFAULT_CITIES: Final[tuple[tuple[str, float, float], ...]] = (
    ("El Ejido", 36.7756, -2.8144),
    ("Almería", 36.8381, -2.4597),
    ("Granada", 37.1773, -3.5986),
    ("Paterna", 39.5028, -0.4408),
    ("Madrid", 40.4168, -3.7038),
    ("Barcelona", 41.3851, 2.1734),
    ("Valencia", 39.4699, -0.3763),
    ("Sevilla", 37.3891, -5.9845),
    ("Zaragoza", 41.6488, -0.8891),
    ("Malaga", 36.7213, -4.4214),
    ("Murcia", 37.9922, -1.1307),
    ("Palma", 39.5696, 2.6502),
    ("Bilbao", 43.2630, -2.9350),
    ("Alicante", 38.3452, -0.4810),
)
