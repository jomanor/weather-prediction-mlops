"""Seed data for the station registry.

These are the stations the original Open-Meteo ingest has always covered, plus
Pasadena. On startup the API inserts them into the ``cities`` collection when it
is empty — that is all this tuple is for. Once seeded, the Mongo registry is the
only runtime source of the station set: the ingest, the backfill, the quality
meter and the analytics aggregates all read their cities from it, so adding or
deleting a station through ``/cities`` immediately changes what they ingest and
report. The tuple is also the fallback when the registry is empty or unreadable.
Names are kept byte-for-byte as that writer used them, including the unaccented
``"Malaga"``.

Pasadena is the one US station: it is fed by the Open-Meteo ingest (not by
AEMET, which has no US coverage), so the AEMET benchmark reports no data for it.
It stays in the tuple because the tuple is what seeds a fresh database and what
the monitors fall back to when the registry is unreadable: dropping it would
silently narrow the station set in both cases.
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
    ("Pasadena", 34.1478, -118.1445),
)
