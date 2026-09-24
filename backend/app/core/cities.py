"""Default station registry.

These are the 14 stations the Kafka producer has always ingested. On startup the
API seeds the ``cities`` collection with them when it is empty. Names are kept
byte-for-byte as the producer used them, including the unaccented ``"Malaga"``.
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
