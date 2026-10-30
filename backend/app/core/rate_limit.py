"""In-process rate limiting for the optional refresh endpoint.

The caps are per process lifetime: state lives in this object's memory and is
lost on restart, redeploy or a free-tier idle spin-down. After the instance
wakes, the counters start from zero. This is a storage guard against a caller
hammering one live process — not a distributed quota, and it does not bound
writes across a billing period. Enabling the endpoint is therefore an explicit
operator decision (``REFRESH_ENABLED=true``), not a default. The state is
guarded by an ``asyncio.Lock``, so concurrent requests cannot both observe a
free slot and then both consume it.

Two limits apply per accepted attempt:

* a per-city cooldown (default 900 s) — one upstream fetch per station per
  15 minutes;
* a global cap of 6/minute and 48/day. The daily cap is a storage guard, not a
  throughput target: every refresh writes one document to ``raw_weather``, the
  largest collection on the 512 MB Atlas free tier, so a client hammering the
  endpoint must not be able to grow it without bound.

Rejections do not consume a slot. ``acquire`` reserves on the attempt, so a
failing upstream still counts against the caps — the upstream is what the
limiter protects.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable
from dataclasses import dataclass

from fastapi import Request

MINUTE_SECONDS = 60.0
DAY_SECONDS = 86400.0


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    allowed: bool
    #: Seconds the caller should wait before retrying; 0 when allowed.
    retry_after: int = 0


def _retry_after(seconds: float) -> int:
    """Integer seconds, never below 1: ``Retry-After: 0`` invites a hot retry loop."""
    return max(1, math.ceil(seconds))


class RefreshRateLimiter:
    def __init__(
        self,
        *,
        cooldown_seconds: int,
        max_per_minute: int,
        max_per_day: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._cooldown = max(0, cooldown_seconds)
        self._max_per_minute = max(1, max_per_minute)
        self._max_per_day = max(1, max_per_day)
        self._clock = clock
        self._lock = asyncio.Lock()
        self._last_by_city: dict[str, float] = {}
        start = clock()
        self._minute_start = start
        self._minute_count = 0
        self._day_start = start
        self._day_count = 0

    def _roll(self, now: float) -> None:
        if now - self._minute_start >= MINUTE_SECONDS:
            self._minute_start = now
            self._minute_count = 0
        if now - self._day_start >= DAY_SECONDS:
            self._day_start = now
            self._day_count = 0
            # City names are registry-validated, but prune anyway so a long-lived
            # process cannot accumulate an unbounded map.
            self._last_by_city = {
                city: last
                for city, last in self._last_by_city.items()
                if now - last < self._cooldown
            }

    async def acquire(self, city: str) -> RateLimitDecision:
        """Reserve one attempt for ``city``, or explain why it was rejected."""
        async with self._lock:
            now = self._clock()
            self._roll(now)

            last = self._last_by_city.get(city)
            if last is not None:
                remaining = self._cooldown - (now - last)
                if remaining > 0:
                    return RateLimitDecision(False, _retry_after(remaining))

            if self._minute_count >= self._max_per_minute:
                return RateLimitDecision(
                    False, _retry_after(self._minute_start + MINUTE_SECONDS - now)
                )
            if self._day_count >= self._max_per_day:
                return RateLimitDecision(False, _retry_after(self._day_start + DAY_SECONDS - now))

            self._last_by_city[city] = now
            self._minute_count += 1
            self._day_count += 1
            return RateLimitDecision(True)


def get_refresh_limiter(request: Request) -> RefreshRateLimiter:
    return request.app.state.refresh_limiter
