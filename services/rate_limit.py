"""
rate_limit.py
-------------
Cost protection for the AI-generating endpoints.

Three independent guards, all checked only when a request is about to make a
*paid* model call — cache hits are free and never counted against any limit:

  1. Per-client sliding window — stops one caller monopolizing the budget.
  2. Global daily cap          — the hard backstop against a bill spike.
  3. Concurrency limit         — bounds parallel in-flight generations, so a
                                 burst of tabs can't launch N×$ calls at once.

State is in-process. Under multiple uvicorn workers each worker keeps its own
counters, so effective limits multiply by the worker count; for the default
single-process deployment the numbers below are exact. A multi-process or
multi-instance deployment should move this state to Redis.
"""

import hmac
import logging
import os
import threading
import time
from collections import defaultdict, deque
from typing import Optional

logger = logging.getLogger(__name__)

# Tunable via env so limits can be raised without a code change.
PER_CLIENT_MAX = int(os.environ.get("RATE_LIMIT_PER_CLIENT", "8"))
PER_CLIENT_WINDOW_SECONDS = int(os.environ.get("RATE_LIMIT_WINDOW_SECONDS", "3600"))
DAILY_MAX = int(os.environ.get("RATE_LIMIT_DAILY_MAX", "150"))
MAX_CONCURRENT = int(os.environ.get("RATE_LIMIT_MAX_CONCURRENT", "3"))

# Addresses that count as "this machine".
LOOPBACK = {"127.0.0.1", "::1", "localhost", "testclient"}


def _bypass_enabled() -> bool:
    """Whether loopback callers skip the per-client and concurrency limits.

    This is deliberately not a plain "is the caller 127.0.0.1" check, and the
    distinction matters more than it looks. `_client_id` ignores
    X-Forwarded-For on purpose — without a trusted-proxy allowlist any caller
    could spoof it — which means that behind a reverse proxy on the same host,
    *every* production request arrives as 127.0.0.1. An address-only bypass
    would therefore switch off all cost protection the moment the app was
    deployed behind nginx, and do it silently.

    So the bypass also requires that the app is not configured for a public
    origin. Setting PITCHSENSE_BASE_URL to a real domain, which deployment
    requires anyway for canonical links and OG images, disables it on its own.
    RATE_LIMIT_BYPASS_LOCAL forces the decision either way:

        RATE_LIMIT_BYPASS_LOCAL=1  always bypass loopback  (local only)
        RATE_LIMIT_BYPASS_LOCAL=0  never bypass            (safe default in prod)
    """
    override = os.environ.get("RATE_LIMIT_BYPASS_LOCAL")
    if override is not None:
        return override.strip().lower() in ("1", "true", "yes", "on")
    base = os.environ.get("PITCHSENSE_BASE_URL", "")
    return (not base) or any(host in base for host in ("127.0.0.1", "localhost"))


def is_local(client_id: str) -> bool:
    return client_id in LOOPBACK


# Header carrying the founder token on a remote session. The token itself is
# never in the codebase — it comes from the environment.
ADMIN_HEADER = "X-PitchSense-Admin"


def admin_token_valid(supplied: Optional[str]) -> bool:
    """Whether a supplied token matches the configured founder token.

    Two things here are deliberate. The comparison is constant-time, because a
    naive == leaks the token a character at a time to anyone willing to time
    the responses. And an unset ADMIN_TOKEN disables token auth outright rather
    than admitting everyone — a bare header flag with no secret behind it would
    let any visitor switch off every spend limit in the app, which is worse
    than having no admin mode at all.
    """
    expected = os.environ.get("ADMIN_TOKEN") or ""
    if not expected or not supplied:
        return False
    return hmac.compare_digest(supplied, expected)


def is_founder(client_id: str, admin_token: Optional[str] = None) -> bool:
    """Whether this caller is exempt from quotas entirely."""
    if admin_token_valid(admin_token):
        return True
    return is_local(client_id) and _bypass_enabled()


class RateLimitExceeded(Exception):
    """Raised when a paid call would breach a limit. Carries a Retry-After."""

    def __init__(self, message: str, retry_after: int):
        super().__init__(message)
        self.message = message
        self.retry_after = retry_after


_lock = threading.Lock()
_client_hits: dict[str, deque] = defaultdict(deque)
_day_stamp: Optional[int] = None
_day_count = 0
_in_flight = 0


def _prune(hits: deque, now: float) -> None:
    cutoff = now - PER_CLIENT_WINDOW_SECONDS
    while hits and hits[0] < cutoff:
        hits.popleft()


def _roll_day(now: float) -> None:
    """Reset the daily counter when the UTC day changes. Caller holds _lock."""
    global _day_stamp, _day_count
    today = int(now // 86400)
    if _day_stamp != today:
        _day_stamp = today
        _day_count = 0


class guard:
    """Context manager reserving one paid-call slot for `client_id`.

    Raises RateLimitExceeded before the model call is made. The concurrency
    slot is released on exit; the per-client and daily counts are not, since
    they intentionally measure spend over time.
    """

    def __init__(self, client_id: str, label: str = "generation",
                 admin_token: Optional[str] = None):
        self.client_id = client_id
        self.label = label
        self.admin_token = admin_token

    def __enter__(self) -> "guard":
        global _day_count, _in_flight
        now = time.time()
        founder = is_founder(self.client_id, self.admin_token)
        with _lock:
            _roll_day(now)

            # A founder session passes every gate. The spend is still counted,
            # so /health keeps reporting the real number and an expensive
            # afternoon is still visible — it simply never becomes a wall.
            if founder:
                hits = _client_hits[self.client_id]
                _prune(hits, now)
                hits.append(now)
                _day_count += 1
                _in_flight += 1
                return self

            if _day_count >= DAILY_MAX:
                # Seconds until the next UTC midnight.
                retry = int((_day_stamp + 1) * 86400 - now)
                logger.warning(
                    "Daily AI budget exhausted (%d calls); rejecting %s for %s",
                    _day_count, self.label, self.client_id,
                )
                raise RateLimitExceeded(
                    "Daily analysis budget reached. Cached matches are still "
                    "available — try again tomorrow.",
                    max(retry, 1),
                )

            hits = _client_hits[self.client_id]
            _prune(hits, now)
            if len(hits) >= PER_CLIENT_MAX:
                retry = int(hits[0] + PER_CLIENT_WINDOW_SECONDS - now)
                logger.info(
                    "Per-client limit hit for %s (%d in window)",
                    self.client_id, len(hits),
                )
                raise RateLimitExceeded(
                    f"Rate limit reached ({PER_CLIENT_MAX} new analyses per hour). "
                    "Previously analyzed matches remain instant.",
                    max(retry, 1),
                )

            if _in_flight >= MAX_CONCURRENT:
                raise RateLimitExceeded(
                    "Too many analyses running right now — try again shortly.",
                    15,
                )

            # Commit: count the spend and take a concurrency slot.
            hits.append(now)
            _day_count += 1
            _in_flight += 1
        return self

    def __exit__(self, *exc_info) -> None:
        global _in_flight
        with _lock:
            _in_flight = max(0, _in_flight - 1)


def snapshot() -> dict[str, int]:
    """Current budget state, for the health endpoint."""
    now = time.time()
    with _lock:
        _roll_day(now)
        return {
            "daily_used": _day_count,
            "daily_max": DAILY_MAX,
            "daily_remaining": max(0, DAILY_MAX - _day_count),
            "in_flight": _in_flight,
            "max_concurrent": MAX_CONCURRENT,
            "local_bypass": _bypass_enabled(),
            "admin_token_configured": bool(os.environ.get("ADMIN_TOKEN")),
        }
