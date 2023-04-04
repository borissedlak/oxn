import re
from typing import Callable
from datetime import datetime
from datetime import timezone

import functools


SECONDS_MAP = {
    "us": 1 / 10**6,
    "ms": 1 / 10**3,
    "s": 1,
    "m": 60,
    "h": 3600,
    "d": 86400,
}
"""Map used to convert time strings to seconds"""

time_string_format_regex = r"(\d+)(us|ms|s|m|h|d)"


def validate_time_string(time_string):
    """
    Validate that a time string has units

    """
    return bool(re.match(time_string_format_regex, time_string))


def time_string_to_seconds(time_string) -> float:
    """Convert a time string with units to a float"""
    matches = re.findall(time_string_format_regex, time_string)
    seconds = 0.0
    for match in matches:
        value_part = match[0]
        unit_part = match[1]
        seconds += float(value_part) * SECONDS_MAP[unit_part]
    return seconds


def to_milliseconds(seconds):
    """Convert seconds to milliseconds"""
    return seconds * 10**3


def to_microseconds(seconds):
    """Convert seconds to microseconds"""
    return seconds * 10**6


def utc_timestamp() -> float:
    """Get the current time in """
    return datetime.now(timezone.utc).timestamp()


def humanize_utc_timestamp(timestamp):
    """Return a human-readable version of a timestamp"""
    return datetime.utcfromtimestamp(timestamp)


def defer_cleanup(func) -> Callable:
    """
    Decorator to tell the experiment runner to only cleanup after the experiment has run

    We provide this decorator to allow deferred cleanup for some treatments where we cannot immediately
    clean the treatment as this would disallow the treatment to function properly. For an example treatment
    where this applies, check the PrometheusScrapeTreatment in treatments.py
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    wrapper.defer_cleanup = True
    return wrapper
