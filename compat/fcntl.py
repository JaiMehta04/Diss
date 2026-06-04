"""Minimal Windows stub for fcntl to satisfy blessings on Earth Engine.
This avoids import errors on Windows; functions are no-ops.
"""

def ioctl(*_args, **_kwargs):
    return 0


def fcntl(*_args, **_kwargs):
    return 0


def flock(*_args, **_kwargs):
    return 0


def lockf(*_args, **_kwargs):
    return 0
