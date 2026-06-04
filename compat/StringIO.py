"""Compatibility shim for modules expecting StringIO on Windows/Python 3."""
from io import StringIO  # re-export for callers
