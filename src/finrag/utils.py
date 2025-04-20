# src/finrag/utils.py
import os
import unicodedata
import re

def clean_env(key):
    """Cleans environment variable values: handles None, normalizes Unicode, replaces dashes."""
    val = os.getenv(key, "")
    if val is None:
        return ""
    # Normalize to ASCII compatible characters
    val = unicodedata.normalize("NFKC", str(val))
    # Replace any kind of Unicode dash with ASCII hyphen
    val = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]", "-", val)
    return val.strip()
