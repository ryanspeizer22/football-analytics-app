"""
textclean.py
------------
Repairs encoding artefacts in model-generated prose.

Structured output is JSON, and a model will occasionally emit an escape
sequence as literal characters rather than as the character it denotes — so a
field arrives holding the six characters `\\u2014` instead of an em dash. The
JSON is still valid, which is why nothing upstream catches it; the artefact
only becomes visible when the string is rendered.

Cleaning runs in two places on purpose: on generation, so nothing new is
cached dirty, and again on the way out, so text cached before this existed is
repaired without paying to regenerate it.
"""

import re
import unicodedata
from typing import Any

# A literal escape sequence sitting in decoded text. Matched with a single
# leading backslash: by the time a string reaches here, JSON decoding has
# already happened, so `\\u2014` in the file is `—` in memory.
_ESCAPE = re.compile(r"\\u([0-9a-fA-F]{4})")

# Other escapes that show up the same way.
_LITERALS = {
    r"\n": "\n",
    r"\t": " ",
    r"\r": "",
    r"\/": "/",
    r"\"": '"',
    r"\'": "'",
}

_SURROGATE = re.compile(r"[\ud800-\udfff]")


def _decode_escapes(text: str) -> str:
    out = _ESCAPE.sub(lambda m: chr(int(m.group(1), 16)), text)
    # An emoji written out as an escape decodes to a surrogate pair, which is
    # not a character Python can render. Re-pair them into the real codepoint.
    if _SURROGATE.search(out):
        out = out.encode("utf-16", "surrogatepass").decode("utf-16", "replace")
    for literal, replacement in _LITERALS.items():
        if literal in out:
            out = out.replace(literal, replacement)
    return out


def clean_text(text: str) -> str:
    """Repair escape artefacts and normalise whitespace in one string."""
    if not text:
        return text
    out = _decode_escapes(text)
    # NFC keeps accented names in their composed form, so "Ødegaard" compares
    # and renders consistently wherever it appears.
    out = unicodedata.normalize("NFC", out)
    out = out.replace(" ", " ")
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out.strip()


def clean(value: Any) -> Any:
    """Clean every string in a nested structure, leaving its shape untouched."""
    if isinstance(value, str):
        return clean_text(value)
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value
