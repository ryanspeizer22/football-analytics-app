"""Bake the CSS filter chain into the stadium asset.

The ambient photo carries `filter: contrast(1.18) brightness(0.78) saturate(1.62)`
in CSS. That is a live filter on a full-viewport fixed layer, so every scale
change re-rasterises the whole thing. Applying the exact same maths once, here,
gives identical pixels for free at runtime.

The CSS filter functions are specified precisely, so this reproduces them
rather than using Pillow's ImageEnhance (which uses different formulas).

Run from anywhere:  python scripts/grade_stadium.py

Keep these constants in step with the comment in index.html. If the grade ever
needs to change, change it here and re-run — do not put the filter back in CSS.
"""
from pathlib import Path

from PIL import Image

STATIC = Path(__file__).resolve().parent.parent / "static"
SRC = STATIC / "stadium-campnou.jpg"
DST = STATIC / "stadium-campnou-graded.jpg"

CONTRAST, BRIGHTNESS, SATURATE = 1.18, 0.78, 1.62

img = Image.open(SRC).convert("RGB")
print(f"source: {img.size[0]}x{img.size[1]}")

# contrast(k): (c - 0.5) * k + 0.5   then   brightness(k): c * k
# Both are per-channel scalar maps, so they compose into a single LUT.
lut = []
for v in range(256):
    c = v / 255.0
    c = (c - 0.5) * CONTRAST + 0.5
    c = c * BRIGHTNESS
    lut.append(max(0, min(255, round(c * 255))))
img = img.point(lut * 3)

# saturate(s), per the filter-effects spec's colour matrix.
s = SATURATE
matrix = (
    0.213 + 0.787 * s, 0.715 - 0.715 * s, 0.072 - 0.072 * s, 0,
    0.213 - 0.213 * s, 0.715 + 0.285 * s, 0.072 - 0.072 * s, 0,
    0.213 - 0.213 * s, 0.715 - 0.715 * s, 0.072 + 0.928 * s, 0,
)
img = img.convert("RGB", matrix)

img.save(DST, "JPEG", quality=86, optimize=True, progressive=True)

import os
print(f"wrote {DST}")
print(f"  original {os.path.getsize(SRC)/1024:.0f}KB -> graded {os.path.getsize(DST)/1024:.0f}KB")
