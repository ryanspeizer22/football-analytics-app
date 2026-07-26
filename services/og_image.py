"""
og_image.py
-----------
Renders the branded preview cards that appear when a link is shared into
Slack, iMessage, WhatsApp, X or Discord.

Two constraints shape this module:

* **Crawlers don't run JavaScript.** The card has to be a real raster image at
  a real URL, not something the page draws on load — hence server-side
  composition rather than reusing the front-end.
* **Fonts are not portable.** macOS ships Arial/Helvetica; `python:3.12-slim`
  ships none at all, and Pillow silently falls back to a tiny bitmap font that
  would make every shared card look broken. So fonts are resolved from a
  candidate chain and the Dockerfile installs DejaVu explicitly.
"""

import logging
from io import BytesIO
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFilter, ImageFont

logger = logging.getLogger(__name__)

WIDTH, HEIGHT = 1200, 630  # the size every major unfurler expects
STATIC = Path(__file__).resolve().parent.parent / "static"
BACKDROP = STATIC / "stadium-campnou.jpg"

# Checked in order; DejaVu is what the container installs.
_BOLD_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]
_REGULAR_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]

NEON = (55, 240, 162)
INK = (238, 241, 248)
DIM = (150, 160, 180)


def _font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    for path in (_BOLD_CANDIDATES if bold else _REGULAR_CANDIDATES):
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    logger.warning("No TrueType font found — OG cards will use the bitmap fallback")
    return ImageFont.load_default()


def _hex_rgb(value: Optional[str], fallback=(120, 130, 150)) -> tuple[int, int, int]:
    if not value or not value.startswith("#") or len(value) != 7:
        return fallback
    try:
        return tuple(int(value[i:i + 2], 16) for i in (1, 3, 5))  # type: ignore[return-value]
    except ValueError:
        return fallback


def _backdrop() -> Image.Image:
    """Darkened stadium base, or a flat gradient if the photo is missing."""
    base = Image.new("RGB", (WIDTH, HEIGHT), (7, 9, 15))
    if BACKDROP.exists():
        try:
            photo = Image.open(BACKDROP).convert("RGB")
            ratio = max(WIDTH / photo.width, HEIGHT / photo.height)
            photo = photo.resize(
                (int(photo.width * ratio), int(photo.height * ratio)), Image.LANCZOS
            )
            left = (photo.width - WIDTH) // 2
            top = int((photo.height - HEIGHT) * 0.45)
            photo = photo.crop((left, top, left + WIDTH, top + HEIGHT))
            photo = photo.filter(ImageFilter.GaussianBlur(2))
            base = Image.blend(base, photo, 0.42)
        except Exception:
            logger.exception("OG backdrop failed; using flat background")
    # Darken toward the bottom so text always has a floor to sit on.
    veil = Image.new("L", (1, HEIGHT))
    for y in range(HEIGHT):
        veil.putpixel((0, y), int(90 + 130 * (y / HEIGHT)))
    veil = veil.resize((WIDTH, HEIGHT))
    return Image.composite(Image.new("RGB", (WIDTH, HEIGHT), (5, 7, 12)), base, veil)


def _brand(draw: ImageDraw.ImageDraw) -> None:
    """Logo mark. The bolt is drawn as a polygon rather than set as ⚡ — text
    fonts carry no emoji glyphs, so the character renders as a hollow box."""
    draw.rounded_rectangle([64, 52, 104, 92], 12, fill=NEON)
    draw.polygon(
        [(89, 58), (75, 75), (83, 75), (79, 88), (93, 70), (85, 70)],
        fill=(4, 19, 11),
    )
    draw.text((118, 58), "PITCHSENSE", font=_font(28), fill=INK)


def _fit(draw: ImageDraw.ImageDraw, text: str, font_size: int, max_w: int) -> ImageFont.FreeTypeFont:
    """Shrink until the string fits — club names vary wildly in length."""
    size = font_size
    while size > 18:
        font = _font(size)
        if draw.textlength(text, font=font) <= max_w:
            return font
        size -= 4
    return _font(18)


def _png(image: Image.Image) -> bytes:
    buf = BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def match_card(home: str, away: str, home_score, away_score,
               competition: str, date: str,
               home_color: str, away_color: str, headline: str = "") -> bytes:
    img = _backdrop()
    draw = ImageDraw.Draw(img)
    _brand(draw)

    hc, ac = _hex_rgb(home_color), _hex_rgb(away_color)
    # Club-colour bar across the top, split down the middle.
    draw.rectangle([0, 0, WIDTH // 2, 8], fill=hc)
    draw.rectangle([WIDTH // 2, 0, WIDTH, 8], fill=ac)

    draw.text((64, 150), competition.upper(), font=_font(22, bold=False), fill=DIM)
    draw.text((64, 182), date, font=_font(20, bold=False), fill=DIM)

    played = home_score is not None and away_score is not None
    score = f"{home_score} – {away_score}" if played else "vs"
    score_font = _font(96 if played else 64)
    score_w = draw.textlength(score, font=score_font)
    draw.text(((WIDTH - score_w) / 2, 250), score, font=score_font, fill=INK)

    name_font = _fit(draw, home, 52, 420)
    draw.text((64, 268), home, font=name_font, fill=hc)
    name_font = _fit(draw, away, 52, 420)
    away_w = draw.textlength(away, font=name_font)
    draw.text((WIDTH - 64 - away_w, 268), away, font=name_font, fill=ac)

    if headline:
        hf = _fit(draw, headline, 34, WIDTH - 128)
        draw.text((64, 470), headline, font=hf, fill=INK)

    draw.text((64, HEIGHT - 70), "AI match intelligence · pitchsense",
              font=_font(20, bold=False), fill=DIM)
    return _png(img)


def compare_card(a_name: str, a_rating, a_team: str,
                 b_name: str, b_rating, b_team: str, season: int) -> bytes:
    img = _backdrop()
    draw = ImageDraw.Draw(img)
    _brand(draw)

    draw.rectangle([0, 0, WIDTH // 2, 8], fill=NEON)
    draw.rectangle([WIDTH // 2, 0, WIDTH, 8], fill=(96, 165, 250))

    draw.text((64, 150), f"HEAD TO HEAD · {season}/{str(season + 1)[-2:]}",
              font=_font(22, bold=False), fill=DIM)

    def side(name: str, rating, team: str, x_center: int, color) -> None:
        nf = _fit(draw, name, 46, 460)
        w = draw.textlength(name, font=nf)
        draw.text((x_center - w / 2, 250), name, font=nf, fill=INK)
        tf = _font(24, bold=False)
        tw = draw.textlength(team or "", font=tf)
        draw.text((x_center - tw / 2, 310), team or "", font=tf, fill=DIM)
        val = f"{rating:.2f}" if isinstance(rating, (int, float)) else "—"
        vf = _font(78)
        vw = draw.textlength(val, font=vf)
        draw.text((x_center - vw / 2, 360), val, font=vf, fill=color)

    side(a_name, a_rating, a_team, 320, NEON)
    side(b_name, b_rating, b_team, WIDTH - 320, (96, 165, 250))

    vs_font = _font(40)
    vs_w = draw.textlength("VS", font=vs_font)
    draw.text(((WIDTH - vs_w) / 2, 330), "VS", font=vs_font, fill=DIM)

    draw.text((64, HEIGHT - 70), "Season average rating · pitchsense",
              font=_font(20, bold=False), fill=DIM)
    return _png(img)


def generic_card(title: str, subtitle: str = "") -> bytes:
    img = _backdrop()
    draw = ImageDraw.Draw(img)
    _brand(draw)
    draw.rectangle([0, 0, WIDTH, 8], fill=NEON)
    tf = _fit(draw, title, 64, WIDTH - 128)
    draw.text((64, 280), title, font=tf, fill=INK)
    if subtitle:
        draw.text((64, 380), subtitle, font=_font(30, bold=False), fill=DIM)
    draw.text((64, HEIGHT - 70), "AI match intelligence · pitchsense",
              font=_font(20, bold=False), fill=DIM)
    return _png(img)
