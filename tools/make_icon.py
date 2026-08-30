r"""Generate assets/subordinant.ico - the window, taskbar and tray icon.

    .venv\Scripts\python.exe -m tools.make_icon

The mark is the thing the app puts on screen: a dark caption plate with a speech
tail and two caption bars. The lower bar is drawn in the accent colour rather
than white because that is what the overlay itself does - committed text solid,
the tail that has not settled yet dimmed - so the icon reads as "captions, still
arriving" instead of just "a rectangle".

This script is committed next to the .ico it produces so the icon can be retuned
without hand-editing a binary. Nothing at runtime imports it - the app loads the
generated file - so its Pillow dependency is deliberately absent from
requirements.txt, where it would only be dead weight. Pillow is currently in the
venv incidentally, as a matplotlib dependency, and matplotlib is excluded from
the bundle; if the import below fails, pip install pillow.

Every size in the ICO is drawn from geometry rather than downsampled from one
large bitmap, because Windows picks a size per context - 16 px in the taskbar,
256 px in Explorer's jumbo view - and detail that survives at 256 does not
survive being resampled to 16. See _geometry for what changes and why.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

# Drawn at SS times the target and downsampled, which is what antialiases the
# rounded corners and the tail. Higher costs only build time.
SS = 8

GRAD_TOP = (74, 92, 135)
GRAD_BOTTOM = (30, 38, 58)
# A rim, because the plate is dark and so is the Windows taskbar it sits on.
# Without it the icon is a hole rather than a shape.
RIM = (255, 255, 255, 58)
RIM_WIDTH = 0.016
BAR_COMMITTED = (255, 255, 255, 255)
BAR_SETTLING = (94, 214, 203, 255)

SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)

# Below this the tail is a two-pixel stub that reads as a rendering artifact, so
# the small sizes drop it and grow the plate into the space instead.
TAIL_MIN_SIZE = 32
# Below this, bar edges are snapped to the target pixel grid.
SNAP_MAX_SIZE = 32


def _geometry(size: int) -> dict:
    """Layout for one icon size, as fractions of the canvas edge.

    Two variants, not a scale factor. The tailed plate is the real mark; the
    small one keeps its proportions - same bar rhythm, same margins - with the
    tail removed and the plate re-centred, so the two read as the same icon at
    the sizes where they are never seen side by side.
    """
    if size >= TAIL_MIN_SIZE:
        return {
            "plate": (0.045, 0.085, 0.955, 0.720),
            "radius": 0.150,
            "tail": ((0.235, 0.655), (0.515, 0.655), (0.285, 0.930)),
            "bar_left": 0.150,
            "bar_top": (0.255, 0.455),
            "bar_right": (0.850, 0.585),
            "bar_height": 0.090,
        }
    return {
        "plate": (0.030, 0.105, 0.970, 0.895),
        "radius": 0.190,
        "tail": None,
        # Wider side margins: at 16 px the fractions above leave barely a pixel
        # between bar and rim, and the two merge into grey.
        "bar_left": 0.185,
        "bar_top": (0.290, 0.560),
        "bar_right": (0.815, 0.600),
        "bar_height": 0.150,
    }


def _gradient(px: int) -> Image.Image:
    """Vertical GRAD_TOP -> GRAD_BOTTOM ramp."""
    column = Image.new("RGB", (1, px))
    for y in range(px):
        t = y / max(px - 1, 1)
        column.putpixel(
            (0, y),
            tuple(round(a + (b - a) * t) for a, b in zip(GRAD_TOP, GRAD_BOTTOM)),
        )
    return column.resize((px, px), Image.Resampling.NEAREST)


def _silhouette(px: int, geom: dict, inset: float) -> Image.Image:
    """Plate plus tail as one mask, optionally shrunk by `inset` pixels.

    The inset copy is what makes the rim: subtracting it from the full
    silhouette leaves a band that follows the whole outline, corners included,
    without stroking the two pieces separately and having to hide the seam where
    the tail meets the plate.
    """
    mask = Image.new("L", (px, px), 0)
    draw = ImageDraw.Draw(mask)

    left, top, right, bottom = (v * px for v in geom["plate"])
    draw.rounded_rectangle(
        (left + inset, top + inset, right - inset, bottom - inset),
        radius=max(geom["radius"] * px - inset, 0),
        fill=255,
    )

    if geom["tail"]:
        tail = [(x * px, y * px) for x, y in geom["tail"]]
        if inset:
            # Shrink toward the centroid. Approximate for a triangle, but the
            # tail's top edge is buried under the plate, so only the two visible
            # sides matter and both move the right way.
            cx = sum(x for x, _ in tail) / 3
            cy = sum(y for _, y in tail) / 3
            span = max(max(abs(x - cx) for x, _ in tail), max(abs(y - cy) for _, y in tail))
            k = max(1 - inset / span, 0.0) if span else 0.0
            tail = [(cx + (x - cx) * k, cy + (y - cy) * k) for x, y in tail]
        draw.polygon(tail, fill=255)

    return mask


def draw_icon(size: int) -> Image.Image:
    geom = _geometry(size)
    px = size * SS
    icon = Image.new("RGBA", (px, px), (0, 0, 0, 0))

    full = _silhouette(px, geom, 0)
    icon.paste(_gradient(px), (0, 0), full)

    # Never thinner than one target pixel, or the rim disappears at 16 px -
    # exactly the size that needs it most.
    inner = _silhouette(px, geom, max(RIM_WIDTH * px, SS))
    band = Image.composite(full, Image.new("L", (px, px), 0), inner.point(lambda v: 255 - v))
    icon.paste(Image.new("RGBA", (px, px), RIM), (0, 0), band)

    bars = ImageDraw.Draw(icon)
    pairs = zip(geom["bar_top"], geom["bar_right"], (BAR_COMMITTED, BAR_SETTLING))
    for top, right, colour in pairs:
        x0, x1 = geom["bar_left"] * size, right * size
        y0, y1 = top * size, (top + geom["bar_height"]) * size
        if size <= SNAP_MAX_SIZE:
            # Snap to the target pixel grid. Downsampling an unsnapped 2.4 px bar
            # smears it across three rows and the icon reads as grey mush.
            x0, y0 = round(x0), round(y0)
            x1, y1 = max(round(x1), x0 + 1), max(round(y1), y0 + 1)
        bars.rounded_rectangle(
            (x0 * SS, y0 * SS, x1 * SS - 1, y1 * SS - 1),
            radius=(y1 - y0) * SS / 2,
            fill=colour,
        )

    return icon.resize((size, size), Image.Resampling.LANCZOS)


def build(out: Path) -> None:
    frames = [draw_icon(size) for size in SIZES]
    out.parent.mkdir(parents=True, exist_ok=True)
    # Pillow writes each appended image as its own directory entry, so one file
    # carries all nine and Windows picks the right one per context.
    frames[-1].save(
        out,
        format="ICO",
        sizes=[(s, s) for s in SIZES],
        append_images=frames[:-1],
    )
    print(f"wrote {out} ({', '.join(str(s) for s in SIZES)} px)")


if __name__ == "__main__":
    build(Path(__file__).resolve().parent.parent / "assets" / "subordinant.ico")
