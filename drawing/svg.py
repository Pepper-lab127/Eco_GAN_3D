import math
import re
from xml.sax.saxutils import escape


# Paper sizes in mm (landscape).
PAPER = {
    "A1": (841.0, 594.0),
    "A2": (594.0, 420.0),
    "A3": (420.0, 297.0),
    "ARCHD": (914.4, 609.6),    # 36 x 24 in
    "ARCHD_P": (609.6, 914.4),  # 36 x 24 in, portrait
    "ARCHE": (1219.2, 914.4),   # 48 x 36 in
}

FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"


def _fmt(v):
    return f"{v:.3f}".rstrip("0").rstrip(".")


def _attrs(**kw):
    out = []
    for k, v in kw.items():
        if v is None:
            continue
        k = k.rstrip("_").replace("_", "-")
        if isinstance(v, float):
            v = _fmt(v)
        out.append(f'{k}="{escape(str(v))}"')
    return " ".join(out)


def points_to_path(xs, ys, close_to=None):
    """Polyline path. Breaks at NaN. If close_to is a y value, each run is
    closed down to that baseline so it can be filled (hidden-line occlusion)."""
    parts = []
    run = []

    def flush():
        if len(run) < 2:
            run.clear()
            return
        d = "M" + " L".join(f"{_fmt(x)},{_fmt(y)}" for x, y in run)
        if close_to is not None:
            d += f" L{_fmt(run[-1][0])},{_fmt(close_to)} L{_fmt(run[0][0])},{_fmt(close_to)} Z"
        parts.append(d)
        run.clear()

    for x, y in zip(xs, ys):
        if math.isnan(y):
            flush()
        else:
            run.append((x, y))
    flush()
    return " ".join(parts)


class Svg:
    """Minimal SVG writer working in millimetres.

    Drawing goes into named layers (see layer()), written as top-level groups
    that Illustrator and Inkscape open as layers, in the order first used.
    """

    def __init__(self, width, height, background="#ffffff"):
        self.w = width
        self.h = height
        self.defs = []
        self._base = []
        self._layers = {}
        self.body = self._base
        self._ids = 0
        if background:
            self.layer("PAPER")
            self.rect(0, 0, width, height, fill=background)
            self.layer(None)

    def layer(self, name):
        """Send what follows to the named layer (None: outside any layer).
        Returning to a layer appends to it; its stacking stays where it began."""
        self.body = self._base if name is None else self._layers.setdefault(name, [])

    def uid(self, prefix="id"):
        self._ids += 1
        return f"{prefix}{self._ids}"

    def raw(self, s):
        self.body.append(s)

    def open_group(self, **kw):
        self.body.append(f"<g {_attrs(**kw)}>")

    def close_group(self):
        self.body.append("</g>")

    def clip_rect(self, x, y, w, h):
        cid = self.uid("clip")
        self.defs.append(
            f'<clipPath id="{cid}"><rect {_attrs(x=x, y=y, width=w, height=h)}/></clipPath>'
        )
        return f"url(#{cid})"

    def clip_path(self, d):
        cid = self.uid("clip")
        self.defs.append(f'<clipPath id="{cid}"><path d="{d}"/></clipPath>')
        return f"url(#{cid})"

    def rect(self, x, y, w, h, **kw):
        self.body.append(f"<rect {_attrs(x=x, y=y, width=w, height=h, **kw)}/>")

    def line(self, x1, y1, x2, y2, stroke="#000", stroke_width=0.1, **kw):
        self.body.append(
            f"<line {_attrs(x1=x1, y1=y1, x2=x2, y2=y2, stroke=stroke, stroke_width=stroke_width, **kw)}/>"
        )

    def path(self, d, **kw):
        if d:
            self.body.append(f"<path {_attrs(d=d, **kw)}/>")

    def image(self, x, y, w, h, href, **kw):
        self.body.append(
            f'<image {_attrs(x=x, y=y, width=w, height=h, preserveAspectRatio="none", **kw)} href="{href}"/>'
        )

    def circle(self, cx, cy, r, **kw):
        self.body.append(f"<circle {_attrs(cx=cx, cy=cy, r=r, **kw)}/>")

    def text(self, x, y, s, size=2.2, fill="#000", anchor="start", weight=None,
             spacing=None, rotate=None, **kw):
        transform = f"rotate({_fmt(rotate)} {_fmt(x)} {_fmt(y)})" if rotate else None
        self.body.append(
            f"<text {_attrs(x=x, y=y, font_family=FONT, font_size=size, fill=fill, text_anchor=anchor, font_weight=weight, letter_spacing=spacing, transform=transform, **kw)}>"
            f"{escape(str(s))}</text>"
        )

    def save(self, path):
        head = (
            '<svg xmlns="http://www.w3.org/2000/svg" '
            'xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" '
            f'width="{_fmt(self.w)}mm" height="{_fmt(self.h)}mm" '
            f'viewBox="0 0 {_fmt(self.w)} {_fmt(self.h)}">'
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(head)
            if self.defs:
                f.write("<defs>" + "".join(self.defs) + "</defs>")
            f.write("\n".join(self._base))
            used = set()
            for name, items in self._layers.items():
                if not items:
                    continue
                lid = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_") or "layer"
                while lid in used:
                    lid += "_"
                used.add(lid)
                f.write(f'\n<g id="{lid}" inkscape:groupmode="layer" inkscape:label="{escape(name)}">\n')
                f.write("\n".join(items))
                f.write("\n</g>")
            f.write("</svg>\n")


# ---------------------------------------------------------------------------
# Annotation vocabulary shared by all sheets
# ---------------------------------------------------------------------------


def scale_bar(svg, x, y, length_mm, label_end, units, color="#000", segments=4):
    """Alternating-block scale bar, labelled 0 and label_end."""
    seg = length_mm / segments
    for i in range(segments):
        svg.rect(x + i * seg, y, seg, 0.8,
                 fill=color if i % 2 == 0 else "none", stroke=color, stroke_width=0.12)
    svg.text(x, y + 3.2, "0", size=1.8, fill=color, anchor="middle")
    svg.text(x + length_mm, y + 3.2, f"{label_end} {units}", size=1.8, fill=color, anchor="middle")


def north_arrow(svg, cx, cy, r=3.0, color="#000", angle=0.0):
    """angle: true north in degrees clockwise from the top of the sheet."""
    svg.circle(cx, cy, r, fill="none", stroke=color, stroke_width=0.15)
    svg.open_group(transform=f"rotate({_fmt(angle)} {_fmt(cx)} {_fmt(cy)})")
    svg.line(cx, cy + r, cx, cy - r, stroke=color, stroke_width=0.15)
    svg.text(cx, cy - r - 0.8, "N", size=1.8, fill=color, anchor="middle")
    svg.close_group()


def flag(svg, x, y, label, dx=12.0, color="#000", size=1.9, side=1, line_layer=None, text_layer=None):
    """Leader line from a data point to a small text flag (as on p.238).
    Optionally puts the line and the text on named layers."""
    ex = x + dx * side
    if line_layer:
        svg.layer(line_layer)
    svg.line(x, y, ex, y, stroke=color, stroke_width=0.08)
    svg.path(f"M{_fmt(ex)},{_fmt(y - 1.2)} L{_fmt(ex + 1.6 * side)},{_fmt(y)} L{_fmt(ex)},{_fmt(y + 1.2)}",
             fill="none", stroke=color, stroke_width=0.08)
    if text_layer:
        svg.layer(text_layer)
    svg.text(ex + 2.4 * side, y + size * 0.35, label, size=size, fill=color,
             anchor="start" if side > 0 else "end")
