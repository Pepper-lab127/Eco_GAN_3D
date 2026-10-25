"""Make sheet SVGs lighter for Illustrator without changing what prints.

Illustrator slows with anchor count and file size, not with what the
drawing shows. Each path's `d` is rewritten:

  - Douglas-Peucker on every straight-line run, at a tolerance well under
    what an Arch D plot can resolve (default 0.01 mm; a 1200 dpi plotter
    dot is 0.021 mm). Points that sit on the line between their neighbours
    to within that distance carry no drawn information and are dropped.
    Ends of runs, corners, curves and single-point marks (dots) are kept.
  - Coordinates on a 0.01 mm grid, written relative, with repeated commands
    and separators omitted.

Layers, groups, styles, dots and text are untouched.

    python -m tools.slim_svg in.svg [out.svg] [--tol 0.01]
"""

import argparse
import re
import sys

import numpy as np

TOK = re.compile(r"[MmLlHhVvQqZz]|-?(?:\d+\.?\d*|\.\d+)(?:e-?\d+)?")
D_ATTR = re.compile(r'(\sd=")([^"]*)(")')


def parse(d):
    """Subpaths as lists of segments: ('L', [(x, y)...]), ('Q', [c, p]...).
    Returns [(start, [segments], closed)] in absolute coordinates."""
    toks = TOK.findall(d)
    i, cmd = 0, None
    cx = cy = sx = sy = 0.0
    subs, cur = [], None

    def num():
        nonlocal i
        v = float(toks[i])
        i += 1
        return v

    while i < len(toks):
        t = toks[i]
        if t.isalpha():
            cmd = t
            i += 1
            if cmd in "Zz":
                if cur is not None:
                    cur[2] = True
                cx, cy = sx, sy
                cur = None
                continue
        elif cmd is None:
            raise ValueError("path data starts with a number")
        rel = cmd.islower()
        c = cmd.upper()
        if c == "M":
            x, y = num(), num()
            if rel:
                x, y = cx + x, cy + y
            cx, cy = sx, sy = x, y
            cur = [(x, y), [], False]
            subs.append(cur)
            cmd = "l" if rel else "L"            # implicit lineto after moveto
            continue
        if cur is None:                          # drawing after Z starts at the old start
            cur = [(cx, cy), [], False]
            subs.append(cur)
        if c in "LHV":
            if c == "L":
                x, y = num(), num()
                if rel:
                    x, y = cx + x, cy + y
            elif c == "H":
                x, y = num() + (cx if rel else 0), cy
            else:
                x, y = cx, num() + (cy if rel else 0)
            segs = cur[1]
            if segs and segs[-1][0] == "L":
                segs[-1][1].append((x, y))
            else:
                segs.append(("L", [(x, y)]))
            cx, cy = x, y
        elif c == "Q":
            x1, y1, x, y = num(), num(), num(), num()
            if rel:
                x1, y1, x, y = cx + x1, cy + y1, cx + x, cy + y
            cur[1].append(("Q", [(x1, y1), (x, y)]))
            cx, cy = x, y
    return subs


def dp(pts, tol):
    """Douglas-Peucker keep-mask for an (n, 2) array."""
    n = len(pts)
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        p, q = pts[a], pts[b]
        seg = pts[a + 1:b]
        dx, dy = q - p
        L = np.hypot(dx, dy)
        if L < 1e-12:
            dist = np.hypot(*(seg - p).T)
        else:
            dist = np.abs(dx * (seg[:, 1] - p[1]) - dy * (seg[:, 0] - p[0])) / L
        k = int(np.argmax(dist))
        if dist[k] > tol:
            m = a + 1 + k
            keep[m] = True
            stack += [(a, m), (m, b)]
    return keep


def fmt(v):
    """Hundredths (int) as compact text: 120 -> 1.2, -5 -> -.05, 0 -> 0."""
    s = "-" if v < 0 else ""
    v = abs(v)
    whole, frac = divmod(v, 100)
    f = f"{frac:02d}".rstrip("0")
    if not f:
        return s + str(whole)
    return s + (str(whole) if whole else "") + "." + f


class Writer:
    def __init__(self):
        self.out = []
        self.last = None        # last command letter written
        self.px = self.py = 0   # current point, hundredths

    def _num(self, v):
        s = fmt(v)
        # A separator is needed unless the number starts with '-', or with
        # '.' after a number that already has a decimal point.
        prev = self.out[-1] if self.out else ""
        if prev and prev[-1] not in "MmLlHhVvQqZz" and s[0] != "-":
            self.out.append(" ")
        self.out.append(s)

    def cmd(self, c, *nums):
        if c != self.last or c in "mz":
            self.out.append(c)
            self.last = c
        for v in nums:
            self._num(v)

    def text(self):
        return "".join(self.out)


def q(v):
    return int(round(v * 100))


def write(subs, tol):
    w = Writer()
    for start, segs, closed in subs:
        sx, sy = q(start[0]), q(start[1])
        if w.out:
            w.cmd("m", sx - w.px, sy - w.py)
        else:
            w.cmd("M", sx, sy)
            w.last = "m"   # so the next lineto is written with its letter
        w.px, w.py = sx, sy
        if not segs:
            continue
        for kind, pts in segs:
            if kind == "L":
                prev = (w.px / 100, w.py / 100)
                arr = np.array([prev] + pts)
                if len(arr) > 2 and tol > 0:
                    arr = arr[dp(arr, tol)]
                for x, y in arr[1:]:
                    nx, ny = q(x), q(y)
                    dx, dy = nx - w.px, ny - w.py
                    if dx == 0 and dy == 0 and len(arr) > 2:
                        continue
                    if dy == 0:
                        w.cmd("h", dx)
                    elif dx == 0:
                        w.cmd("v", dy)
                    else:
                        w.cmd("l", dx, dy)
                    w.px, w.py = nx, ny
            else:
                (x1, y1), (x, y) = pts
                a, b, nx, ny = q(x1), q(y1), q(x), q(y)
                w.cmd("q", a - w.px, b - w.py, nx - w.px, ny - w.py)
                w.px, w.py = nx, ny
        if closed:
            w.cmd("z")
            w.px, w.py = sx, sy
    return w.text()


def slim(text, tol=0.01):
    stats = dict(before=0, after=0)

    def sub(m):
        d = m.group(2)
        subs = parse(d)
        nd = write(subs, tol)
        stats["before"] += len(TOK.findall(d))
        stats["after"] += len(TOK.findall(nd))
        return m.group(1) + nd + m.group(3)

    out = D_ATTR.sub(sub, text)
    # Coordinates of rects, circles and lines to 0.01 mm (transforms left exact).
    out = re.sub(r'(\s(?:x|y|x1|y1|x2|y2|cx|cy|r|width|height)="-?\d+\.\d{2})\d+"', r'\1"', out)
    out = re.sub(r">\s*\n\s*<", ">\n<", out)
    return out, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst", nargs="?")
    ap.add_argument("--tol", type=float, default=0.01, help="mm")
    a = ap.parse_args()
    text = open(a.src, encoding="utf-8").read()
    out, st = slim(text, a.tol)
    dst = a.dst or a.src
    open(dst, "w", encoding="utf-8").write(out)
    print(f"{a.src}: {len(text) / 1e6:.2f} MB -> {len(out) / 1e6:.2f} MB; path tokens {st['before']:,} -> {st['after']:,}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
