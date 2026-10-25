"""Illustrator package for a sheet: a lighter whole sheet plus one file per layer.

Illustrator struggles with three things these sheets have a lot of:

  - SVG blur filters, which it keeps as live effects and re-renders on every
    redraw. Each blurred group becomes a 300 dpi image of itself (the blur is
    already soft, so nothing is lost at print size), in the same layer, under
    the same clip and opacity.
  - Compound paths with tens of thousands of subpaths (the stipple dots) or
    anchors (long streamlines): every selection or move recomputes the whole
    compound. They are split into paths of at most --chunk subpaths and
    --max-anchors anchors, grouped together in the same layer.
  - Everything at once: parts/ holds each layer as its own SVG on the same
    artboard, so a layer can be opened, edited and pasted back in place (or
    the sheet assembled from only the layers being worked on).

    python -m tools.ai_pack outputs/sheets/04b_riverbank_section.svg [--out outputs/illustrator]
"""

import argparse
import base64
import copy
import os
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from tools.slim_svg import parse, write

SVG_NS = "http://www.w3.org/2000/svg"
INK_NS = "http://www.inkscape.org/namespaces/inkscape"
XLINK_NS = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG_NS)
ET.register_namespace("inkscape", INK_NS)
ET.register_namespace("xlink", XLINK_NS)
T = lambda tag: f"{{{SVG_NS}}}{tag}"
LABEL = f"{{{INK_NS}}}label"
GROUPMODE = f"{{{INK_NS}}}groupmode"

RENDER_JS = r"""
const { chromium } = require('/opt/node22/lib/node_modules/playwright');
(async () => {
  const [svg, png, w, h] = process.argv.slice(2);
  const b = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium-1194/chrome-linux/chrome' });
  const p = await b.newPage({ viewport: { width: +w, height: +h } });
  await p.goto('file://' + svg);
  await p.screenshot({ path: png, omitBackground: true, clip: { x: 0, y: 0, width: +w, height: +h } });
  await b.close();
})();
"""


def bbox(el):
    """Bounding box (mm) of an element's drawn geometry, ignoring stroke."""
    xs, ys = [], []
    for e in el.iter():
        if e.get("transform"):
            raise ValueError("transform inside a blurred group")
        if e.tag == T("path"):
            for start, segs, _ in parse(e.get("d", "")):
                xs.append(start[0]); ys.append(start[1])
                for _, pts in segs:
                    for x, y in pts:
                        xs.append(x); ys.append(y)
        elif e.tag == T("circle"):
            cx, cy, r = (float(e.get(k, 0)) for k in ("cx", "cy", "r"))
            xs += [cx - r, cx + r]; ys += [cy - r, cy + r]
        elif e.tag == T("line"):
            xs += [float(e.get("x1")), float(e.get("x2"))]; ys += [float(e.get("y1")), float(e.get("y2"))]
        elif e.tag == T("rect"):
            x, y, w, h = (float(e.get(k, 0)) for k in ("x", "y", "width", "height"))
            xs += [x, x + w]; ys += [y, y + h]
    return min(xs), min(ys), max(xs), max(ys)


def rasterize(root, el, dpi, tmp, w_mm, h_mm):
    """A PNG <image> standing in for blurred element el."""
    std = 0.0
    fid = re.search(r"#([^)]+)", el.get("filter")).group(1)
    for f in root.iter(T("filter")):
        if f.get("id") == fid:
            std = max(float(g.get("stdDeviation", 0)) for g in f.iter(T("feGaussianBlur")))
    x0, y0, x1, y1 = bbox(el)
    pad = 3.5 * std + 1.0                  # blur tail + widest stroke
    x0, y0 = max(0.0, x0 - pad), max(0.0, y0 - pad)
    x1, y1 = min(w_mm, x1 + pad), min(h_mm, y1 + pad)
    # Snap to whole pixels so the image sits exactly where the vector was.
    # Resolution follows the blur: at least 3 px per standard deviation, up to dpi.
    dpi = min(dpi, max(150.0, 3 * 25.4 / std)) if std else dpi
    px = dpi / 25.4
    wpx, hpx = int(np.ceil((x1 - x0) * px)), int(np.ceil((y1 - y0) * px))
    x1, y1 = x0 + wpx / px, y0 + hpx / px

    doc = ET.Element(T("svg"), {"width": f"{wpx}px", "height": f"{hpx}px",
                                "viewBox": f"{x0:.4f} {y0:.4f} {x1 - x0:.4f} {y1 - y0:.4f}"})
    defs = root.find(T("defs"))
    if defs is not None:
        doc.append(defs)
    # Filter only: clip and opacity go back on a wrapper around the image,
    # where SVG applies them anyway (after the filter).
    bare = copy.copy(el)
    bare.attrib = {k: v for k, v in el.attrib.items() if k not in ("opacity", "clip-path")}
    doc.append(bare)
    src = os.path.join(tmp, "blur.svg")
    png = os.path.join(tmp, "blur.png")
    ET.ElementTree(doc).write(src, encoding="utf-8", xml_declaration=False)
    js = os.path.join(tmp, "render.js")
    with open(js, "w") as f:
        f.write(RENDER_JS)
    subprocess.run(["node", js, src, png, str(wpx), str(hpx)], check=True)
    if defs is not None:
        doc.remove(defs)
    doc.remove(bare)
    shrink_png(png)
    with open(png, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    img = ET.Element(T("image"), {"x": f"{x0:.3f}", "y": f"{y0:.3f}", "width": f"{x1 - x0:.3f}",
                                  "height": f"{y1 - y0:.3f}", "preserveAspectRatio": "none",
                                  f"{{{XLINK_NS}}}href": "data:image/png;base64," + data})
    return img, (wpx, hpx)


def shrink_png(png):
    """One-ink images (the usual case) saved as grey + alpha, losslessly."""
    from PIL import Image
    im = np.asarray(Image.open(png).convert("RGBA"))
    ink = im[..., :3][im[..., 3] > 0]
    if len(ink) and np.ptp(ink, axis=0).max() <= 2 and np.ptp(ink[:, :1]) <= 2 and (np.ptp(ink, axis=1) <= 2).all():
        g = np.full(im.shape[:2], int(np.median(ink[:, 0])), np.uint8)
        Image.fromarray(np.dstack([g, im[..., 3]]), "LA").save(png, optimize=True)
    else:
        Image.fromarray(im, "RGBA").save(png, optimize=True)


def chunk_paths(root, n, max_anchors):
    """Split paths with more than n subpaths or max_anchors anchors into
    several paths. Opacity and clip move to a group around the pieces, so
    overlaps composite exactly as they did in the single path."""
    split = 0
    parents = {c: p for p in root.iter() for c in p}
    for el in list(root.iter(T("path"))):
        parent = parents.get(el)
        d = el.get("d", "")
        if parent is None or parent.tag == T("clipPath") or len(d) < 4 * max_anchors:
            continue
        subs = parse(d)
        sizes = [1 + sum(len(pts) for _, pts in segs) for _, segs, _ in subs]
        groups, cur, cur_a = [], [], 0
        for sub, a in zip(subs, sizes):
            if cur and (len(cur) >= n or cur_a + a > max_anchors):
                groups.append(cur)
                cur, cur_a = [], 0
            cur.append(sub)
            cur_a += a
        groups.append(cur)
        if len(groups) < 2:
            continue
        i = list(parent).index(el)
        parent.remove(el)
        keep = {k: v for k, v in el.attrib.items() if k in ("opacity", "clip-path")}
        wrap = ET.Element(T("g"), keep)
        wrap.text = "\n"
        wrap.tail = "\n"
        for g in groups:
            e = ET.SubElement(wrap, T("path"), {k: v for k, v in el.attrib.items() if k not in keep and k != "d"})
            e.set("d", write(g, 0))
            e.tail = "\n"
        parent.insert(i, wrap)
        split += 1
    return split


def layers(root):
    return [g for g in root if g.tag == T("g") and g.get(GROUPMODE) == "layer"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="+")
    ap.add_argument("--out", default="outputs/illustrator")
    ap.add_argument("--dpi", type=float, default=300)
    ap.add_argument("--chunk", type=int, default=2000, help="most subpaths per path")
    ap.add_argument("--max-anchors", type=int, default=6000, help="most anchors per path")
    a = ap.parse_args()
    for src in a.src:
        name = os.path.splitext(os.path.basename(src))[0]
        out = os.path.join(a.out, name)
        os.makedirs(os.path.join(out, "parts"), exist_ok=True)
        tree = ET.parse(src)
        root = tree.getroot()
        w_mm, h_mm = (float(v) for v in root.get("viewBox").split()[2:])
        parents = {c: p for p in root.iter() for c in p}
        blurred = [e for e in root.iter() if e.get("filter")]
        with tempfile.TemporaryDirectory() as tmp:
            for el in blurred:
                parent = parents[el]
                i = list(parent).index(el)
                parent.remove(el)
                attrs = {k: v for k, v in el.attrib.items() if k not in ("filter",)}
                img, size = rasterize(root, el, a.dpi, tmp, w_mm, h_mm)
                # Opacity and clip stay on a wrapper, as on the original group.
                wrap = ET.Element(T("g"), {k: v for k, v in attrs.items() if k in ("opacity", "clip-path")})
                wrap.append(img)
                wrap.tail = "\n"
                parent.insert(i, wrap)
                print(f"  {name}: blurred group -> {size[0]}x{size[1]} px image", file=sys.stderr)
        # Filters no longer referenced.
        defs = root.find(T("defs"))
        if defs is not None:
            for f in list(defs.findall(T("filter"))):
                defs.remove(f)
        n_split = chunk_paths(root, a.chunk, a.max_anchors)
        whole = os.path.join(out, f"{name}_illustrator.svg")
        tree.write(whole, encoding="utf-8", xml_declaration=False)

        for k, g in enumerate(layers(root)):
            label = g.get(LABEL, g.get("id"))
            if label == "PAPER":
                continue
            doc = ET.Element(root.tag, dict(root.attrib))
            if defs is not None:
                used = set(re.findall(r"url\(#([^)]+)\)", ET.tostring(g, encoding="unicode")))
                d2 = ET.SubElement(doc, T("defs"))
                for c in defs:
                    if c.get("id") in used:
                        d2.append(c)
            doc.append(g)
            words = " ".join(p for p in label.split("·") if "mm" not in p)
            fname = f"{k:02d}_" + re.sub(r"[^A-Za-z0-9]+", "_", words).strip("_").lower()[:60] + ".svg"
            ET.ElementTree(doc).write(os.path.join(out, "parts", fname), encoding="utf-8", xml_declaration=False)
        print(f"{name}: {os.path.getsize(src) / 1e6:.2f} MB -> {os.path.getsize(whole) / 1e6:.2f} MB, "
              f"{len(blurred)} blurs as images, {n_split} paths split; parts in {out}/parts", file=sys.stderr)


if __name__ == "__main__":
    main()
