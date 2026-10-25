"""Illustrator package for a sheet: a lighter whole sheet plus one file per layer.

Illustrator struggles with three things these sheets have a lot of:

  - SVG blur filters, which it keeps as live effects and re-renders on every
    redraw. Each blurred group becomes a 600 dpi image (300 dpi at least for the largest) of itself (the blur is
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


def rasterize(root, el, dpi, tmp, w_mm, h_mm, max_px=60e6):
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
    # Full resolution unless the image would be very large; never under 300 dpi.
    dpi = max(300.0, min(dpi, 25.4 * np.sqrt(max_px / ((x1 - x0) * (y1 - y0)))))
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


def render_layers(root, group, tmp, box, px_per_mm, opaque):
    """Render layer elements `group` over box (x0, y0, x1, y1 mm). Returns
    the PNG path and its pixel size."""
    x0, y0, x1, y1 = box
    wpx, hpx = int(np.ceil((x1 - x0) * px_per_mm)), int(np.ceil((y1 - y0) * px_per_mm))
    doc = ET.Element(T("svg"), {"width": f"{wpx}px", "height": f"{hpx}px",
                                "viewBox": f"{x0:.4f} {y0:.4f} {wpx / px_per_mm:.4f} {hpx / px_per_mm:.4f}"})
    defs = root.find(T("defs"))
    if defs is not None:
        doc.append(copy.deepcopy(defs))
    if opaque:
        ET.SubElement(doc, T("rect"), {"x": str(x0 - 1), "y": str(y0 - 1), "width": str(x1 - x0 + 2),
                                       "height": str(y1 - y0 + 2), "fill": "#ffffff"})
    for g in group:
        doc.append(copy.deepcopy(g))
    src, png, js = (os.path.join(tmp, n) for n in ("flat.svg", "flat.png", "render.js"))
    ET.ElementTree(doc).write(src, encoding="utf-8", xml_declaration=False)
    with open(js, "w") as f:
        f.write(RENDER_JS)
    subprocess.run(["node", js, src, png, str(wpx), str(hpx)], check=True)
    return png, (wpx, hpx)


def flatten(root, prefix, dpi, tmp, w_mm, h_mm):
    """Replace every layer whose label starts with prefix by one layer holding
    a single opaque image of them, rendered exactly as the browser and PDF
    draw them (blurs included). Returns the new layer's label, or None."""
    from PIL import Image
    group = [g for g in layers(root) if g.get(LABEL, "").startswith(prefix)]
    if not group:
        return None
    # Where they actually show (clips applied): a quick pass at 4 px/mm.
    png, _ = render_layers(root, group, tmp, (0, 0, w_mm, h_mm), 4, False)
    alpha = np.asarray(Image.open(png).convert("RGBA"))[..., 3]
    ys, xs = np.nonzero(alpha)
    box = (max(0.0, xs.min() / 4 - 1), max(0.0, ys.min() / 4 - 1),
           min(w_mm, (xs.max() + 1) / 4 + 1), min(h_mm, (ys.max() + 1) / 4 + 1))
    px = dpi / 25.4
    png, size = render_layers(root, group, tmp, box, px, True)
    im = np.asarray(Image.open(png).convert("RGB"))
    if np.ptp(im.astype(int), axis=2).max() <= 2:
        Image.fromarray(im[..., 0], "L").save(png, optimize=True)
    else:
        Image.fromarray(im, "RGB").save(png, optimize=True)
    with open(png, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    label = f"{prefix} · IMAGE, {dpi:.0f} DPI (ALL ITS LAYERS)"
    lid = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")
    layer = ET.Element(T("g"), {"id": lid, GROUPMODE: "layer", LABEL: label})
    layer.text = "\n"
    layer.tail = "\n"
    ET.SubElement(layer, T("image"), {
        "x": f"{box[0]:.3f}", "y": f"{box[1]:.3f}", "width": f"{size[0] / px:.3f}", "height": f"{size[1] / px:.3f}",
        "preserveAspectRatio": "none", f"{{{XLINK_NS}}}href": "data:image/png;base64," + data}).tail = "\n"
    i = list(root).index(group[0])
    for g in group:
        root.remove(g)
    root.insert(i, layer)
    return label, size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="+")
    ap.add_argument("--out", default="outputs/illustrator")
    ap.add_argument("--dpi", type=float, default=600)
    ap.add_argument("--chunk", type=int, default=2000, help="most subpaths per path")
    ap.add_argument("--max-anchors", type=int, default=6000, help="most anchors per path")
    ap.add_argument("--flatten", nargs="*", default=["WIND PLAN", "WIND ROSE"],
                    help="layer-name prefixes drawn as one opaque image each (vector copies go to parts/vector)")
    a = ap.parse_args()
    for src in a.src:
        name = os.path.splitext(os.path.basename(src))[0]
        out = os.path.join(a.out, name)
        os.makedirs(os.path.join(out, "parts"), exist_ok=True)
        tree = ET.parse(src)
        root = tree.getroot()
        w_mm, h_mm = (float(v) for v in root.get("viewBox").split()[2:])
        vec = os.path.join(out, "parts", "vector")
        with tempfile.TemporaryDirectory() as tmp:
            for prefix in a.flatten:
                originals = [copy.deepcopy(g) for g in layers(root) if g.get(LABEL, "").startswith(prefix)]
                done = flatten(root, prefix, a.dpi, tmp, w_mm, h_mm)
                if not done:
                    continue
                os.makedirs(vec, exist_ok=True)
                doc = ET.Element(root.tag, dict(root.attrib))
                if root.find(T("defs")) is not None:
                    doc.append(copy.deepcopy(root.find(T("defs"))))
                doc.extend(originals)
                fn = re.sub(r"[^A-Za-z0-9]+", "_", prefix).strip("_").lower() + "_vector.svg"
                ET.ElementTree(doc).write(os.path.join(vec, fn), encoding="utf-8", xml_declaration=False)
                print(f"  {name}: {prefix} -> {done[1][0]}x{done[1][1]} px image", file=sys.stderr)
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
