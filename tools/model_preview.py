"""Quick oblique preview (PNG) of a site-model OBJ: card layers, base board, wood and roads.
Heights are exaggerated (--ve) so the layers read; the model itself is true scale.

    python -m tools.model_preview outputs/site_models/A/lockport_1to500.obj out.png --title "..."
"""

import argparse

import numpy as np
from PIL import Image, ImageDraw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("obj")
    ap.add_argument("png")
    ap.add_argument("--ve", type=float, default=4.0)
    ap.add_argument("--zoom", type=float, default=2.0)
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    V, objs, cur = [], {}, None
    for line in open(a.obj):
        if line.startswith("v "):
            V.append(tuple(map(float, line.split()[1:])))
        elif line.startswith("o "):
            cur = line.split()[1]
            objs[cur] = []
        elif line[:2] in ("f ", "l "):
            objs[cur].append((line[0], [int(t) for t in line.split()[1:]]))
    V = np.array(V)
    k = a.zoom
    X = (V[:, 0] - V[:, 1] * 0.35) * k
    Y = (-V[:, 1] * 0.55 - V[:, 2] * a.ve - V[:, 0] * 0.12) * k
    ox, oy = -X.min() + 40, -Y.min() + 60
    im = Image.new("RGB", (int(X.max() + ox + 40), int(Y.max() + oy + 40)), (250, 250, 248))
    d = ImageDraw.Draw(im)

    def poly(ids, fill):
        d.polygon([(X[i - 1] + ox, Y[i - 1] + oy) for i in ids], fill=fill)

    order = ["base_board"] + sorted(n for n in objs if n.startswith("card_layer")) + \
            [n for n in objs if n.startswith("wood")] + ["roads"]
    for n in order:
        if n not in objs:
            continue
        if n == "roads":
            for _, ids in objs[n]:
                d.line([(X[i - 1] + ox, Y[i - 1] + oy) for i in ids], fill=(70, 70, 70), width=1)
            continue
        if n.startswith("card_layer"):
            kk = int(n[-2:])
            g = int(200 + kk * 2.8)
            top, side = (g, g - 8, g - 25), (g - 60, g - 70, g - 85)
        elif n == "base_board":
            top, side = (110, 130, 150), (80, 80, 75)
        else:
            top, side = (205, 160, 105), (150, 110, 70)
        for _, ids in objs[n]:
            if len(ids) == 4:
                poly(ids, side)
        tris = [ids for _, ids in objs[n] if len(ids) == 3]
        if tris:
            zmax = max(V[i - 1][2] for ids in tris for i in ids)
            for ids in tris:
                if abs(V[ids[0] - 1][2] - zmax) < 1e-6 or n.startswith("wood"):
                    poly(ids, top)
    d.text((40, 20), f"{a.title}  -  preview, heights x{a.ve:g} so the layers read (the model is true scale)",
           fill=(30, 30, 30))
    im.save(a.png)


if __name__ == "__main__":
    main()
