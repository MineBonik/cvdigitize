"""Human-in-the-loop guided digitization (standalone; not wired into the CLI).

Two steps around the paint-like tracer in ``tools/trace_assist.html``:

  1. export a figure panel to a PNG you can load into the tracer:

         python scripts/trace_guided.py panel paper.pdf 4 -o panel.png

  2. after scribbling a rough guide along each curve and exporting guides.json,
     turn those rough guides into pixel-accurate curves:

         python scripts/trace_guided.py run panel.png guides.json -o out/

     writes one CSV per curve (pixel coords) plus overlay.png for a visual check.

The rough guide only resolves *which* ink is which curve and the sweep order;
the trace snaps to the real ink (see ``cvdigitize.guided.extract_near_guide``).
This is deliberately separate from ``python -m cvdigitize`` — merge later if
wanted.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _cmd_panel(args):
    import cv2
    from cvdigitize.raster_extract import find_image_regions, render_region
    from cvdigitize.ingest import render_page
    regions = find_image_regions(args.pdf, args.page)
    if regions and args.region is not None:
        img = render_region(args.pdf, args.page, regions[args.region].bbox, zoom=args.zoom)
    elif regions:
        # largest image region on the page by area
        r = max(regions, key=lambda r: (r.bbox[2]-r.bbox[0])*(r.bbox[3]-r.bbox[1]))
        img = render_region(args.pdf, args.page, r.bbox, zoom=args.zoom)
    else:
        img = render_page(args.pdf, args.page, zoom=args.zoom)
    out = args.out or f"panel_p{args.page}.png"
    cv2.imwrite(out, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"wrote {out}  ({img.shape[1]}x{img.shape[0]} px) — load it in tools/trace_assist.html")


def _cmd_run(args):
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cvdigitize.guided import extract_guides

    bgr = cv2.imread(args.panel)
    if bgr is None:
        sys.exit(f"could not read image: {args.panel}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    with open(args.guides, encoding="utf-8") as f:
        guides = json.load(f).get("guides", [])
    if not guides:
        sys.exit("no guides in the JSON")

    results = extract_guides(rgb, guides, radius=args.radius)
    os.makedirs(args.out, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(rgb)
    for res in results:
        poly = res["polyline_px"]
        name = res["name"] or "curve"
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name) or "curve"
        with open(os.path.join(args.out, f"{safe}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["x_px", "y_px"])
            w.writerows(np.round(poly, 2))
        ax.plot(poly[:, 0], poly[:, 1], lw=1.2, label=name)
        print(f"  {name}: {len(poly)} points -> {safe}.csv")
    ax.legend(fontsize=8)
    ax.set_title("guided extraction (traced on your rough guides)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "overlay.png"), dpi=110)
    print(f"wrote {len(results)} curve(s) + overlay.png to {args.out}/")
    if not results:
        print("  (no ink found in any corridor — widen with --radius or redraw guides)")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Guided (human-in-the-loop) digitization.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("panel", help="export a figure panel PNG for the tracer")
    p.add_argument("pdf")
    p.add_argument("page", type=int)
    p.add_argument("--region", type=int, default=None, help="image-region index on the page")
    p.add_argument("--zoom", type=float, default=4.0)
    p.add_argument("-o", "--out")
    p.set_defaults(func=_cmd_panel)

    r = sub.add_parser("run", help="guided extraction from panel.png + guides.json")
    r.add_argument("panel")
    r.add_argument("guides")
    r.add_argument("--radius", type=int, default=None, help="corridor half-width in px")
    r.add_argument("-o", "--out", default="guided_out")
    r.set_defaults(func=_cmd_run)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
