"""Experimental ML tracer: a learned instance-embedding net for disentangling
overlapping/dashed curve bundles — the one failure mode the classical pipeline
cannot resolve by construction.

**Why this, not LineFormer.** The corpus's real hard cases (chen_2024 panel b:
7 same-hue curves; hoshi_2013 panel b: 8 curves + an in-plot legend) are not
about finding ink — the classical mask/skeleton/strand pipeline finds every
pixel correctly. The problem is *which strand belongs to which physical curve*
when several curves interleave (a dashed curve's dashes alternate between the
anodic and cathodic branches of ITS OWN loop, so no single traversal order is
right) or overlap almost exactly (near-identical Pt(hkl) loops in one panel).
LineFormer (arxiv 2305.01837) solves exactly this via instance segmentation,
but needs its own trained weights + mmdetection — not installable/verifiable
here (no GPU, fragile Windows build, no network access to fetch the checkpoint
under test conditions). So: same idea (learn what makes two ink pixels/strands
"the same curve"), minimal dependency (plain PyTorch, CPU), trained from
scratch on procedurally-generated synthetic bundles (free, unlimited, exactly
matches the target failure mode) rather than a borrowed general-purpose model.

**Approach: discriminative pixel embedding** (De Brabandere et al. 2017,
"Semantic Instance Segmentation with a Discriminative Loss Function"). A small
FCN maps each ink pixel to a D-dim embedding; a push-pull loss trains same-
instance pixels to cluster tightly and different-instance clusters to repel.
At inference, the classical strand decomposition (cvdigitize.strands) still
finds the smooth pieces — the net only re-answers "which pieces belong
together", by clustering each strand's mean embedding. This keeps the ML model
narrow (a clustering signal, not a full tracer) and lets it be evaluated in
isolation against the classical nearest-endpoint stitch on the exact curves
that stitch broke.

Usage:
    .venv\\Scripts\\python.exe scripts\\experimental\\ml_tracer.py train -o model.pt
    .venv\\Scripts\\python.exe scripts\\experimental\\ml_tracer.py eval-synthetic model.pt
    .venv\\Scripts\\python.exe scripts\\experimental\\ml_tracer.py eval-real model.pt <panel.png>
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    sys.exit("scripts/experimental/ml_tracer.py needs torch: "
             "pip install -r requirements-ml.txt")


# ---------------------------------------------------------------------------
# 1. Synthetic training data: procedural overlapping/dashed curve bundles
# ---------------------------------------------------------------------------
def _make_loop(rng, cx, cy, rx, ry, n=220, phase=0.0, wobble=0.15):
    """A closed CV-ish loop: an ellipse perturbed by a couple of harmonics, so
    it has the asymmetric, peak-y silhouette of a real voltammogram, not a
    perfect circle (which would make instance separation trivially easy)."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False) + phase
    r = 1.0 + wobble * np.sin(2 * t + rng.uniform(0, 6.28)) \
            + 0.5 * wobble * np.sin(5 * t + rng.uniform(0, 6.28))
    x = cx + rx * r * np.cos(t)
    y = cy + ry * r * np.sin(t)
    return np.column_stack([x, y])


def _rasterize_instances(canvas_size, curves, dashed_flags, *, thickness=2):
    """Render each curve polyline into its own instance-id layer (dashes as
    real gaps in the stroke), return (ink_mask, instance_map) both HxW."""
    import cv2
    h, w = canvas_size
    inst = np.zeros((h, w), np.int32)
    ink = np.zeros((h, w), np.uint8)
    for k, (poly, dashed) in enumerate(zip(curves, dashed_flags), start=1):
        pts = poly.astype(np.int32)
        segs = list(zip(pts[:-1], pts[1:])) + [(pts[-1], pts[0])]
        for i, (p0, p1) in enumerate(segs):
            if dashed and (i // 4) % 2 == 1:      # ~4-step dash / 4-step gap
                continue
            layer = np.zeros((h, w), np.uint8)
            cv2.line(layer, tuple(p0), tuple(p1), 1, thickness)
            inst[layer > 0] = k
            ink[layer > 0] = 1
    return ink, inst


def synthetic_batch(rng, *, n_images=8, size=96, n_curves_range=(2, 5)):
    """One training batch: (ink [B,1,H,W] float, inst [B,H,W] int64)."""
    h = w = size
    inks, insts = [], []
    for _ in range(n_images):
        n_curves = rng.integers(*n_curves_range, endpoint=True)
        margin = size * 0.18
        cx, cy = size / 2, size / 2
        curves, dashed = [], []
        for i in range(n_curves):
            rx = rng.uniform(size * 0.25, size * 0.42)
            ry = rng.uniform(size * 0.20, size * 0.38)
            jitter = rng.uniform(-margin * 0.3, margin * 0.3, 2)
            poly = _make_loop(rng, cx + jitter[0], cy + jitter[1], rx, ry,
                              phase=rng.uniform(0, 6.28))
            curves.append(poly)
            dashed.append(bool(i % 2) and rng.random() < 0.6)
        ink, inst = _rasterize_instances((h, w), curves, dashed)
        inks.append(ink); insts.append(inst)
    ink_t = torch.from_numpy(np.stack(inks)).float().unsqueeze(1)
    inst_t = torch.from_numpy(np.stack(insts)).long()
    return ink_t, inst_t


# ---------------------------------------------------------------------------
# 2. Tiny embedding FCN (CoordConv input helps it use position/shape context)
# ---------------------------------------------------------------------------
class EmbedNet(nn.Module):
    """Dilated-conv trunk with a deliberately LARGE receptive field.

    First attempt (dilations 1,2,4,8 -> receptive radius ~16px on a 96px
    canvas) plateaued at ARI ~0.3: disambiguating which of two overlapping
    loops a pixel near a crossing belongs to needs enough context to follow
    each strand's curvature well past the crossing, not just a small
    neighbourhood (a single global-average-pooled vector was tried and
    discarded first — it broadcasts the SAME context to every pixel, so it
    cannot help tell two instances in the same image apart; the fix has to be
    a genuinely larger per-pixel receptive field). This version pushes
    dilation to 16 (radius ~31px) on a smaller 64px canvas, so the field of
    view covers roughly half the loop's own curvature."""
    def __init__(self, emb_dim: int = 8):
        super().__init__()
        c = 24
        self.net = nn.Sequential(
            nn.Conv2d(3, c, 5, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=4, dilation=4), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=8, dilation=8), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=16, dilation=16), nn.ReLU(inplace=True),
            nn.Conv2d(c, emb_dim, 1),
        )

    def forward(self, ink: torch.Tensor) -> torch.Tensor:
        b, _, h, w = ink.shape
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w),
                                indexing="ij")
        coord = torch.stack([yy, xx]).unsqueeze(0).expand(b, -1, -1, -1).to(ink.device)
        x = torch.cat([ink, coord], dim=1)
        return self.net(x)


# ---------------------------------------------------------------------------
# 3. Discriminative push-pull loss (De Brabandere et al. 2017)
# ---------------------------------------------------------------------------
def discriminative_loss(emb: torch.Tensor, inst: torch.Tensor, *, delta_v=0.5,
                        delta_d=2.0, reg_w=0.001) -> torch.Tensor:
    """emb: [B,D,H,W] embeddings; inst: [B,H,W] instance ids (0 = background)."""
    b = emb.shape[0]
    total = emb.new_zeros(())
    n_valid = 0
    for i in range(b):
        e = emb[i].permute(1, 2, 0)                  # H,W,D
        lab = inst[i]
        ids = [int(v) for v in torch.unique(lab) if v != 0]
        if len(ids) < 1:
            continue
        means = []
        var_loss = e.new_zeros(())
        for cid in ids:
            m = lab == cid
            pts = e[m]                                 # N,D
            mean = pts.mean(0)
            means.append(mean)
            d = torch.clamp(torch.norm(pts - mean, dim=1) - delta_v, min=0.0)
            var_loss = var_loss + (d ** 2).mean()
        var_loss = var_loss / len(ids)
        dist_loss = e.new_zeros(())
        if len(means) > 1:
            M = torch.stack(means)                     # K,D
            diff = M.unsqueeze(0) - M.unsqueeze(1)      # K,K,D
            dist = torch.norm(diff, dim=2)
            k = len(means)
            mask = ~torch.eye(k, dtype=torch.bool, device=e.device)
            d = torch.clamp(2 * delta_d - dist[mask], min=0.0)
            dist_loss = (d ** 2).mean()
        reg_loss = torch.stack([torch.norm(m) for m in means]).mean()
        total = total + var_loss + dist_loss + reg_w * reg_loss
        n_valid += 1
    return total / max(n_valid, 1)


# ---------------------------------------------------------------------------
# 4. Train / evaluate
# ---------------------------------------------------------------------------
def _cmd_train(args):
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    net = EmbedNet(emb_dim=args.emb_dim)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for step in range(args.steps):
        ink, inst = synthetic_batch(rng, n_images=args.batch, size=args.size)
        emb = net(ink)
        loss = discriminative_loss(emb, inst)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % max(1, args.steps // 10) == 0 or step == args.steps - 1:
            print(f"  step {step:4d}/{args.steps}  loss {loss.item():.4f}")
    torch.save({"state_dict": net.state_dict(), "emb_dim": args.emb_dim}, args.out)
    print(f"wrote {args.out}")


def load_net(path: str) -> EmbedNet:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    net = EmbedNet(emb_dim=ckpt["emb_dim"])
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net


def cluster_strands(net: EmbedNet, ink_mask: np.ndarray, strands: list[np.ndarray],
                    *, dist_thresh: float = 2.0) -> list[int]:
    """Assign each strand to a cluster id via the net's embedding + simple
    agglomerative merge (distance < dist_thresh -> same curve). Returns a
    cluster-id list parallel to ``strands``."""
    h, w = ink_mask.shape
    with torch.no_grad():
        ink_t = torch.from_numpy(ink_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        emb = net(ink_t)[0].permute(1, 2, 0).numpy()   # H,W,D

    means = []
    for s in strands:
        xi = np.clip(np.round(s[:, 0]).astype(int), 0, w - 1)
        yi = np.clip(np.round(s[:, 1]).astype(int), 0, h - 1)
        means.append(emb[yi, xi].mean(axis=0))
    means = np.asarray(means)
    if len(means) <= 1:
        return [0] * len(means)

    from scipy.cluster.hierarchy import linkage, fcluster
    Z = linkage(means, method="average", metric="euclidean")
    labels = fcluster(Z, t=dist_thresh, criterion="distance")
    return [int(l) - 1 for l in labels]


def _adjusted_rand_index(true_labels, pred_labels) -> float:
    """Self-contained ARI (Hubert & Arabie 1985) — avoids adding scikit-learn as
    a dependency just for one metric. Standard contingency-table formula."""
    from itertools import product
    from math import comb
    true_labels = list(true_labels); pred_labels = list(pred_labels)
    tset = sorted(set(true_labels)); pset = sorted(set(pred_labels))
    cont = {(t, p): 0 for t, p in product(tset, pset)}
    for t, p in zip(true_labels, pred_labels):
        cont[(t, p)] += 1
    a = {t: sum(cont[(t, p)] for p in pset) for t in tset}
    b = {p: sum(cont[(t, p)] for t in tset) for p in pset}
    n = len(true_labels)
    sum_comb_c = sum(comb(v, 2) for v in cont.values() if v >= 2)
    sum_comb_a = sum(comb(v, 2) for v in a.values() if v >= 2)
    sum_comb_b = sum(comb(v, 2) for v in b.values() if v >= 2)
    total_comb = comb(n, 2) if n >= 2 else 1
    expected = sum_comb_a * sum_comb_b / total_comb
    max_index = 0.5 * (sum_comb_a + sum_comb_b)
    denom = max_index - expected
    if denom == 0:
        return 1.0 if sum_comb_c == expected else 0.0
    return (sum_comb_c - expected) / denom


def _cmd_eval_synthetic(args):
    """Quantitative check: adjusted-rand-index between predicted strand
    clusters and ground truth, on held-out synthetic bundles."""
    net = load_net(args.model)
    rng = np.random.default_rng(args.seed + 999)   # different seed = held-out
    scores = []
    for _ in range(args.n):
        ink_t, inst_t = synthetic_batch(rng, n_images=1, size=args.size)
        ink = ink_t[0, 0].numpy(); inst = inst_t[0].numpy()
        # "strands" here = ground-truth per-instance connected components split
        # into a few arbitrary chunks, mimicking real over-segmented strands
        strands, true_id = [], []
        for cid in np.unique(inst):
            if cid == 0:
                continue
            ys, xs = np.where(inst == cid)
            order = np.argsort(np.arctan2(ys - ys.mean(), xs - xs.mean()))
            pts = np.column_stack([xs, ys])[order]
            for chunk in np.array_split(pts, 3):
                if len(chunk) >= 3:
                    strands.append(chunk.astype(float)); true_id.append(cid)
        if len(strands) < 2:
            continue
        pred = cluster_strands(net, ink, strands, dist_thresh=args.dist_thresh)
        scores.append(_adjusted_rand_index(true_id, pred))
    print(f"synthetic held-out ARI over {len(scores)} bundles: "
          f"mean {np.mean(scores):.3f}  min {np.min(scores):.3f}  max {np.max(scores):.3f}")


def _cmd_eval_real(args):
    """Run the ML clustering on a real panel's dark-curve strands and compare
    against the classical (colour/nearest-endpoint) assembly by ink-fidelity."""
    import cv2
    from cvdigitize.guided import _ink_mask
    from cvdigitize.strands import skeleton_to_strands, strands_to_curves
    from cvdigitize.raster_extract import skeletonize_curve
    from cvdigitize.fidelity import ink_fidelity
    from cvdigitize.postprocess import order_curve

    net = load_net(args.model)
    bgr = cv2.imread(args.panel)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    ink = _ink_mask(rgb)
    skel = skeletonize_curve(ink)
    strands = skeleton_to_strands(skel)
    strands = [s for s in strands if len(s) >= 3]
    print(f"{len(strands)} strands found on {args.panel}")
    if len(strands) < 2:
        print("too few strands for a meaningful comparison"); return

    # classical: today's assembly (long/short split, nearest-endpoint stitch)
    classical_curves = strands_to_curves(strands, ink_mask=ink)
    classical_scores = [ink_fidelity(c, ink)["score"] for c in classical_curves if len(c) >= 10]

    # ML: cluster strands by learned embedding, stitch within each cluster
    labels = cluster_strands(net, ink, strands, dist_thresh=args.dist_thresh)
    clusters: dict[int, list[np.ndarray]] = {}
    for s, l in zip(strands, labels):
        clusters.setdefault(l, []).append(s)
    ml_curves = [order_curve(v, ink_mask=ink) for v in clusters.values() if len(v) >= 1]
    ml_scores = [ink_fidelity(c, ink)["score"] for c in ml_curves if len(c) >= 10]

    print(f"classical: {len(classical_curves)} curve(s), fidelity {classical_scores}")
    print(f"ML       : {len(ml_curves)} cluster(s) ({len(set(labels))} groups from "
          f"{len(strands)} strands), fidelity {ml_scores}")
    print(f"classical mean {np.mean(classical_scores) if classical_scores else float('nan'):.1f}  "
          f"ML mean {np.mean(ml_scores) if ml_scores else float('nan'):.1f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("-o", "--out", default="scripts/experimental/ml_tracer.pt")
    t.add_argument("--steps", type=int, default=400)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--size", type=int, default=96)
    t.add_argument("--emb-dim", type=int, default=8)
    t.add_argument("--seed", type=int, default=0)
    t.set_defaults(func=_cmd_train)

    e = sub.add_parser("eval-synthetic")
    e.add_argument("model")
    e.add_argument("--n", type=int, default=40)
    e.add_argument("--size", type=int, default=96)
    e.add_argument("--dist-thresh", type=float, default=2.0)
    e.add_argument("--seed", type=int, default=0)
    e.set_defaults(func=_cmd_eval_synthetic)

    r = sub.add_parser("eval-real")
    r.add_argument("model")
    r.add_argument("panel")
    r.add_argument("--dist-thresh", type=float, default=2.0)
    r.set_defaults(func=_cmd_eval_real)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
