"""Junction-aware decomposition of a curve skeleton into smooth strands.

A pixel skeleton of overlapping strokes (a solid curve crossed by its dashed
sibling, several CV curves in a bundle, a curve crossing its axes) is not a
path — it is a *graph* with many junction nodes where strokes touch. Walking it
naively dies at the first junction (it turns onto a dead-end stub and cannot
back out), which is exactly why "just trace the black line" fails on the hard
raster figures.

This module collapses the skeleton to a topological graph (junction/endpoint
nodes joined by degree-2 pixel chains = "edges"), then at every junction pairs
the incident branches by *straight-through* tangent continuity: a real crossing
has two strokes passing through, each continuing in nearly its own direction, so
the branch you arrived on is paired with the one most opposite it. Following
those pairings yields **strands** — maximal smooth paths, each belonging to one
physical stroke. A sharp CV peak lives *inside* one branch and is never a node,
so it is preserved; only genuine touch/cross points are resolved.

The output feeds curve assembly (solid vs dashed vs axis classification and
stitching) one level up.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


@dataclass
class _Edge:
    a: int                      # node id at the start
    b: int                      # node id at the end
    pts: np.ndarray             # (M, 2) pixel path (row, col), a -> b
    dir_a: np.ndarray = field(default_factory=lambda: np.zeros(2))  # unit dir leaving node a
    dir_b: np.ndarray = field(default_factory=lambda: np.zeros(2))  # unit dir leaving node b
    used: bool = False


def _pixel_set(skel: np.ndarray) -> set[tuple[int, int]]:
    ys, xs = np.where(skel)
    return set(zip(ys.tolist(), xs.tolist()))


def _neighbors(p, pixels):
    y, x = p
    return [(y + dy, x + dx) for dy, dx in _NB if (y + dy, x + dx) in pixels]


def _cluster_junctions(junctions: set) -> dict[tuple[int, int], int]:
    """8-connected components of junction pixels -> {pixel: node_id}."""
    node_of: dict[tuple[int, int], int] = {}
    nid = 0
    for p in junctions:
        if p in node_of:
            continue
        stack = [p]
        node_of[p] = nid
        while stack:
            q = stack.pop()
            for dy, dx in _NB:
                r = (q[0] + dy, q[1] + dx)
                if r in junctions and r not in node_of:
                    node_of[r] = nid
                    stack.append(r)
        nid += 1
    return node_of


def _tangent(pts: np.ndarray, at_start: bool, k: int) -> np.ndarray:
    m = min(k, len(pts) - 1)
    if m < 1:
        return np.zeros(2)
    v = (pts[m] - pts[0]) if at_start else (pts[-1] - pts[-1 - m])
    n = float(np.hypot(*v))
    return v / n if n > 0 else np.zeros(2)


def _trace_edges(pixels: set, node_of: dict, k: int) -> tuple[list[_Edge], int]:
    """Walk degree-2 chains between nodes. Endpoints (degree 1) become nodes too.

    Returns the edge list and the total node count. Each edge stores the outward
    tangent at both ends, used for junction pairing.
    """
    deg = {p: len(_neighbors(p, pixels)) for p in pixels}
    next_id = (max(node_of.values()) + 1) if node_of else 0
    # promote endpoints to their own nodes
    for p in pixels:
        if deg[p] == 1 and p not in node_of:
            node_of[p] = next_id
            next_id += 1
    node_ids = set(node_of.values())

    edges: list[_Edge] = []
    seen_starts: set = set()

    for p in list(node_of.keys()):
        for q in _neighbors(p, pixels):
            if (p, q) in seen_starts:
                continue
            if q in node_of and node_of[q] == node_of[p]:
                continue  # inside the same junction cluster
            # walk from q away from node(p) until the next node pixel
            path = [p, q]
            prev, cur = p, q
            seen_starts.add((p, q))
            while cur not in node_of:
                nxts = [r for r in _neighbors(cur, pixels) if r != prev]
                # prefer a non-node continuation; stop cleanly at a node
                node_next = [r for r in nxts if r in node_of]
                plain = [r for r in nxts if r not in node_of]
                if node_next:
                    prev, cur = cur, node_next[0]
                    path.append(cur)
                    break
                if len(plain) != 1:
                    break  # dead end or unexpected branch
                prev, cur = cur, plain[0]
                path.append(cur)
            if cur not in node_of:
                continue
            seen_starts.add((path[-1], path[-2]))  # mark the reverse walk consumed
            pts = np.array(path, dtype=float)
            e = _Edge(a=node_of[p], b=node_of[cur], pts=pts)
            # both directions point OUTWARD from their node (away along the edge),
            # so two branches "go straight through" when their dirs are opposite
            e.dir_a = _tangent(pts, at_start=True, k=k)
            e.dir_b = -_tangent(pts, at_start=False, k=k)
            edges.append(e)
    return edges, len(node_ids)


def _pair_at_node(node: int, edges: list[_Edge], idxs: list[int],
                  max_bend_deg: float) -> dict[int, int]:
    """Pair edges incident to a node by straight-through continuity.

    Each incident edge has an outward direction from the node. Two strokes pass
    straight through when their outward directions are ~opposite. Greedily match
    the most-opposite pairs; leave the rest unpaired (strand terminates)."""
    branches = []  # (edge_idx, which_end 'a'/'b', outward_dir)
    for i in idxs:
        e = edges[i]
        if e.a == node:
            branches.append((i, "a", e.dir_a))
        if e.b == node:
            branches.append((i, "b", e.dir_b))

    thresh = -np.cos(np.radians(max_bend_deg))  # accept dot <= thresh (bend<=max)
    cands = []
    for u in range(len(branches)):
        for v in range(u + 1, len(branches)):
            iu, _, du = branches[u]
            iv, _, dv = branches[v]
            if iu == iv:
                continue  # a self-loop edge can't pair with itself here
            dot = float(np.dot(du, dv))
            if dot <= thresh:
                cands.append((dot, u, v))
    cands.sort()  # most negative (straightest through) first

    pairing: dict[int, int] = {}   # branch-slot -> branch-slot
    matched = set()
    for _, u, v in cands:
        if u in matched or v in matched:
            continue
        pairing[u] = v
        pairing[v] = u
        matched.add(u)
        matched.add(v)
    # map slot pairing to a per-(edge_idx,end) continuation
    out: dict[tuple[int, str], tuple[int, str]] = {}
    for u, v in pairing.items():
        iu, eu, _ = branches[u]
        iv, ev, _ = branches[v]
        out[(iu, eu)] = (iv, ev)
    return out


def skeleton_to_strands(skel: np.ndarray, *, tangent_k: int = 8,
                        max_bend_deg: float = 55.0, min_len: int = 6
                        ) -> list[np.ndarray]:
    """Decompose a skeleton into smooth strands (each an (x, y) polyline).

    Junctions are resolved by straight-through tangent pairing, so a strand
    follows one physical stroke across crossings instead of dying at them.
    Pure loops (no junctions/endpoints) are returned whole.
    """
    pixels = _pixel_set(skel)
    if len(pixels) < min_len:
        return []
    deg = {p: len(_neighbors(p, pixels)) for p in pixels}
    junctions = {p for p, d in deg.items() if d >= 3}

    if not junctions and all(d == 2 for d in deg.values()):
        # a bare closed loop: walk it as one strand
        return _walk_pure_loop(pixels)

    node_of = _cluster_junctions(junctions)
    edges, _ = _trace_edges(pixels, node_of, tangent_k)
    if not edges:
        return _walk_pure_loop(pixels)

    # continuation map per node
    from collections import defaultdict
    node_edges: dict[int, list[int]] = defaultdict(list)
    for i, e in enumerate(edges):
        node_edges[e.a].append(i)
        if e.b != e.a:
            node_edges[e.b].append(i)
    cont: dict[tuple[int, str], tuple[int, str]] = {}
    for node, idxs in node_edges.items():
        cont.update(_pair_at_node(node, edges, idxs, max_bend_deg))

    strands = _walk_strands(edges, cont)
    out = []
    for s in strands:
        if len(s) >= min_len:
            out.append(s[:, ::-1].astype(float))  # (row,col) -> (x,y)
    return out


def _oriented(edge: _Edge, enter_end: str) -> np.ndarray:
    """Edge points ordered so traversal starts at ``enter_end`` node."""
    return edge.pts if enter_end == "a" else edge.pts[::-1]


def _walk_strands(edges: list[_Edge], cont: dict) -> list[np.ndarray]:
    """Follow through-pairings into maximal strands, consuming each edge once."""
    strands = []
    for start in range(len(edges)):
        if edges[start].used:
            continue
        # walk forward from end 'a', then from end 'b', and join
        chain_pts = [_oriented(edges[start], "a")]
        edges[start].used = True
        # extend beyond b
        _extend(edges, cont, start, "b", chain_pts, append=True)
        # extend beyond a (prepend, reversed)
        pre: list[np.ndarray] = []
        _extend(edges, cont, start, "a", pre, append=True)
        if pre:
            joined = [p[::-1] for p in reversed(pre)] + chain_pts
        else:
            joined = chain_pts
        strands.append(np.vstack(joined))
    return strands


def _extend(edges, cont, edge_idx, out_end, acc, append):
    """From ``edges[edge_idx]`` leaving via ``out_end``, follow the paired edge
    chain, appending oriented point arrays to ``acc``."""
    cur_idx, cur_end = edge_idx, out_end
    while True:
        nxt = cont.get((cur_idx, cur_end))
        if nxt is None:
            return
        n_idx, n_enter = nxt
        if edges[n_idx].used:
            return
        edges[n_idx].used = True
        acc.append(_oriented(edges[n_idx], n_enter))
        cur_idx = n_idx
        cur_end = "b" if n_enter == "a" else "a"  # leave via the far end


def _arclen(xy: np.ndarray) -> float:
    return float(np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1])).sum()) if len(xy) > 1 else 0.0


def _is_axis_strand(xy: np.ndarray, *, straight_tol: float = 0.06,
                    align_tol: float = 0.18, min_len: float = 40.0) -> bool:
    """A long, dead-straight, horizontal-or-vertical strand = a plot axis."""
    if len(xy) < 5:
        return False
    span = xy[-1] - xy[0]
    length = float(np.hypot(*span))
    if length < min_len:
        return False
    # straightness: max perpendicular distance to the endpoint chord / length
    d = span / (length or 1.0)
    rel = xy - xy[0]
    perp = np.abs(rel[:, 0] * d[1] - rel[:, 1] * d[0])
    if perp.max() / length > straight_tol:
        return False
    # axis-aligned: nearly horizontal or nearly vertical
    return abs(d[0]) < align_tol or abs(d[1]) < align_tol


def strands_to_curves(strands: list[np.ndarray], *, drop_axes: bool = False,
                      long_frac: float = 0.30, min_dashes: int = 4,
                      ink_mask=None) -> list[np.ndarray]:
    """Assemble strands into curves: a solid curve (long strands stitched) and,
    if a regular set of short strands is present, a separate dashed curve.

    Returns ordered (x, y) polylines, solid first. ``drop_axes`` removes
    straight axis-aligned strands (use on the dark/scan mask, not colour masks
    where the axes are a different colour). ``ink_mask`` (the ink these strands
    were traced from) makes the stitch ink-aware — decisive for the dashed
    sibling, whose short strands are otherwise joined by pure nearest-endpoint
    and can chord across the plot; with the mask, the stitch pays for crossing
    empty space, so dashes connect along the real curve instead."""
    from .postprocess import order_curve

    strands = [s for s in strands if len(s) >= 3]
    if drop_axes:
        strands = [s for s in strands if not _is_axis_strand(s)]
    if not strands:
        return []
    lens = [_arclen(s) for s in strands]
    mx = max(lens) or 1.0
    long = [s for s, l in zip(strands, lens) if l >= long_frac * mx]
    short = [s for s, l in zip(strands, lens) if l < long_frac * mx]

    curves: list[np.ndarray] = []
    if long:
        curves.append(order_curve(long, ink_mask=ink_mask))
    # A dashed sibling only when there are several short strands that, together,
    # sweep a real fraction of the width (a dashed CV spans the plot). Localised
    # short bits — anti-aliasing spurs on a clean curve — do not, so a lone
    # solid curve does not sprout a phantom "dashed" partner.
    if len(short) >= min_dashes:
        allx = np.concatenate([s[:, 0] for s in short])
        total_w = max(np.ptp(np.concatenate([s[:, 0] for s in strands])), 1.0)
        if np.ptp(allx) >= 0.4 * total_w:
            # The dashed sibling of a *loop* interleaves dashes between the two
            # branches, so no single stitch order is right — ink-aware ordering
            # doesn't help here (measured: net-negative), and the whole dashed
            # trace is flagged low-fidelity for guided re-tracing anyway. Keep the
            # plain nearest-endpoint stitch; ink-awareness is for the solid.
            curves.append(order_curve(short))
    return curves


def _walk_pure_loop(pixels: set) -> list[np.ndarray]:
    """Order a junction-free pixel set (open chain or closed loop) into one path."""
    deg = {p: len(_neighbors(p, pixels)) for p in pixels}
    ends = [p for p, d in deg.items() if d == 1]
    start = ends[0] if ends else next(iter(pixels))
    order = [start]
    visited = {start}
    prev, cur = None, start
    while True:
        nxts = [r for r in _neighbors(cur, pixels) if r != prev and r not in visited]
        if not nxts:
            break
        nxt = nxts[0]
        order.append(nxt)
        visited.add(nxt)
        prev, cur = cur, nxt
    arr = np.array(order, dtype=float)
    return [arr[:, ::-1]] if len(arr) >= 2 else []
