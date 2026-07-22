# ML tracer spike — decision memo

**Question:** does a learned instance-embedding model beat the classical
strand/stitch pipeline on the corpus's real hard case (overlapping / dashed
same-colour curve bundles), and is it worth the dependency?

**Verdict: no, not as built here — keep the classical pipeline + triage.**
The idea is directionally sound (embeddings DO separate 2-3 curves cleanly),
but a from-scratch model trainable in one CPU session doesn't reliably beat
today's classical + fidelity-triage combination on the corpus's real bundles.
Don't ship it; the harness stays as an isolated, documented spike.

## What was built

`scripts/experimental/ml_tracer.py` (+ `requirements-ml.txt`, torch CPU only,
~200MB, no CUDA/mmdetection/mmcv): a discriminative pixel-embedding net
(De Brabandere et al. 2017) trained from scratch on procedurally-generated
synthetic curve bundles (free, unlimited, matches the target failure mode
directly — no borrowed LineFormer weights, which need mmdetection + a GPU +
network access to a checkpoint not available in this environment). At
inference: classical strand decomposition still finds the smooth pieces; the
net only re-answers *which pieces belong to the same physical curve*, via
agglomerative clustering of each strand's mean embedding.

## What was measured

**Synthetic (held-out, ARI — 1.0 perfect, 0.0 random):**
- v1 (dilations 1/2/4/8, receptive radius ~16px, 96px canvas, 400 steps): **0.284**
- v2 (same net, 2000 steps): **0.301** — more training barely moved it
- v3 (dilations to 16, radius ~31px, 64px canvas, 1000 steps): **0.276** — a
  materially bigger receptive field didn't help either

**Diagnosis (not just "it's bad" — checked why):** raw embedding means for 3
overlapping curves separate well (pairwise distance 2.9–3.2 vs within-cluster
spread ~0.6–0.8 — clusters *would* split correctly). At 5 curves — the real
corpus's actual bundle size (chen: 7, hoshi: 8) — three pairs land at distance
1.1–2.1, at or below the clustering threshold: the embedding space gets
crowded past ~3-4 simultaneous instances within this training budget. This
is a genuine capacity/training-budget wall, not a bug: De Brabandere's method
and LineFormer both assume far more training (many epochs over thousands of
images) and/or capacity than an interactive CPU session can give.

**Real corpus (ink-fidelity, classical vs ML clustering on the same strands):**

| panel | classical mean fidelity | ML mean fidelity |
|---|---|---|
| chen_2024 panel b (dashed Pt bundle) | **98.2** | 80.1 |
| hoshi_2013 panel b (8 curves + in-plot legend) | **92.0** | 89.0 |

Classical wins both, by a clear margin on chen. (Caveat for honesty: this
harness clusters strands from the single dark/bright ink mask directly — it
does not run the full colour-split pipeline `cvdigitize extract` actually
uses, so these two rows aren't a perfect apples-to-apples "the whole tool" A/B;
they isolate the one sub-step — strand-to-curve assignment — the ML idea
targets.)

## Why classical is still the right choice for this corpus, right now

1. **The fidelity+triage combination (Phases A/B/D) already gives you the
   actionable answer** for these exact bundles: a red-flagged score plus a
   one-click trace_assist handoff. A human resolves the genuinely ambiguous
   cases in ~15 seconds — cheaper and more reliable than a model that gets
   3-curve bundles right and 7-8-curve bundles wrong.
2. **The corpus's bundles (5-8 curves) are past this spike's working range**,
   and closing that gap needs either a properly pretrained model (LineFormer:
   real weights, GPU, mmdetection — not available/verifiable here) or far more
   training compute than an interactive session affords.
3. **No regression risk**: this stays fully isolated (`requirements-ml.txt`,
   `scripts/experimental/`) — zero impact on the shipped tool's dependencies
   or behaviour.

## If revisited later (with a GPU box)

- Try LineFormer directly (real weights) rather than training from scratch.
- Or: curriculum training (start at 2-3 instances, ramp to 8) and a larger
  embedding dimension — the 3-curve result suggests the *idea* works; it is
  the many-instance regime that needs more capacity/budget than tested here.
- Re-run this exact harness (`eval-synthetic`, `eval-real`) as the acceptance
  gate — the numbers above are the baseline to beat.
