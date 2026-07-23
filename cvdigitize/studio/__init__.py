"""CV Studio — the guided, per-paper digitization workflow.

A local stdlib-only HTTP server (:mod:`.server`) pairs with a browser client
(``tools/studio.html``) to walk one paper at a time through crop -> calibrate
-> measure -> extract -> label, reusing the existing ``cvdigitize`` functions
rather than reimplementing them. See ``STUDIO_PLAN.md`` for the full spec.
"""
