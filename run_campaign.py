"""Standalone campaign: calls the same focus_beam tool directly (no LLM).
Use it to test the bench; the graded table should come from the agent's own
run, which writes to the same results/experiments.jsonl."""
import json
from focus_tool import focus_beam

CAMPAIGN = [
    dict(macropixel_size=8, iterations=300, reason="coarse check: 16 macropixels, fast"),
    dict(macropixel_size=4, iterations=600, reason="64 macropixels, ~10x iterations"),
    dict(macropixel_size=2, iterations=1500, reason="256 macropixels for higher enhancement"),
    dict(macropixel_size=2, iterations=1500, focus_size=4, reason="larger focus region (2x2 speckles)"),
    dict(macropixel_size=2, iterations=1500, bg_inner=3, bg_outer=8, n_bg_regions=4,
         reason="tighter background ring, 4 sectors"),
    dict(macropixel_size=1, iterations=5000, reason="all 1024 mirrors independent, long run"),
]
for c in CAMPAIGN:
    r = json.loads(focus_beam(**c))
    print(r["run_id"], r["settings"]["macropixel_size"], r["enhancement"], r["enhancement_vs_random"], r["theory_binary_amplitude"])
