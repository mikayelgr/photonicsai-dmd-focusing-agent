"""The focusing tool the executor calls.

Every call: runs the search on the simulated bench, saves camera images to
results/<run_id>/, appends a row to results/experiments.jsonl and returns a
JSON summary (settings + enhancement + image paths) to the agent.
"""
from __future__ import annotations

import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bench import SimulatedBench, run_focus_search

RESULTS_DIR = os.environ.get("FOCUS_RESULTS_DIR", "results")
_BENCH = None


def get_bench():
    """One bench (one scattering medium) shared by all runs in a session."""
    global _BENCH
    if _BENCH is None:
        _BENCH = SimulatedBench(dmd_size=32, cam_size=64, grain=2.0,
                                photons=2000.0, read_noise=2.0, seed=0)
    return _BENCH


def _save_images(res, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    vmax = res["final_image"].max()
    for key in ("initial_image", "final_image"):
        fig, ax = plt.subplots(figsize=(3.2, 3.2))
        ax.imshow(res[key], cmap="inferno", vmin=0, vmax=vmax)
        ax.contour(res["focus_mask"], levels=[0.5], colors="cyan", linewidths=0.6)
        ax.contour(np.any(res["bg_masks"], axis=0), levels=[0.5], colors="white",
                   linewidths=0.4, linestyles="dashed")
        ax.set_axis_off()
        p = os.path.join(out_dir, f"{key}.png")
        fig.savefig(p, dpi=150, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        paths[key] = p
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.imshow(res["pattern"], cmap="gray")
    ax.set_axis_off()
    p = os.path.join(out_dir, "dmd_pattern.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    paths["dmd_pattern"] = p
    fig, ax = plt.subplots(figsize=(4, 2.6))
    ax.plot(res["history"], lw=1)
    ax.set_xlabel("iteration")
    ax.set_ylabel("enhancement")
    ax.grid(alpha=0.3)
    p = os.path.join(out_dir, "history.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths["history"] = p
    np.save(os.path.join(out_dir, "final_image.npy"), res["final_image"])
    return paths


def focus_beam(macropixel_size: int = 4, focus_size: int = 2, bg_inner: int = 4,
               bg_outer: int = 12, n_bg_regions: int = 8, iterations: int = 1500,
               seed: int = 1, reason: str = "") -> str:
    """Run the DMD focusing search on the simulated bench.

    Args:
        macropixel_size: side of a DMD macropixel in mirrors (1, 2, 4, 8, 16; DMD is 32x32).
        focus_size: side of the square focus region in camera pixels (speckle grain ~2 px).
        bg_inner: inner radius (px) of the proximal background annulus.
        bg_outer: outer radius (px) of the background annulus (camera is 64x64).
        n_bg_regions: number of angular sectors the background is split into.
        iterations: number of flip-and-measure iterations (one camera shot each).
        seed: random seed for the initial pattern and the flips.
        reason: one sentence on why these settings were chosen.
    Returns:
        JSON string with settings, enhancement and the paths of the camera images.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log = os.path.join(RESULTS_DIR, "experiments.jsonl")
    run_id = sum(1 for _ in open(log)) + 1 if os.path.exists(log) else 1
    t0 = time.time()
    res = run_focus_search(get_bench(), macropixel_size=macropixel_size,
                           focus_size=focus_size, bg_inner=bg_inner, bg_outer=bg_outer,
                           n_bg_regions=n_bg_regions, iterations=iterations, seed=seed)
    images = _save_images(res, os.path.join(RESULTS_DIR, f"run_{run_id:02d}"))
    row = {
        "run_id": run_id,
        "settings": res["settings"],
        "initial_enhancement": round(res["initial_enhancement"], 2),
        "enhancement": round(res["enhancement"], 2),
        "enhancement_vs_random": round(res["enhancement_vs_random"], 2),
        "theory_binary_amplitude": round(res["theory_binary_amplitude"], 2),
        "seconds": round(time.time() - t0, 1),
        "reason": reason,
        "images": images,
    }
    with open(log, "a") as f:
        f.write(json.dumps(row) + "\n")
    return json.dumps(row)


# LangChain / LangGraph tool object for the executor
try:
    from langchain_core.tools import tool
    focus_beam_tool = tool(focus_beam)
except ImportError:  # the bench works without LangChain installed
    focus_beam_tool = None
