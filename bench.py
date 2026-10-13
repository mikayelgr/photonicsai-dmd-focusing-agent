"""Simulated DMD -> scattering medium -> camera bench and the focusing search.

Camera model:  I = |T E_in|^2 + noise
  * E_in   : binary amplitude pattern on the DMD (mirror on = 1, off = 0)
  * T      : complex transmission matrix (camera pixels x DMD mirrors), each
             column is a speckle field with a finite grain size
  * noise  : Poisson shot noise + Gaussian read noise, so two shots of the
             same pattern are never identical
"""
from __future__ import annotations

import numpy as np


class SimulatedBench:
    def __init__(self, dmd_size=32, cam_size=64, grain=2.0, photons=2000.0,
                 read_noise=2.0, seed=0):
        self.dmd_size = dmd_size
        self.cam_size = cam_size
        self.photons = photons          # mean photons / camera pixel, all mirrors on
        self.read_noise = read_noise    # std of Gaussian read noise (counts)
        self.rng = np.random.default_rng(seed)
        self.T = self._make_tm(grain)
        # normalise so all-on pattern gives `photons` mean counts per pixel
        mean_all_on = np.mean(np.abs(self.T.sum(axis=1)) ** 2)
        self.T *= np.sqrt(photons / mean_all_on)
        self.n_shots = 0

    def _make_tm(self, grain):
        n_in = self.dmd_size ** 2
        c = self.cam_size
        g = (self.rng.standard_normal((c, c, n_in))
             + 1j * self.rng.standard_normal((c, c, n_in))).astype(np.complex64)
        # circular pupil in k-space sets the speckle grain size (~grain pixels)
        fx = np.fft.fftfreq(c)
        kx, ky = np.meshgrid(fx, fx, indexing="ij")
        pupil = (np.sqrt(kx ** 2 + ky ** 2) <= 1.0 / (2.0 * grain)).astype(np.float32)
        g = np.fft.ifft2(np.fft.fft2(g, axes=(0, 1)) * pupil[..., None], axes=(0, 1))
        return g.reshape(c * c, n_in).astype(np.complex64)

    def shoot(self, dmd_pattern: np.ndarray) -> np.ndarray:
        """One camera exposure for a DMD mirror pattern (dmd_size x dmd_size, 0/1)."""
        self.n_shots += 1
        e_in = dmd_pattern.reshape(-1).astype(np.complex64)
        ideal = np.abs(self.T @ e_in) ** 2
        img = self.rng.poisson(ideal).astype(np.float64)
        img += self.rng.normal(0.0, self.read_noise, img.shape)
        return img.reshape(self.cam_size, self.cam_size)


def make_regions(cam_size, focus_size, bg_inner, bg_outer, n_bg_regions, center=None):
    """Square focus region + proximal background annulus split into angular sectors."""
    if center is None:
        center = (cam_size // 2, cam_size // 2)
    yy, xx = np.mgrid[0:cam_size, 0:cam_size]
    dy, dx = yy - center[0], xx - center[1]
    half = focus_size / 2.0
    focus = (np.abs(dy + 0.5) <= half) & (np.abs(dx + 0.5) <= half)
    r = np.sqrt(dy ** 2 + dx ** 2)
    ring = (r >= bg_inner) & (r <= bg_outer) & ~focus
    ang = (np.arctan2(dy, dx) + np.pi) / (2 * np.pi)
    sector = np.minimum((ang * n_bg_regions).astype(int), n_bg_regions - 1)
    bg = [ring & (sector == k) for k in range(n_bg_regions)]
    return focus, bg


def enhancement(img, focus, bg_regions):
    i_focus = img[focus].mean()
    i_bg = np.mean([img[m].mean() for m in bg_regions])
    return float(i_focus / max(i_bg, 1e-9)), float(i_focus), float(i_bg)


def expand(macro, m):
    return np.kron(macro, np.ones((m, m), dtype=macro.dtype))


def run_focus_search(bench: SimulatedBench, macropixel_size=4, focus_size=2,
                     bg_inner=4, bg_outer=12, n_bg_regions=8, iterations=1500,
                     n_flip_start=None, seed=1):
    """Random-flip hill climb on DMD macropixels, maximising focus/background."""
    rng = np.random.default_rng(seed)
    if bench.dmd_size % macropixel_size:
        raise ValueError("macropixel_size must divide the DMD size")
    n_side = bench.dmd_size // macropixel_size
    n_macro = n_side * n_side
    focus, bg = make_regions(bench.cam_size, focus_size, bg_inner, bg_outer, n_bg_regions)

    pattern = rng.integers(0, 2, (n_side, n_side)).astype(np.uint8)
    img0 = bench.shoot(expand(pattern, macropixel_size))
    best_eta, _, _ = enhancement(img0, focus, bg)
    initial_eta = best_eta
    history = [best_eta]
    if n_flip_start is None:
        n_flip_start = max(1, n_macro // 8)

    for it in range(1, iterations + 1):
        # number of flipped macropixels shrinks from n_flip_start to 1
        k = max(1, int(round(n_flip_start * (1 - it / iterations))))
        idx = rng.choice(n_macro, size=k, replace=False)
        trial = pattern.copy().reshape(-1)
        trial[idx] ^= 1
        trial = trial.reshape(n_side, n_side)
        eta, _, _ = enhancement(bench.shoot(expand(trial, macropixel_size)), focus, bg)
        if eta > best_eta:
            pattern, best_eta = trial, eta
        elif it % 25 == 0:
            # re-measure the kept pattern so one lucky noisy shot cannot lock in
            best_eta, _, _ = enhancement(bench.shoot(expand(pattern, macropixel_size)), focus, bg)
        history.append(best_eta)

    # final value: average of 5 fresh shots of the kept pattern
    final_shots = [bench.shoot(expand(pattern, macropixel_size)) for _ in range(5)]
    final_img = final_shots[0]
    final_eta = float(np.mean([enhancement(s, focus, bg)[0] for s in final_shots]))

    # classic reference (Vellekoop & Mosk): optimised focus intensity over the
    # focus intensity averaged across random DMD patterns
    ref = [bench.shoot(expand(rng.integers(0, 2, (n_side, n_side)).astype(np.uint8),
                              macropixel_size))[focus].mean() for _ in range(20)]
    i_focus_opt = float(np.mean([s[focus].mean() for s in final_shots]))
    eta_vs_random = i_focus_opt / max(float(np.mean(ref)), 1e-9)

    return {
        "enhancement_vs_random": float(eta_vs_random),
        "settings": dict(macropixel_size=macropixel_size, n_macropixels=n_macro,
                         focus_size=focus_size, bg_inner=bg_inner, bg_outer=bg_outer,
                         n_bg_regions=n_bg_regions, iterations=iterations,
                         n_flip_start=n_flip_start, seed=seed),
        "initial_enhancement": float(initial_eta),
        "enhancement": final_eta,
        "theory_binary_amplitude": float(n_macro / (2 * np.pi) + 0.5),
        "history": history,
        "initial_image": img0,
        "final_image": final_img,
        "pattern": expand(pattern, macropixel_size),
        "focus_mask": focus,
        "bg_masks": bg,
    }
