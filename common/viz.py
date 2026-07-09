"""Shared headless rendering / animation helpers for the GAMES103 reimplementation.

All rendering uses the non-interactive Agg backend so nothing ever pops a window
(the labs run fully headless and only write PNG frames + assembled GIFs).
"""
from __future__ import annotations

import os
import matplotlib

matplotlib.use("Agg")  # headless, never opens a window

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import imageio.v2 as imageio  # noqa: E402


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def frames_to_gif(frame_paths, out_gif: str, fps: int = 20):
    """Assemble a list of PNG frame paths into a looping GIF."""
    imgs = [imageio.imread(p) for p in frame_paths]
    imageio.mimsave(out_gif, imgs, fps=fps, loop=0)
    return out_gif


def set_3d_axes_equal(ax, bounds):
    """Give a 3D axis a true 1:1:1 aspect over ``bounds`` = (xmin,xmax,ymin,ymax,zmin,zmax)."""
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    try:
        ax.set_box_aspect((xmax - xmin, ymax - ymin, zmax - zmin))
    except Exception:
        pass


def depth_sort_faces(verts: np.ndarray, faces: np.ndarray, view_dir: np.ndarray):
    """Return face indices sorted back-to-front along ``view_dir`` (painter's algorithm)."""
    centroids = verts[faces].mean(axis=1)
    key = centroids @ view_dir
    return np.argsort(key)


def shade_faces(verts: np.ndarray, faces: np.ndarray, light_dir: np.ndarray,
                base_rgb, ambient: float = 0.35):
    """Flat Lambert shading -> per-face RGB array in [0,1]."""
    tri = verts[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln == 0] = 1.0
    n = n / ln
    ld = light_dir / (np.linalg.norm(light_dir) + 1e-12)
    diff = np.abs(n @ ld)
    inten = ambient + (1.0 - ambient) * diff
    base = np.asarray(base_rgb, dtype=float)
    return np.clip(inten[:, None] * base[None, :], 0, 1)
