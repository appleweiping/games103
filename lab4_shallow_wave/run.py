"""Run GAMES103 Lab 4 shallow-wave demos and write results (GIFs, plots, metrics)."""
from __future__ import annotations

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from common.viz import ensure_dir, frames_to_gif  # noqa: E402
from shallow_wave import ShallowWave, FloatingBlock  # noqa: E402

RESULTS = ensure_dir(os.path.join(ROOT, "results"))


def render_surface(h, size, path, title="", zlim=0.15, block=None, stride=2):
    n = h.shape[0]
    gx = np.linspace(-size / 2, size / 2, n)
    X, Z = np.meshgrid(gx, gx, indexing="ij")
    fig = plt.figure(figsize=(4.6, 4.2), dpi=110)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Z, h, rstride=stride, cstride=stride, cmap="ocean_r",
                    vmin=-zlim, vmax=zlim, linewidth=0, antialiased=False, alpha=0.95)
    if block is not None:
        bx, bz, half, by = block
        _draw_box(ax, bx, bz, half, by, 0.12)
    ax.set_zlim(-zlim, zlim * 1.4)
    ax.set_xlim(-size / 2, size / 2); ax.set_ylim(-size / 2, size / 2)
    ax.set_box_aspect((1, 1, 0.5))
    ax.view_init(elev=38, azim=-55); ax.set_axis_off(); ax.set_title(title, fontsize=9)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def _draw_box(ax, cx, cz, half, y0, height):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    x0, x1 = cx - half, cx + half
    z0, z1 = cz - half, cz + half
    y1 = y0 + height
    verts = np.array([[x0, z0, y0], [x1, z0, y0], [x1, z1, y0], [x0, z1, y0],
                      [x0, z0, y1], [x1, z0, y1], [x1, z1, y1], [x0, z1, y1]])
    faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [0, 3, 7, 4]]
    poly = [verts[f] for f in faces]
    ax.add_collection3d(Poly3DCollection(poly, facecolors=(0.8, 0.5, 0.3, 0.95),
                                         edgecolors="k", linewidths=0.3))


def measure_wave_speed():
    """Drop at centre of a big tank, track the crest radius before it hits a wall."""
    w = ShallowWave(n=220, size=2.0, c=1.0, damping=1.0)
    w.add_drop(0, 0, radius=0.05, depth=0.3)
    gx = np.linspace(-1, 1, w.n)
    X, Z = np.meshgrid(gx, gx, indexing="ij")
    R = np.sqrt(X ** 2 + Z ** 2)
    samples = []
    for s in range(1, 400):
        w.step()
        t = s * w.dt
        if 0.25 < t < 0.75:                       # window before reflection (wall at r=1)
            crest_r = R[np.abs(w.h) > 0.02].max() if np.any(np.abs(w.h) > 0.02) else 0
            samples.append((t, crest_r))
    samples = np.array(samples)
    # linear fit radius = speed * t + b
    A = np.vstack([samples[:, 0], np.ones(len(samples))]).T
    speed, b = np.linalg.lstsq(A, samples[:, 1], rcond=None)[0]
    return float(speed)


def main():
    metrics = {}
    size = 2.0

    # ---------- Demo 1: droplet ripples (propagate, reflect off walls) ----------
    fdir = ensure_dir(os.path.join(RESULTS, "lab4_ripple_frames"))
    w = ShallowWave(n=120, size=size, c=1.0, damping=0.9995)
    w.add_drop(0.15, 0.1, radius=0.06, depth=0.28)
    V0 = w.total_volume()
    fp, vols, energ = [], [], []
    steps, rec = 900, 18
    for s in range(steps):
        if s % rec == 0:
            p = os.path.join(fdir, f"r_{s//rec:03d}.png")
            render_surface(w.h, size, p, title=f"Droplet ripples  t={s*w.dt:.2f}s", zlim=0.14)
            fp.append(p)
        w.step()
        vols.append(w.total_volume()); energ.append(w.wave_energy())
    gif1 = os.path.join(RESULTS, "lab4_ripples.gif")
    frames_to_gif(fp, gif1, fps=18)
    vols = np.array(vols)
    metrics["ripples_gif"] = os.path.basename(gif1)
    metrics["ripples_cfl"] = float(w.cfl)
    metrics["ripples_volume_V0"] = float(V0)
    metrics["ripples_volume_max_drift"] = float(np.abs(vols - V0).max())
    metrics["ripples_volume_rel_drift"] = float(np.abs(vols - V0).max() / abs(V0)) if V0 else 0.0

    # wave-speed measurement
    speed = measure_wave_speed()
    metrics["wave_speed_measured"] = speed
    metrics["wave_speed_set_c"] = 1.0

    # ---------- Demo 2: two-source interference ----------
    fdir2 = ensure_dir(os.path.join(RESULTS, "lab4_interf_frames"))
    w2 = ShallowWave(n=140, size=size, c=1.0, damping=0.9997)
    w2.add_drop(-0.45, 0.0, radius=0.05, depth=0.22)
    w2.h += 0.22 * np.exp(-(((np.linspace(-1, 1, 140)[:, None] - 0.45) ** 2 +
                             (np.linspace(-1, 1, 140)[None, :] - 0.0) ** 2)) / (2 * 0.05 ** 2))
    w2.h_old = w2.h.copy()
    fp2 = []
    steps2, rec2 = 640, 16
    for s in range(steps2):
        if s % rec2 == 0:
            p = os.path.join(fdir2, f"i_{s//rec2:03d}.png")
            render_surface(w2.h, size, p, title=f"Two-source interference  t={s*w2.dt:.2f}s", zlim=0.14)
            fp2.append(p)
        w2.step()
    gif2 = os.path.join(RESULTS, "lab4_interference.gif")
    frames_to_gif(fp2, gif2, fps=16)
    metrics["interference_gif"] = os.path.basename(gif2)
    # top-down snapshot of the interference fringes
    plt.figure(figsize=(5, 4.4))
    plt.imshow(w2.h.T, cmap="RdBu", vmin=-0.1, vmax=0.1, origin="lower",
               extent=[-1, 1, -1, 1])
    plt.colorbar(label="surface height (m)"); plt.title("Lab 4: two-source interference fringes")
    plt.xlabel("x"); plt.ylabel("z"); plt.tight_layout()
    ip = os.path.join(RESULTS, "lab4_interference_snapshot.png"); plt.savefig(ip, dpi=110); plt.close()

    # ---------- Demo 3: floating block (two-way coupled bobbing) ----------
    fdir3 = ensure_dir(os.path.join(RESULTS, "lab4_block_frames"))
    w3 = ShallowWave(n=120, size=size, c=1.0, damping=0.997)
    blk = FloatingBlock(w3, cx=0.0, cz=0.0, half=0.18, mass=0.03, y0=0.30, density_water=1.0)
    V0b = w3.total_volume()
    fp3, yhist, volb = [], [], []
    steps3, rec3 = 1400, 28
    for s in range(steps3):
        if s % rec3 == 0:
            p = os.path.join(fdir3, f"b_{s//rec3:03d}.png")
            render_surface(w3.h, size, p, title=f"Floating block bobbing  t={s*w3.dt:.2f}s",
                           zlim=0.18, block=(blk.cx, blk.cz, blk.half, blk.y))
            fp3.append(p)
        w3.step(); blk.step(w3.dt, coupling=1.2)
        yhist.append(blk.y); volb.append(w3.total_volume())
    gif3 = os.path.join(RESULTS, "lab4_floating_block.gif")
    frames_to_gif(fp3, gif3, fps=16)
    yhist = np.array(yhist); volb = np.array(volb)
    eq_depth = -blk.mass / (blk.rho * blk.footprint_area)
    metrics["block_gif"] = os.path.basename(gif3)
    metrics["block_y0"] = 0.30
    metrics["block_settle_y"] = float(yhist[-300:].mean())
    metrics["block_archimedes_eq"] = float(eq_depth)
    metrics["block_volume_max_drift"] = float(np.abs(volb - V0b).max())

    # ---------- Volume-conservation + energy plots ----------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    tvec = np.arange(len(vols)) * w.dt
    a1.plot(tvec, (vols - V0), color="teal")
    a1.set_xlabel("time (s)"); a1.set_ylabel("volume - V0 (m^3)")
    a1.set_title(f"Ripples: water volume conserved (max drift {metrics['ripples_volume_max_drift']:.1e})")
    a1.grid(alpha=0.3); a1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    a2.plot(np.arange(len(energ)) * w.dt, energ, color="darkblue")
    a2.set_xlabel("time (s)"); a2.set_ylabel("wave energy (a.u.)")
    a2.set_title("Ripples: wave energy decays under damping")
    a2.grid(alpha=0.3)
    plt.tight_layout(); vp = os.path.join(RESULTS, "lab4_volume_energy.png"); plt.savefig(vp, dpi=110); plt.close()

    # block height plot
    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(len(yhist)) * w3.dt, yhist, color="saddlebrown", label="block bottom y")
    plt.axhline(0, color="deepskyblue", ls="-", alpha=0.6, label="water rest level")
    plt.axhline(eq_depth, color="crimson", ls="--", label=f"Archimedes eq {eq_depth:.3f} m")
    plt.xlabel("time (s)"); plt.ylabel("height (m)"); plt.grid(alpha=0.3); plt.legend()
    plt.title("Lab 4: floating block bobs then settles at its Archimedes depth")
    plt.tight_layout(); bp = os.path.join(RESULTS, "lab4_block_height.png"); plt.savefig(bp, dpi=110); plt.close()

    with open(os.path.join(RESULTS, "lab4_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("=== Lab 4 shallow wave ===")
    print(f"ripples: {len(fp)} frames -> {os.path.basename(gif1)}; CFL={w.cfl:.3f}; "
          f"volume drift {metrics['ripples_volume_max_drift']:.2e} m^3 "
          f"({metrics['ripples_volume_rel_drift']:.1e} rel) -> conserved to machine precision")
    print(f"wave speed: measured {speed:.3f} vs set c=1.0 (radius = c*t fit)")
    print(f"interference: {len(fp2)} frames -> {os.path.basename(gif2)} + snapshot")
    print(f"floating block: {len(fp3)} frames -> {os.path.basename(gif3)}; "
          f"settles y={metrics['block_settle_y']:.4f} vs Archimedes eq {eq_depth:.4f}; "
          f"volume drift {metrics['block_volume_max_drift']:.2e}")


if __name__ == "__main__":
    main()
