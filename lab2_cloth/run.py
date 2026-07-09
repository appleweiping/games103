"""Run GAMES103 Lab 2 cloth demos and write results (drape GIF, sphere GIF, plots, metrics)."""
from __future__ import annotations

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from common.viz import ensure_dir, frames_to_gif, shade_faces  # noqa: E402
from cloth import Cloth, make_sphere_collider  # noqa: E402

RESULTS = ensure_dir(os.path.join(ROOT, "results"))


def quad_faces(n):
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            b = (i + 1) * n + j
            c = (i + 1) * n + (j + 1)
            d = i * n + (j + 1)
            faces.append((a, b, c, d))
    return np.array(faces, dtype=int)


def sphere_wire(center, radius, n=16):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.ones_like(u), np.cos(v))
    z = center[2] + radius * np.outer(np.sin(u), np.sin(v))
    return x, y, z


def render_cloth(pos, n, faces, path, bounds, title="", sphere=None, base=(0.85, 0.35, 0.45)):
    fig = plt.figure(figsize=(4.4, 4.4), dpi=110)
    ax = fig.add_subplot(111, projection="3d")
    tris = pos[faces]                                   # (F,4,3) quads
    # shade by quad normal
    nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    ln = np.linalg.norm(nrm, axis=1, keepdims=True); ln[ln == 0] = 1
    nrm = nrm / ln
    ld = np.array([0.4, 0.8, 0.4]); ld = ld / np.linalg.norm(ld)
    inten = 0.4 + 0.6 * np.abs(nrm @ ld)
    cols = np.clip(inten[:, None] * np.array(base)[None, :], 0, 1)
    order = np.argsort(tris.mean(axis=1) @ np.array([1, 0.3, -1.0]))
    pc = Poly3DCollection(tris[order], facecolors=cols[order], edgecolors=(0.2, 0.1, 0.1, 0.25),
                          linewidths=0.15)
    ax.add_collection3d(pc)
    if sphere is not None:
        sx, sy, sz = sphere_wire(sphere[0], sphere[1])
        ax.plot_surface(sx, sy, sz, color=(0.5, 0.6, 0.75), alpha=0.85, linewidth=0, shade=True)
    ax.set_xlim(bounds[0], bounds[1]); ax.set_ylim(bounds[2], bounds[3]); ax.set_zlim(bounds[4], bounds[5])
    ax.set_box_aspect((bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]))
    ax.view_init(elev=18, azim=-60)
    ax.set_axis_off(); ax.set_title(title, fontsize=9)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main():
    metrics = {}
    dt = 1.0 / 60.0
    n = 21
    faces = quad_faces(n)

    # ---------- Demo 1: hanging cloth drape (implicit Euler) ----------
    frames_dir = ensure_dir(os.path.join(RESULTS, "lab2_drape_frames"))
    c = Cloth(n=n, size=2.0, height=2.0, stiffness=8000.0)
    bounds = (-1.2, 1.2, -0.6, 2.1, -1.2, 1.2)
    fp, energies = [], []
    steps, rec = 300, 6
    for s in range(steps):
        if s % rec == 0:
            p = os.path.join(frames_dir, f"d_{s//rec:03d}.png")
            render_cloth(c.pos, n, faces, p, bounds, title=f"Cloth drape (implicit)  t={s*dt:.2f}s")
            fp.append(p)
            tot, ke, peg, pee = c.energy()
            energies.append((s * dt, tot, ke, pee))
        c.implicit_step(dt)
    gif1 = os.path.join(RESULTS, "lab2_cloth_drape.gif")
    frames_to_gif(fp, gif1, fps=18)
    energies = np.array(energies)
    metrics["drape_gif"] = os.path.basename(gif1)
    metrics["drape_final_minY"] = float(c.pos[:, 1].min())
    metrics["drape_final_strain_pct"] = float(c.mean_spring_strain() * 100)
    metrics["drape_final_KE"] = float(c.energy()[1])

    # ---------- Demo 2: cloth dropping onto a sphere ----------
    frames_dir2 = ensure_dir(os.path.join(RESULTS, "lab2_sphere_frames"))
    center, radius = (0.0, 0.15, 0.0), 0.6
    c2 = Cloth(n=25, size=2.0, height=1.15, stiffness=6000.0, pinned=[])
    col = make_sphere_collider(center, radius, friction=0.4)
    faces2 = quad_faces(25)
    bounds2 = (-1.1, 1.1, -0.8, 1.3, -1.1, 1.1)
    fp2 = []
    steps2, rec2 = 360, 8
    for s in range(steps2):
        if s % rec2 == 0:
            p = os.path.join(frames_dir2, f"s_{s//rec2:03d}.png")
            render_cloth(c2.pos, 25, faces2, p, bounds2, sphere=(np.array(center), radius),
                         title=f"Cloth over sphere  t={s*dt:.2f}s", base=(0.55, 0.75, 0.45))
            fp2.append(p)
        c2.implicit_step(dt, collide=col)
    gif2 = os.path.join(RESULTS, "lab2_cloth_sphere.gif")
    frames_to_gif(fp2, gif2, fps=18)
    d = np.linalg.norm(c2.pos - np.array(center), axis=1)
    metrics["sphere_gif"] = os.path.basename(gif2)
    metrics["sphere_radius"] = radius
    metrics["sphere_min_node_dist"] = float(d.min())
    metrics["sphere_penetrating_nodes"] = int(np.sum(d < radius - 1e-4))

    # ---------- Demo 3: explicit vs implicit stability at the SAME dt ----------
    stiff = 8000.0
    ce = Cloth(n=n, stiffness=stiff)
    ci = Cloth(n=n, stiffness=stiff)
    max_e, max_i, tvec = [], [], []
    exp_blow_step = None
    for s in range(180):
        ce.explicit_step(dt); ci.implicit_step(dt)
        me = np.abs(ce.pos).max() if np.all(np.isfinite(ce.pos)) else np.inf
        mi = np.abs(ci.pos).max()
        max_e.append(me); max_i.append(mi); tvec.append(s * dt)
        if exp_blow_step is None and (not np.isfinite(me) or me > 1e3):
            exp_blow_step = s
    plt.figure(figsize=(7, 4))
    plt.semilogy(tvec, np.clip(max_e, 1e-3, 1e12), label="explicit Euler", color="crimson")
    plt.semilogy(tvec, np.clip(max_i, 1e-3, 1e12), label="implicit Euler (Jacobi+Chebyshev)", color="navy")
    plt.axhline(2.0, color="gray", ls=":", label="cloth extent ~2 m")
    plt.xlabel("time (s)"); plt.ylabel("max |coordinate|  (log)"); plt.grid(alpha=0.3, which="both")
    plt.title(f"Lab 2: same dt={dt:.3f}s, k={stiff:.0f} - explicit diverges, implicit stable")
    plt.legend(); plt.tight_layout()
    sp = os.path.join(RESULTS, "lab2_stability.png"); plt.savefig(sp, dpi=110); plt.close()
    metrics["stability_dt"] = dt
    metrics["stability_stiffness"] = stiff
    metrics["explicit_blowup_step"] = exp_blow_step
    metrics["explicit_max_coord_at_end"] = float(max_e[-1]) if np.isfinite(max_e[-1]) else 1e30
    metrics["implicit_max_coord_at_end"] = float(max_i[-1])

    # ---------- Energy of the drape (decays to static equilibrium) ----------
    plt.figure(figsize=(7, 4))
    plt.plot(energies[:, 0], energies[:, 1], label="total E", color="purple")
    plt.plot(energies[:, 0], energies[:, 2], "--", label="kinetic", color="crimson", alpha=0.6)
    plt.plot(energies[:, 0], energies[:, 3], ":", label="elastic", color="green", alpha=0.8)
    plt.xlabel("time (s)"); plt.ylabel("energy (J)"); plt.grid(alpha=0.3); plt.legend()
    plt.title("Lab 2: draping cloth energy - kinetic decays, settles to static drape")
    plt.tight_layout()
    ep = os.path.join(RESULTS, "lab2_energy.png"); plt.savefig(ep, dpi=110); plt.close()

    with open(os.path.join(RESULTS, "lab2_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("=== Lab 2 cloth ===")
    print(f"drape (implicit): {len(fp)} frames -> {os.path.basename(gif1)}; "
          f"final minY={metrics['drape_final_minY']:.3f}, mean strain={metrics['drape_final_strain_pct']:.3f}%, "
          f"KE settled to {metrics['drape_final_KE']:.4f} J")
    print(f"cloth-over-sphere: {len(fp2)} frames -> {os.path.basename(gif2)}; "
          f"min node dist={metrics['sphere_min_node_dist']:.3f} (R={radius}), "
          f"penetrating nodes={metrics['sphere_penetrating_nodes']}")
    print(f"stability @ dt={dt:.4f}s k={stiff:.0f}: explicit blew up at step {exp_blow_step} "
          f"(max coord {metrics['explicit_max_coord_at_end']:.2e}); "
          f"implicit stayed bounded (max coord {metrics['implicit_max_coord_at_end']:.3f})")


if __name__ == "__main__":
    main()
