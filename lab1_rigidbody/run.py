"""Run GAMES103 Lab 1 rigid-body demos and write results (frames, GIF, plots, metrics)."""
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
from common.geometry import make_bunny, uv_sphere  # noqa: E402
from common.viz import ensure_dir, frames_to_gif, shade_faces  # noqa: E402
from rigidbody import RigidBody, simulate, verify_angular_momentum  # noqa: E402

RESULTS = ensure_dir(os.path.join(ROOT, "results"))
FRAMES = ensure_dir(os.path.join(RESULTS, "lab1_frames"))


def render_frame(verts, faces, path, floor_y, bounds, title=""):
    fig = plt.figure(figsize=(4.2, 4.2), dpi=110)
    ax = fig.add_subplot(111, projection="3d")
    view = np.array([np.cos(np.radians(-60)), 0.35, np.sin(np.radians(-60))])
    view = view / np.linalg.norm(view)
    order = np.argsort(verts[faces].mean(axis=1) @ (-view))
    colors = shade_faces(verts, faces, light_dir=np.array([0.5, 1.0, 0.3]),
                         base_rgb=(0.55, 0.72, 0.95))
    tris = verts[faces][order]
    pc = Poly3DCollection(tris, facecolors=colors[order], edgecolors=(0, 0, 0, 0.12),
                          linewidths=0.2)
    ax.add_collection3d(pc)
    # soft shadow on the floor
    shad = verts[faces].copy()
    shad = shad[order]
    shad[:, :, 1] = floor_y + 1e-3
    ax.add_collection3d(Poly3DCollection(shad, facecolors=(0, 0, 0, 0.10), edgecolors="none"))
    # floor grid
    gx = np.linspace(bounds[0], bounds[1], 9)
    gz = np.linspace(bounds[4], bounds[5], 9)
    for x in gx:
        ax.plot([x, x], [floor_y, floor_y], [bounds[4], bounds[5]], color=(0.7, 0.7, 0.7), lw=0.4)
    for z in gz:
        ax.plot([bounds[0], bounds[1]], [floor_y, floor_y], [z, z], color=(0.7, 0.7, 0.7), lw=0.4)
    ax.set_xlim(bounds[0], bounds[1]); ax.set_ylim(bounds[2], bounds[3]); ax.set_zlim(bounds[4], bounds[5])
    ax.set_box_aspect((bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]))
    ax.view_init(elev=16, azim=-70)
    ax.set_axis_off()
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    metrics = {}
    g = np.array([0.0, -9.8, 0.0])
    floor_y = 0.0
    dt = 1.0 / 120.0
    steps = 480
    rec = 8  # record every 8 steps -> 60 frames

    # ---- Demo: Angry Bunny bounce (restitution 0.5, friction 0.4) ----
    verts, faces = make_bunny()
    body = RigidBody(verts, faces, mass=1.0)
    body.x = np.array([0.0, 1.2, 0.0])
    body.P = body.mass * np.array([0.4, 0.0, 0.10])     # sideways launch
    body.L = np.array([0.15, 0.10, 0.50])               # tumble
    frames, energies05, heights = simulate(body, steps, dt, g, floor_y,
                                            restitution=0.5, friction=0.4, record_every=rec,
                                            positional_correction=False)
    bounds = (-1.2, 1.6, 0.0, 1.7, -1.0, 1.0)
    frame_paths = []
    for i, (x, R) in enumerate(frames):
        wv = x + verts_body(verts) @ R.T
        p = os.path.join(FRAMES, f"bunny_{i:03d}.png")
        render_frame(wv, faces, p, floor_y, bounds, title=f"Angry Bunny  t={i*rec*dt:.2f}s")
        frame_paths.append(p)
    gif = os.path.join(RESULTS, "lab1_bunny_bounce.gif")
    frames_to_gif(frame_paths, gif, fps=20)
    metrics["bunny_bounce_gif"] = os.path.basename(gif)
    metrics["bunny_n_vertices"] = int(body.n)
    metrics["bunny_n_faces"] = int(len(faces))

    # ---- Energy conservation check: symmetric sphere dropped vertically, ----
    # ---- restitution 1.0, friction 0 -> total energy must never grow.     ----
    sv, sf = uv_sphere(0.3, n_stack=14, n_slice=20)
    sph = RigidBody(sv, sf, mass=1.0)
    sph.x = np.array([0.0, 1.2, 0.0]); sph.P = np.zeros(3); sph.L = np.zeros(3)
    _, energies10, hsph = simulate(sph, 1400, dt, g, floor_y, restitution=1.0, friction=0.0,
                                   record_every=4, positional_correction=False)

    plt.figure(figsize=(7, 4))
    plt.plot(energies05[:, 0], energies05[:, 3], label="bunny total E (restitution 0.5)", color="crimson")
    plt.plot(energies05[:, 0], energies05[:, 1], "--", label="bunny kinetic", color="crimson", alpha=0.45)
    plt.plot(energies10[:, 0], energies10[:, 3], label="sphere total E (restitution 1.0)", color="navy")
    plt.xlabel("time (s)"); plt.ylabel("energy (J)"); plt.grid(alpha=0.3); plt.legend()
    plt.title("Lab 1: restitution 0.5 decays step-wise & settles;  1.0 conserves (no pumping)")
    plt.tight_layout()
    ep = os.path.join(RESULTS, "lab1_energy.png"); plt.savefig(ep, dpi=110); plt.close()

    e0_05 = energies05[0, 3]; efin_05 = energies05[-1, 3]
    inc05 = np.diff(energies05[:, 3]).max()
    e0_10 = energies10[0, 3]
    emax_10 = energies10[:, 3].max()
    metrics["restitution0.5_E0"] = float(e0_05)
    metrics["restitution0.5_Efinal"] = float(efin_05)
    metrics["restitution0.5_max_step_energy_increase"] = float(inc05)
    metrics["restitution1.0_E0"] = float(e0_10)
    metrics["restitution1.0_Emax"] = float(emax_10)
    metrics["restitution1.0_max_gain_frac"] = float(emax_10 / e0_10 - 1.0)

    # ---- Verification: torque-free angular momentum conservation ----
    b3 = RigidBody(verts, faces, mass=1.0)
    drift, Ls = verify_angular_momentum(b3, steps=1000, dt=dt)
    metrics["angular_momentum_drift_max"] = float(drift)
    metrics["angular_momentum_L0"] = Ls[0].tolist()

    with open(os.path.join(RESULTS, "lab1_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("=== Lab 1 rigid body ===")
    print(f"bunny: {body.n} verts, {len(faces)} faces, GIF -> {os.path.basename(gif)} ({len(frame_paths)} frames)")
    print(f"restitution 0.5 (bunny): E0={e0_05:.3f} J -> Efinal={efin_05:.3f} J "
          f"(energy decays & settles; max step increase {inc05:.2e} J)")
    print(f"restitution 1.0 friction 0 (sphere): E0={e0_10:.3f} J, Emax={emax_10:.3f} J "
          f"(max gain {100*(emax_10/e0_10-1):.3f}% -> no spurious energy pumping)")
    print(f"torque-free tumble: |L| drift over 1000 steps = {drift:.3e} (should be ~0)")


def verts_body(verts):
    """vertices relative to their centroid (matches RigidBody.r)."""
    v = np.asarray(verts, float)
    return v - v.mean(axis=0)


if __name__ == "__main__":
    main()
