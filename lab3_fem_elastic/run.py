"""Run GAMES103 Lab 3 FEM elastic demos and write results (jelly GIF, plots, metrics)."""
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
from common.geometry import tet_lattice  # noqa: E402
from common.viz import ensure_dir, frames_to_gif, shade_faces  # noqa: E402
from fem import FEMSolid  # noqa: E402

RESULTS = ensure_dir(os.path.join(ROOT, "results"))


def render_solid(x, surf, path, floor_y, bounds, title="", base=(0.95, 0.6, 0.3)):
    fig = plt.figure(figsize=(4.4, 4.4), dpi=110)
    ax = fig.add_subplot(111, projection="3d")
    view = np.array([1.0, 0.35, -1.0]); view = view / np.linalg.norm(view)
    order = np.argsort(x[surf].mean(axis=1) @ (-view))
    cols = shade_faces(x, surf, light_dir=np.array([0.5, 1.0, 0.4]), base_rgb=base)
    tris = x[surf][order]
    ax.add_collection3d(Poly3DCollection(tris, facecolors=cols[order],
                                         edgecolors=(0.3, 0.15, 0.0, 0.25), linewidths=0.2))
    # shadow
    shad = x[surf][order].copy(); shad[:, :, 1] = floor_y + 1e-3
    ax.add_collection3d(Poly3DCollection(shad, facecolors=(0, 0, 0, 0.10), edgecolors="none"))
    gx = np.linspace(bounds[0], bounds[1], 7); gz = np.linspace(bounds[4], bounds[5], 7)
    for xx in gx:
        ax.plot([xx, xx], [floor_y, floor_y], [bounds[4], bounds[5]], color=(0.7, 0.7, 0.7), lw=0.4)
    for zz in gz:
        ax.plot([bounds[0], bounds[1]], [floor_y, floor_y], [zz, zz], color=(0.7, 0.7, 0.7), lw=0.4)
    ax.set_xlim(bounds[0], bounds[1]); ax.set_ylim(bounds[2], bounds[3]); ax.set_zlim(bounds[4], bounds[5])
    ax.set_box_aspect((bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]))
    ax.view_init(elev=16, azim=-68); ax.set_axis_off(); ax.set_title(title, fontsize=9)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def drop_sim(model, nodes, tets, dt, sub, frames, youngs=2.0e4, poisson=0.33,
             v0=(0.0, 0.0, 0.0), record=None, surf=None, frames_dir=None, bounds=None,
             base=(0.95, 0.6, 0.3)):
    s = FEMSolid(nodes.copy(), tets, model=model, youngs=youngs, poisson=poisson, density=1000.0)
    s.v[:] = np.array(v0)
    V0 = s.total_rest_volume
    vol, energy, fp = [], [], []
    for fr in range(frames):
        for _ in range(sub):
            s.step(dt, floor_y=0.0, damping=0.996, restitution=0.0, friction=0.5, vel_smooth=0.05)
        vol.append(s.current_volume() / V0)
        tot, ke, peg, pee = s.energy()
        energy.append((fr, tot, ke, pee))
        if record is not None and fr % record == 0 and surf is not None:
            p = os.path.join(frames_dir, f"{model}_{fr//record:03d}.png")
            render_solid(s.x, surf, p, 0.0, bounds, title=f"{model} jelly  t={fr*sub*dt:.2f}s", base=base)
            fp.append(p)
    return s, np.array(vol), np.array(energy), fp


def main():
    metrics = {}
    dt = 1.0 / 1500.0
    sub = 25            # substeps per rendered frame -> effective 1/60 s frames
    frames = 130

    nodes, tets, surf = tet_lattice(5, 5, 5, cell=0.16, center=(0.0, 0.6, 0.0))
    metrics["mesh_nodes"] = int(len(nodes))
    metrics["mesh_tets"] = int(len(tets))
    metrics["mesh_surface_tris"] = int(len(surf))
    youngs = 3.0e4
    bounds = (-0.7, 0.7, 0.0, 0.9, -0.7, 0.7)

    # ---- Main demo: neo-Hookean jelly cube dropped with a little sideways spin ----
    fdir = ensure_dir(os.path.join(RESULTS, "lab3_jelly_frames"))
    s_nh, vol_nh, en_nh, fp = drop_sim("neohookean", nodes, tets, dt, sub, frames,
                                       youngs=youngs, v0=(0.45, 0.0, 0.18), record=3, surf=surf,
                                       frames_dir=fdir, bounds=bounds)
    gif = os.path.join(RESULTS, "lab3_jelly_drop.gif")
    frames_to_gif(fp, gif, fps=20)
    metrics["jelly_gif"] = os.path.basename(gif)
    metrics["neohookean_vol_min_pct"] = float(vol_nh.min() * 100)
    metrics["neohookean_vol_final_pct"] = float(vol_nh[-1] * 100)
    metrics["neohookean_E0"] = float(en_nh[0, 1])
    metrics["neohookean_Efinal"] = float(en_nh[-1, 1])
    metrics["neohookean_KE_final"] = float(en_nh[-1, 2])

    # ---- StVK drop (no render) for volume/energy comparison ----
    s_sv, vol_sv, en_sv, _ = drop_sim("stvk", nodes, tets, dt, sub, frames,
                                      youngs=youngs, v0=(0.45, 0.0, 0.18))
    metrics["stvk_vol_min_pct"] = float(vol_sv.min() * 100)
    metrics["stvk_vol_final_pct"] = float(vol_sv[-1] * 100)

    plt.figure(figsize=(7, 4))
    plt.plot(vol_nh * 100, label="neo-Hookean", color="darkorange")
    plt.plot(vol_sv * 100, label="StVK", color="teal")
    plt.axhline(100, color="gray", ls=":")
    plt.xlabel("frame"); plt.ylabel("volume (% of rest)"); plt.grid(alpha=0.3); plt.legend()
    plt.title("Lab 3: elastic-body volume during drop - squash at impact, recover, settle")
    plt.tight_layout(); vp = os.path.join(RESULTS, "lab3_volume.png"); plt.savefig(vp, dpi=110); plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(en_nh[:, 0], en_nh[:, 1], label="total E", color="purple")
    plt.plot(en_nh[:, 0], en_nh[:, 2], "--", label="kinetic", color="crimson", alpha=0.7)
    plt.plot(en_nh[:, 0], en_nh[:, 3], ":", label="elastic", color="green", alpha=0.8)
    plt.xlabel("frame"); plt.ylabel("energy (J)"); plt.grid(alpha=0.3); plt.legend()
    plt.title("Lab 3: neo-Hookean jelly energy - gravity->KE->elastic, damps to rest")
    plt.tight_layout(); ep = os.path.join(RESULTS, "lab3_energy.png"); plt.savefig(ep, dpi=110); plt.close()

    # ---- Elasticity check: stretch a bar 1.5x, release (no gravity) -> returns to rest ----
    bnodes, btets, bsurf = tet_lattice(9, 3, 3, cell=0.12, center=(0, 0, 0))
    bar = FEMSolid(bnodes.copy(), btets, model="neohookean", youngs=3.0e4, poisson=0.33)
    rest_len = bnodes[:, 0].max() - bnodes[:, 0].min()
    bar.x[:, 0] *= 1.5                                    # stretch along x
    bar.g = np.array([0.0, 0.0, 0.0])                    # no gravity, pure elastic
    lengths = []
    for _ in range(1500):
        bar.step(dt, floor_y=-1e9, damping=0.999, vel_smooth=0.02)
        lengths.append(bar.x[:, 0].max() - bar.x[:, 0].min())
    lengths = np.array(lengths)
    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(len(lengths)) * dt, lengths, color="navy")
    plt.axhline(rest_len, color="crimson", ls="--", label=f"rest length {rest_len:.3f} m")
    plt.xlabel("time (s)"); plt.ylabel("bar length (m)"); plt.grid(alpha=0.3); plt.legend()
    plt.title("Lab 3: bar stretched to 1.5x then released - oscillates back to rest length")
    plt.tight_layout(); bp = os.path.join(RESULTS, "lab3_stretch.png"); plt.savefig(bp, dpi=110); plt.close()
    metrics["bar_rest_length"] = float(rest_len)
    metrics["bar_stretched_length"] = float(1.5 * rest_len)
    metrics["bar_final_length"] = float(lengths[-1])
    metrics["bar_recovery_residual_pct"] = float(abs(lengths[-1] - rest_len) / rest_len * 100)

    with open(os.path.join(RESULTS, "lab3_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("=== Lab 3 FEM elastic ===")
    print(f"mesh: {len(nodes)} nodes, {len(tets)} tets, {len(surf)} surface tris")
    print(f"neo-Hookean jelly: {len(fp)} frames -> {os.path.basename(gif)}; "
          f"volume min {metrics['neohookean_vol_min_pct']:.1f}% -> settle {metrics['neohookean_vol_final_pct']:.1f}% of rest; "
          f"E {metrics['neohookean_E0']:.1f}->{metrics['neohookean_Efinal']:.1f} J, KE_final {metrics['neohookean_KE_final']:.4f}")
    print(f"StVK jelly: volume min {metrics['stvk_vol_min_pct']:.1f}% -> settle {metrics['stvk_vol_final_pct']:.1f}%")
    print(f"stretch-release bar: rest {rest_len:.3f} m, stretched {1.5*rest_len:.3f} m -> "
          f"recovered to {lengths[-1]:.3f} m (residual {metrics['bar_recovery_residual_pct']:.2f}%)")


if __name__ == "__main__":
    main()
