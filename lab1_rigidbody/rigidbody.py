"""GAMES103 Lab 1 - Rigid Body Simulation (impulse-based, "Angry Bunny").

Reimplements the Unity/C# lab core in NumPy, headless. Physics:
  * state = center-of-mass x, linear momentum P, orientation R, angular momentum L
  * torque-free rotational dynamics  w = I(t)^-1 L,  I(t) = R I_ref R^T
  * impulse-based collision response against a floor plane with Coulomb friction
    and a coefficient of restitution (the exact GAMES103 impulse formulation).

Verifications produced:
  * free zero-gravity tumble conserves angular momentum L (torque-free precession)
  * a bouncing body with restitution<1 loses energy step-wise at each bounce and
    settles; with restitution=1, friction=0 it (near) conserves energy.
"""
from __future__ import annotations

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.geometry import make_bunny, cube_mesh  # noqa: E402


def skew(a):
    return np.array([[0, -a[2], a[1]],
                     [a[2], 0, -a[0]],
                     [-a[1], a[0], 0.0]])


def rodrigues(w, dt):
    """Rotation matrix for rotating by angular velocity w over time dt."""
    theta = np.linalg.norm(w) * dt
    if theta < 1e-12:
        return np.eye(3)
    axis = w / np.linalg.norm(w)
    K = skew(axis)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def orthonormalize(R):
    U, _, Vt = np.linalg.svd(R)
    return U @ Vt


class RigidBody:
    def __init__(self, verts, faces, mass=1.0):
        self.faces = faces
        v = np.asarray(verts, float)
        self.n = len(v)
        m_i = mass / self.n
        self.mass = mass
        com = v.mean(axis=0)
        self.r = v - com                      # body-frame vertex offsets (COM at origin)
        # reference inertia tensor (body frame)
        I = np.zeros((3, 3))
        for ri in self.r:
            I += m_i * (np.dot(ri, ri) * np.eye(3) - np.outer(ri, ri))
        self.I_ref = I
        self.I_ref_inv = np.linalg.inv(I)
        # dynamic state
        self.x = com.copy()
        self.P = np.zeros(3)                   # linear momentum
        self.R = np.eye(3)
        self.L = np.zeros(3)                   # angular momentum (world)

    # --- derived quantities ---
    @property
    def v(self):
        return self.P / self.mass

    def I_world_inv(self):
        return self.R @ self.I_ref_inv @ self.R.T

    def w(self):
        return self.I_world_inv() @ self.L

    def world_verts(self):
        return self.x + self.r @ self.R.T

    def kinetic_energy(self):
        lin = 0.5 * self.mass * np.dot(self.v, self.v)
        w = self.w()
        I_world = self.R @ self.I_ref @ self.R.T
        rot = 0.5 * np.dot(w, I_world @ w)
        return lin + rot

    def potential_energy(self, g, floor_y):
        # sum m_i g (y - floor) over vertices == M g (y_com - floor)
        return self.mass * (-g[1]) * (self.x[1] - floor_y)

    # --- dynamics ---
    def collision_impulse(self, floor_y, normal, restitution, friction, contact_eps=1e-4):
        n = np.asarray(normal, float)
        n = n / np.linalg.norm(n)
        Rr = self.r @ self.R.T                 # world offsets
        world = self.x + Rr
        # point velocities
        w = self.w()
        vel = self.v + np.cross(np.tile(w, (self.n, 1)), Rr)
        # contact = at-or-below the plane (within eps) AND moving into it. The eps
        # is essential: the positional correction rests vertices exactly on the
        # plane, so a strict < test would miss resting contact and let gravity
        # accumulate velocity every step.
        penetrating = (world[:, 1] < floor_y + contact_eps) & (vel @ n < 0)
        if not np.any(penetrating):
            return False
        # averaged contact (GAMES103): mean offset of colliding verts
        rr = Rr[penetrating].mean(axis=0)
        v_point = self.v + np.cross(w, rr)
        vn = np.dot(v_point, n) * n
        vt = v_point - vn
        vt_norm = np.linalg.norm(vt)
        vn_new = -restitution * vn
        if vt_norm > 1e-8:
            a = max(1.0 - friction * (1.0 + restitution) * np.linalg.norm(vn) / vt_norm, 0.0)
        else:
            a = 0.0
        vt_new = a * vt
        v_target = vn_new + vt_new
        Iinv = self.I_world_inv()
        rr_x = skew(rr)
        K = (1.0 / self.mass) * np.eye(3) - rr_x @ Iinv @ rr_x
        J = np.linalg.solve(K, v_target - v_point)
        self.P += J
        self.L += np.cross(rr, J)
        return True

    def step(self, dt, g, floor_y=None, normal=(0, 1, 0), restitution=0.5, friction=0.4,
             positional_correction=True):
        # 1. collision impulse FIRST, on the true incoming velocity. Resolving the
        #    bounce before adding this step's gravity avoids reflecting the g*dt
        #    velocity increment, which would otherwise pump energy at restitution=1.
        if floor_y is not None:
            self.collision_impulse(floor_y, normal, restitution, friction)
        # 2. gravity (linear only; no torque about the COM)
        self.P += self.mass * np.asarray(g, float) * dt
        # 3. integrate position + orientation
        self.x = self.x + self.v * dt
        self.R = orthonormalize(rodrigues(self.w(), dt) @ self.R)
        # 4. optional non-penetration position correction (visual; off for energy runs)
        if floor_y is not None and positional_correction:
            ys = (self.x + self.r @ self.R.T)[:, 1]
            depth = floor_y - ys.min()
            if depth > 0:
                self.x[1] += depth


# --------------------------------------------------------------------------
def simulate(body, steps, dt, g, floor_y, restitution, friction, record_every=1,
             positional_correction=False):
    frames, energies, heights = [], [], []
    for s in range(steps):
        if s % record_every == 0:
            frames.append((body.x.copy(), body.R.copy()))
            ke = body.kinetic_energy()
            pe = body.potential_energy(g, floor_y)
            energies.append((s * dt, ke, pe, ke + pe))
            heights.append((s * dt, body.x[1]))
        body.step(dt, g, floor_y=floor_y, restitution=restitution, friction=friction,
                  positional_correction=positional_correction)
    return frames, np.array(energies), np.array(heights)


def verify_angular_momentum(body, steps, dt):
    """Torque-free tumble in zero gravity: angular momentum L must stay constant."""
    body.L = np.array([0.4, 1.0, 0.2])   # give it spin
    Ls = []
    for _ in range(steps):
        Ls.append(body.L.copy())
        body.step(dt, g=(0, 0, 0), floor_y=None)  # no gravity, no floor -> torque free
    Ls = np.array(Ls)
    drift = np.linalg.norm(Ls - Ls[0], axis=1).max()
    return drift, Ls
