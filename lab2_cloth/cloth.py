"""GAMES103 Lab 2 - Mass-Spring Cloth Simulation.

Three integrators over the same structural/shear/bend spring network:
  * explicit_step  - symplectic (forward) Euler; blows up for stiff springs / big dt
  * implicit_step  - implicit Euler solved with the GAMES103 Jacobi + Chebyshev
                     acceleration scheme; unconditionally stable at large dt
  * pbd_step       - Position-Based Dynamics; Gauss-Seidel distance-constraint
                     projection (stiffness independent of dt)

Also supports sphere and floor collision so the cloth can drape over an object.
"""
from __future__ import annotations

import numpy as np


def build_grid(n=21, size=2.0, height=2.0):
    """(n x n) cloth in the xz-ish plane, hung horizontally at y=height.

    Returns positions (N,3), and the (i,j) grid index of each particle.
    """
    xs = np.linspace(-size / 2, size / 2, n)
    zs = np.linspace(-size / 2, size / 2, n)
    pos = np.zeros((n * n, 3))
    for i in range(n):
        for j in range(n):
            pos[i * n + j] = [xs[i], height, zs[j]]
    return pos, n


def build_springs(n):
    """Structural (4-neighbour), shear (diagonal), bend (2-apart) springs.

    Returns edges (E,2) int and a per-edge stiffness scale (E,) so bend springs
    can be softer.
    """
    edges, scale = [], []

    def idx(i, j):
        return i * n + j

    for i in range(n):
        for j in range(n):
            # structural
            if i + 1 < n:
                edges.append((idx(i, j), idx(i + 1, j))); scale.append(1.0)
            if j + 1 < n:
                edges.append((idx(i, j), idx(i, j + 1))); scale.append(1.0)
            # shear
            if i + 1 < n and j + 1 < n:
                edges.append((idx(i, j), idx(i + 1, j + 1))); scale.append(1.0)
                edges.append((idx(i + 1, j), idx(i, j + 1))); scale.append(1.0)
            # bending
            if i + 2 < n:
                edges.append((idx(i, j), idx(i + 2, j))); scale.append(0.3)
            if j + 2 < n:
                edges.append((idx(i, j), idx(i, j + 2))); scale.append(0.3)
    return np.array(edges, dtype=int), np.array(scale, dtype=float)


class Cloth:
    def __init__(self, n=21, size=2.0, height=2.0, mass=1.0, stiffness=8000.0,
                 pinned=None):
        self.pos, self.n = build_grid(n, size, height)
        self.N = len(self.pos)
        self.vel = np.zeros_like(self.pos)
        self.edges, self.escale = build_springs(n)
        self.rest = np.linalg.norm(self.pos[self.edges[:, 0]] - self.pos[self.edges[:, 1]], axis=1)
        self.k = stiffness * self.escale                    # per-edge stiffness
        self.m = mass / self.N                              # per-particle mass
        self.g = np.array([0.0, -9.8, 0.0])
        if pinned is None:
            pinned = [0, (n - 1) * n]  # two adjacent-row corners at one edge
        self.pinned = np.array(pinned, dtype=int)
        self.pin_mask = np.ones(self.N, dtype=bool)
        self.pin_mask[self.pinned] = False                  # False => fixed
        self.pin_pos = self.pos[self.pinned].copy()
        # number of springs incident on each node (for Jacobi PBD averaging)
        cc = np.zeros(self.N)
        np.add.at(cc, self.edges[:, 0], 1.0)
        np.add.at(cc, self.edges[:, 1], 1.0)
        self.constraint_count = np.maximum(cc, 1.0)

    # ---- spring force (vectorised) ----
    def spring_force(self, pos):
        d = pos[self.edges[:, 0]] - pos[self.edges[:, 1]]
        L = np.linalg.norm(d, axis=1)
        L = np.maximum(L, 1e-9)
        dirv = d / L[:, None]
        mag = -self.k * (L - self.rest)                     # magnitude on i (Hooke)
        fe = mag[:, None] * dirv
        f = np.zeros_like(pos)
        np.add.at(f, self.edges[:, 0], fe)
        np.add.at(f, self.edges[:, 1], -fe)
        return f

    def elastic_energy(self, pos):
        d = pos[self.edges[:, 0]] - pos[self.edges[:, 1]]
        L = np.linalg.norm(d, axis=1)
        return float(np.sum(0.5 * self.k * (L - self.rest) ** 2))

    def energy(self, floor_y=-1e9):
        ke = 0.5 * self.m * np.sum(self.vel ** 2)
        pe_g = self.m * (-self.g[1]) * np.sum(self.pos[:, 1] - floor_y)
        pe_e = self.elastic_energy(self.pos)
        return ke + pe_g + pe_e, ke, pe_g, pe_e

    def _apply_pins(self):
        self.pos[self.pinned] = self.pin_pos
        self.vel[self.pinned] = 0.0

    def mean_spring_strain(self):
        d = self.pos[self.edges[:, 0]] - self.pos[self.edges[:, 1]]
        L = np.linalg.norm(d, axis=1)
        return float(np.mean(np.abs(L - self.rest) / self.rest))

    # ---- integrators ----
    def explicit_step(self, dt, damping=0.999):
        f = self.spring_force(self.pos) + self.m * self.g
        acc = f / self.m
        self.vel = damping * self.vel + dt * acc
        self.pos = self.pos + dt * self.vel * self.pin_mask[:, None]
        self._apply_pins()

    def implicit_step(self, dt, iterations=48, damping=0.99, cheb_rho=0.99,
                      collide=None):
        """Implicit Euler via Jacobi iteration + Chebyshev acceleration (GAMES103)."""
        m_dt2 = self.m / dt ** 2
        # inertial prediction with gravity
        x_tilde = self.pos + dt * self.vel + dt ** 2 * self.g
        x_tilde[~self.pin_mask] = self.pin_pos                # pinned stay put
        x = self.pos + dt * self.vel                          # initial guess
        x[~self.pin_mask] = self.pin_pos
        # diagonal Hessian approximation:  m/dt^2 + sum_j k_ij
        diag = np.full(self.N, m_dt2)
        np.add.at(diag, self.edges[:, 0], self.k)
        np.add.at(diag, self.edges[:, 1], self.k)
        x_prev = x.copy()
        omega = 1.0
        for it in range(iterations):
            # gradient of E:  m/dt^2 (x - x_tilde) - spring_force(x)
            grad = m_dt2 * (x - x_tilde) - self.spring_force(x)
            dx = -grad / diag[:, None]
            x_new = x + dx
            # Chebyshev semi-iterative acceleration
            if it < 10:
                omega = 1.0
            elif it == 10:
                omega = 2.0 / (2.0 - cheb_rho ** 2)
            else:
                omega = 4.0 / (4.0 - cheb_rho ** 2 * omega)
            x_new = omega * (x_new - x_prev) + x_prev
            x_prev = x.copy()
            x = x_new
            x[~self.pin_mask] = self.pin_pos
        self.vel = damping * (x - self.pos) / dt
        self.pos = x
        self._apply_pins()
        if collide is not None:
            collide(self, dt)

    def pbd_step(self, dt, iterations=25, damping=0.99, collide=None):
        """Position-Based Dynamics: predict, then Gauss-Seidel distance projection."""
        self.vel = damping * self.vel
        self.vel[self.pin_mask] += dt * self.g               # gravity on free nodes
        pred = self.pos + dt * self.vel
        pred[~self.pin_mask] = self.pin_pos
        # inverse mass (0 for pinned)
        w = np.where(self.pin_mask, 1.0 / self.m, 0.0)
        e0, e1 = self.edges[:, 0], self.edges[:, 1]
        for _ in range(iterations):
            d = pred[e0] - pred[e1]
            L = np.linalg.norm(d, axis=1)
            L = np.maximum(L, 1e-9)
            n = d / L[:, None]
            C = L - self.rest
            wsum = w[e0] + w[e1]
            wsum = np.maximum(wsum, 1e-12)
            corr = (C / wsum)[:, None] * n
            # Jacobi accumulation; average by each node's incident-constraint count
            # (Macklin et al.) so a node touched by ~12 springs is not over-corrected.
            dp = np.zeros_like(pred)
            np.add.at(dp, e0, -(w[e0][:, None]) * corr)
            np.add.at(dp, e1, (w[e1][:, None]) * corr)
            pred += 1.5 * dp / self.constraint_count[:, None]
            pred[~self.pin_mask] = self.pin_pos
        self.vel = (pred - self.pos) / dt
        self.pos = pred
        self._apply_pins()
        if collide is not None:
            collide(self, dt)


def make_sphere_collider(center, radius, friction=0.5):
    c = np.asarray(center, float)

    def collide(cloth, dt):
        d = cloth.pos - c
        dist = np.linalg.norm(d, axis=1)
        inside = dist < radius
        if np.any(inside):
            n = d[inside] / np.maximum(dist[inside], 1e-9)[:, None]
            cloth.pos[inside] = c + n * radius
            # remove inward velocity + apply friction to tangential
            vn = np.sum(cloth.vel[inside] * n, axis=1)[:, None] * n
            vt = cloth.vel[inside] - vn
            cloth.vel[inside] = friction * vt
    return collide
