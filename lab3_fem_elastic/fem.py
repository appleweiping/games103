"""GAMES103 Lab 3 - FEM / FVM Elastic Body Simulation.

Tetrahedral finite-element solid with two hyperelastic constitutive models:
  * St. Venant-Kirchhoff (StVK):  Psi = mu E:E + (lambda/2) tr(E)^2,  E = 1/2 (F^T F - I)
  * (stable) neo-Hookean:         Psi = mu/2 (I1-3) - mu ln J + lambda/2 (ln J)^2

Per element we form the rest-shape matrix Dm, its inverse Bm and rest volume W,
the deformation gradient F = Ds Bm, the first Piola-Kirchhoff stress P(F), and the
nodal elastic forces H = -W P Bm^T.  Time integration is symplectic (semi-implicit)
Euler with velocity damping + a little intra-element velocity smoothing for
stability, plus floor collision by position projection.
"""
from __future__ import annotations

import numpy as np


class FEMSolid:
    def __init__(self, nodes, tets, model="neohookean", density=1000.0,
                 youngs=3.0e4, poisson=0.33):
        self.X = np.asarray(nodes, float)          # rest positions
        self.x = self.X.copy()                     # current positions
        self.tets = np.asarray(tets, int)
        self.N = len(self.X)
        self.T = len(self.tets)
        self.v = np.zeros_like(self.X)
        self.model = model
        self.g = np.array([0.0, -9.8, 0.0])
        # Lame parameters
        self.mu = youngs / (2.0 * (1.0 + poisson))
        self.lam = youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
        # rest-shape matrices Dm (T,3,3), Bm = inv(Dm), rest volume W
        i0, i1, i2, i3 = self.tets[:, 0], self.tets[:, 1], self.tets[:, 2], self.tets[:, 3]
        Dm = np.stack([self.X[i1] - self.X[i0],
                       self.X[i2] - self.X[i0],
                       self.X[i3] - self.X[i0]], axis=2)     # columns are edge vectors
        self.Bm = np.linalg.inv(Dm)
        self.W = np.abs(np.linalg.det(Dm)) / 6.0
        # lumped nodal mass
        self.mass = np.zeros(self.N)
        node_m = density * self.W / 4.0
        for k in range(4):
            np.add.at(self.mass, self.tets[:, k], node_m)
        self.mass = np.maximum(self.mass, 1e-8)
        self.total_rest_volume = float(self.W.sum())

    # ---- deformation gradient ----
    def _F(self, x):
        i0, i1, i2, i3 = self.tets[:, 0], self.tets[:, 1], self.tets[:, 2], self.tets[:, 3]
        Ds = np.stack([x[i1] - x[i0], x[i2] - x[i0], x[i3] - x[i0]], axis=2)
        return Ds @ self.Bm

    def current_volume(self, x=None):
        if x is None:
            x = self.x
        i0, i1, i2, i3 = self.tets[:, 0], self.tets[:, 1], self.tets[:, 2], self.tets[:, 3]
        Ds = np.stack([x[i1] - x[i0], x[i2] - x[i0], x[i3] - x[i0]], axis=2)
        return float(np.abs(np.linalg.det(Ds)).sum() / 6.0)

    # ---- stress ----
    def _PK1(self, F):
        I = np.eye(3)[None]
        if self.model == "stvk":
            E = 0.5 * (np.transpose(F, (0, 2, 1)) @ F - I)
            trE = np.trace(E, axis1=1, axis2=2)
            P = F @ (2 * self.mu * E + self.lam * trE[:, None, None] * I)
            psi = self.mu * np.sum(E * E, axis=(1, 2)) + 0.5 * self.lam * trE ** 2
        elif self.model == "neohookean":
            J = np.linalg.det(F)
            J = np.where(np.abs(J) < 1e-6, 1e-6, J)
            FinvT = np.transpose(np.linalg.inv(F), (0, 2, 1))
            lnJ = np.log(np.abs(J))
            P = self.mu * (F - FinvT) + self.lam * lnJ[:, None, None] * FinvT
            I1 = np.sum(F * F, axis=(1, 2))
            psi = 0.5 * self.mu * (I1 - 3.0) - self.mu * lnJ + 0.5 * self.lam * lnJ ** 2
        else:
            raise ValueError(self.model)
        return P, psi

    def elastic_forces(self, x):
        F = self._F(x)
        P, _ = self._PK1(F)
        H = -self.W[:, None, None] * (P @ np.transpose(self.Bm, (0, 2, 1)))  # (T,3,3)
        f = np.zeros_like(x)
        f1, f2, f3 = H[:, :, 0], H[:, :, 1], H[:, :, 2]
        np.add.at(f, self.tets[:, 1], f1)
        np.add.at(f, self.tets[:, 2], f2)
        np.add.at(f, self.tets[:, 3], f3)
        np.add.at(f, self.tets[:, 0], -(f1 + f2 + f3))
        return f

    def elastic_energy(self, x=None):
        if x is None:
            x = self.x
        F = self._F(x)
        _, psi = self._PK1(F)
        return float(np.sum(self.W * psi))

    def energy(self, floor_y=0.0):
        ke = 0.5 * float(np.sum(self.mass * np.sum(self.v ** 2, axis=1)))
        pe_g = float(np.sum(self.mass * (-self.g[1]) * (self.x[:, 1] - floor_y)))
        pe_e = self.elastic_energy()
        return ke + pe_g + pe_e, ke, pe_g, pe_e

    # ---- integration ----
    def step(self, dt, floor_y=0.0, damping=0.995, restitution=0.0, friction=0.4,
             vel_smooth=0.0):
        f = self.elastic_forces(self.x) + self.mass[:, None] * self.g
        acc = f / self.mass[:, None]
        self.v = damping * self.v + dt * acc
        if vel_smooth > 0:
            self.v = (1 - vel_smooth) * self.v + vel_smooth * self._smooth_velocity()
        self.x = self.x + dt * self.v
        # floor collision by projection
        below = self.x[:, 1] < floor_y
        if np.any(below):
            self.x[below, 1] = floor_y
            vy = self.v[below, 1]
            self.v[below, 1] = np.where(vy < 0, -restitution * vy, vy)
            self.v[below, 0] *= (1 - friction)
            self.v[below, 2] *= (1 - friction)

    def _smooth_velocity(self):
        """Average velocity with tet neighbours (Laplacian smoothing) for stability."""
        vs = np.zeros_like(self.v)
        cnt = np.zeros(self.N)
        for k in range(4):
            idx = self.tets[:, k]
            tet_mean = self.v[self.tets].mean(axis=1)
            np.add.at(vs, idx, tet_mean)
            np.add.at(cnt, idx, 1.0)
        cnt = np.maximum(cnt, 1.0)
        return vs / cnt[:, None]

    def surface_points(self):
        return self.x
