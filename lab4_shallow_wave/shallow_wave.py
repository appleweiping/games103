"""GAMES103 Lab 4 - Shallow Wave (Ripples) Simulation.

Height-field water on an (n x n) grid solved with the leapfrog 2D wave equation
(the GAMES103 "shallow wave" update):

    h_new = h + damping * (h - h_old) + (c*dt/dx)^2 * Laplacian(h)

Reflecting (Neumann, zero-flux) boundaries make waves bounce off the tank walls
and conserve the total water volume exactly. Two-way coupling with a floating
rigid block: buoyancy lifts the block from the submerged depth while the block's
vertical motion injects a volume-conserving disturbance into the surface.
"""
from __future__ import annotations

import numpy as np


def laplacian(h):
    """5-point Laplacian with zero-gradient (Neumann) boundaries -> sum == 0."""
    hp = np.pad(h, 1, mode="edge")
    return (hp[:-2, 1:-1] + hp[2:, 1:-1] + hp[1:-1, :-2] + hp[1:-1, 2:] - 4.0 * h)


class ShallowWave:
    def __init__(self, n=100, size=2.0, c=1.0, dt=None, damping=1.0, rest_height=0.0):
        self.n = n
        self.size = size
        self.dx = size / (n - 1)
        self.c = c
        # CFL for 2D wave eq: c*dt/dx <= 1/sqrt(2); pick dt safely below it if not given
        self.dt = dt if dt is not None else 0.6 * self.dx / c / np.sqrt(2)
        self.alpha = (c * self.dt / self.dx) ** 2
        self.damping = damping
        self.rest = rest_height
        self.h = np.full((n, n), rest_height, dtype=float)
        self.h_old = self.h.copy()

    @property
    def cfl(self):
        return self.c * self.dt / self.dx

    def add_drop(self, cx, cz, radius, depth):
        """Add a Gaussian bump/dip (a droplet impact) centred at world (cx,cz)."""
        gx = np.linspace(-self.size / 2, self.size / 2, self.n)
        X, Z = np.meshgrid(gx, gx, indexing="ij")
        self.h += depth * np.exp(-((X - cx) ** 2 + (Z - cz) ** 2) / (2 * radius ** 2))
        self.h_old = self.h.copy()          # start from rest (zero velocity)

    def step(self):
        lap = laplacian(self.h)
        h_new = self.h + self.damping * (self.h - self.h_old) + self.alpha * lap
        self.h_old = self.h
        self.h = h_new

    def total_volume(self):
        return float(self.h.sum()) * self.dx ** 2

    def wave_energy(self):
        # ~ potential (gradient) + kinetic (time-derivative) surrogate
        vel = (self.h - self.h_old)
        gx = np.diff(self.h, axis=0); gz = np.diff(self.h, axis=1)
        return float(0.5 * np.sum(vel ** 2) + 0.5 * self.alpha * (np.sum(gx ** 2) + np.sum(gz ** 2)))


class FloatingBlock:
    """A rigid block bobbing on the shallow-wave surface (two-way coupled)."""

    def __init__(self, wave: ShallowWave, cx=0.0, cz=0.0, half=0.3, mass=0.25,
                 y0=0.6, density_water=1.0, g=9.8):
        self.w = wave
        self.cx, self.cz, self.half = cx, cz, half
        self.mass = mass
        self.y = y0                 # block bottom height
        self.vy = 0.0
        self.g = g
        self.rho = density_water
        gx = np.linspace(-wave.size / 2, wave.size / 2, wave.n)
        X, Z = np.meshgrid(gx, gx, indexing="ij")
        self.mask = (np.abs(X - cx) <= half) & (np.abs(Z - cz) <= half)
        self.ring = ring_mask(self.mask)
        self.area_cells = int(self.mask.sum())
        self.ring_cells = int(self.ring.sum())
        self.cell_area = wave.dx ** 2
        self.footprint_area = self.area_cells * self.cell_area
        self.prev_disp = 0.0

    def step(self, dt, coupling=0.25):
        """Two-way coupling (stable, volume-conserving).

        (1) Buoyancy: the block bobs on the water, using its bottom depth below
            the local free surface (Archimedes). This is a stable 1-D ODE.
        (2) Ripples: the block's vertical motion injects a small velocity source
            into the surface under its footprint (capped), and the equal volume
            is drawn from the ring so total water volume stays exactly constant.
        """
        w = self.w
        # local free-surface level the block floats on (mean over footprint+ring)
        ambient = float(w.h[self.ring].mean()) if self.ring_cells else 0.0
        submersion = max(ambient - self.y, 0.0)
        Fb = self.rho * self.g * self.footprint_area * submersion
        Fg = -self.mass * self.g
        drag = -1.2 * self.vy
        acc = (Fb + Fg + drag) / self.mass
        self.vy += acc * dt
        self.y += self.vy * dt
        # ripple source proportional to (capped) downward speed; conserve volume by
        # taking the same total from the ring. Small coefficient -> no feedback blow-up.
        if self.area_cells and self.ring_cells:
            src = np.clip(-self.vy, -1.0, 1.0) * coupling * dt          # height per cell
            w.h[self.mask] -= src
            w.h[self.ring] += src * self.area_cells / self.ring_cells   # conserve volume


def ring_mask(mask):
    """One-cell dilation ring just outside a boolean mask."""
    m = mask
    d = np.zeros_like(m)
    d[1:, :] |= m[:-1, :]; d[:-1, :] |= m[1:, :]
    d[:, 1:] |= m[:, :-1]; d[:, :-1] |= m[:, 1:]
    return d & (~m)
