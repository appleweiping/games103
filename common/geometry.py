"""Procedural mesh generators (no external asset files needed / committed).

Everything is generated at run time so the repo ships no copyrighted meshes.
"""
from __future__ import annotations

import numpy as np


def uv_sphere(radius=1.0, center=(0, 0, 0), n_stack=12, n_slice=18, scale=(1, 1, 1)):
    """A UV sphere / ellipsoid mesh -> (verts (N,3), faces (M,3))."""
    cx, cy, cz = center
    sx, sy, sz = scale
    verts = []
    for i in range(n_stack + 1):
        theta = np.pi * i / n_stack           # 0..pi (pole to pole)
        for j in range(n_slice):
            phi = 2 * np.pi * j / n_slice
            x = radius * sx * np.sin(theta) * np.cos(phi)
            y = radius * sy * np.cos(theta)
            z = radius * sz * np.sin(theta) * np.sin(phi)
            verts.append((cx + x, cy + y, cz + z))
    verts = np.array(verts, dtype=float)

    faces = []
    for i in range(n_stack):
        for j in range(n_slice):
            a = i * n_slice + j
            b = i * n_slice + (j + 1) % n_slice
            c = (i + 1) * n_slice + j
            d = (i + 1) * n_slice + (j + 1) % n_slice
            faces.append((a, b, d))
            faces.append((a, d, c))
    return verts, np.array(faces, dtype=int)


def merge_meshes(parts):
    """Merge a list of (verts, faces) into a single mesh."""
    all_v, all_f, off = [], [], 0
    for v, f in parts:
        all_v.append(v)
        all_f.append(f + off)
        off += len(v)
    return np.vstack(all_v), np.vstack(all_f)


def make_bunny():
    """A cartoon 'Angry Bunny' assembled from ellipsoids (body, head, 2 ears, tail).

    Returns (verts, faces). Physics treats the vertex cloud as one rigid body;
    rendering uses the triangle faces. Recognisably a bunny, fully procedural.
    """
    parts = [
        uv_sphere(0.30, center=(0.00, 0.00, 0.00), scale=(1.0, 0.85, 0.9)),   # body
        uv_sphere(0.18, center=(0.26, 0.20, 0.00), scale=(1.0, 1.0, 0.95)),   # head
        uv_sphere(0.11, center=(0.34, 0.42, 0.07), scale=(0.45, 1.5, 0.5)),   # ear L
        uv_sphere(0.11, center=(0.34, 0.42, -0.07), scale=(0.45, 1.5, 0.5)),  # ear R
        uv_sphere(0.09, center=(-0.30, 0.05, 0.00), scale=(1.0, 1.0, 1.0)),   # tail
    ]
    v, f = merge_meshes(parts)
    v = v - v.mean(axis=0)  # center at origin so rotation is about centroid
    return v, f


def cube_mesh(size=0.5, center=(0, 0, 0)):
    """Axis-aligned cube -> (8 verts, 12 triangle faces)."""
    s = size / 2.0
    cx, cy, cz = center
    v = np.array([
        [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
        [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s],
    ], dtype=float) + np.array([cx, cy, cz])
    f = np.array([
        [0, 1, 2], [0, 2, 3],  # bottom
        [4, 6, 5], [4, 7, 6],  # top
        [0, 4, 5], [0, 5, 1],
        [1, 5, 6], [1, 6, 2],
        [2, 6, 7], [2, 7, 3],
        [3, 7, 4], [3, 4, 0],
    ], dtype=int)
    return v, f


def tet_lattice(nx, ny, nz, cell=0.25, center=(0, 0, 0)):
    """Tetrahedralised solid box on an (nx,ny,nz) node lattice.

    Each cubic cell is split into 5 tetrahedra with alternating orientation so
    faces match across cells. Returns:
        nodes  (Nn,3) rest positions
        tets   (Nt,4) node indices
        surf   (Ns,3) surface triangle faces (for rendering)
    """
    cx, cy, cz = center
    nodes = np.zeros((nx * ny * nz, 3))

    def nid(i, j, k):
        return (i * ny + j) * nz + k

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                nodes[nid(i, j, k)] = [i * cell, j * cell, k * cell]
    nodes -= nodes.mean(axis=0)
    nodes += np.array([cx, cy, cz])

    tets = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                # 8 corners of the cell
                c = [nid(i, j, k), nid(i + 1, j, k), nid(i + 1, j + 1, k), nid(i, j + 1, k),
                     nid(i, j, k + 1), nid(i + 1, j, k + 1), nid(i + 1, j + 1, k + 1), nid(i, j + 1, k + 1)]
                if (i + j + k) % 2 == 0:
                    cells = [(0, 1, 3, 4), (1, 2, 3, 6), (1, 4, 5, 6), (3, 4, 6, 7), (1, 3, 4, 6)]
                else:
                    cells = [(0, 1, 2, 5), (0, 2, 3, 7), (0, 4, 5, 7), (2, 5, 6, 7), (0, 2, 5, 7)]
                for a, b, cc, d in cells:
                    tets.append((c[a], c[b], c[cc], c[d]))
    tets = np.array(tets, dtype=int)

    # surface faces = triangles that belong to exactly one tet
    from collections import defaultdict
    face_count = defaultdict(int)
    face_repr = {}
    tet_face = [(0, 1, 2), (0, 2, 3), (0, 3, 1), (1, 3, 2)]
    for t in tets:
        for fa in tet_face:
            tri = (t[fa[0]], t[fa[1]], t[fa[2]])
            key = tuple(sorted(tri))
            face_count[key] += 1
            face_repr[key] = tri
    surf = [face_repr[k] for k, cnt in face_count.items() if cnt == 1]
    return nodes, tets, np.array(surf, dtype=int)
