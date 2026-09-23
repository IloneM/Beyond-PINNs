"""Minimal, self-contained deterministic FEM utilities, extracted verbatim
(algorithm-for-algorithm) from femennstein-petrov-galerkin-experiments-fixed.py,
for reconstructing the "standalone near-budget FEM" reference solution field
used in the appendix solution/gradient/cross-section figures.

This module performs ONLY deterministic linear-algebra FEM assembly/solve --
no neural-network training of any kind.
"""
import time as _time
import numpy as np
import jax
import jax.numpy as jnp
from scipy.sparse import coo_matrix as _coo_matrix
from scipy.sparse.linalg import spsolve as _spsolve_sparse

jax.config.update("jax_enable_x64", True)

# ============================================================
# 1D high-order Lagrange FEM (Section 1)
# ============================================================
def _lagrange_reference_1d(degree, n_quad=20):
    p = int(degree)
    nodes = np.linspace(0.0, 1.0, p + 1)
    powers = np.arange(p + 1)
    V = nodes[:, None] ** powers[None, :]
    coeff = np.linalg.inv(V)
    q0, w0 = np.polynomial.legendre.leggauss(max(int(n_quad), p + 3))
    q = 0.5 * (q0 + 1.0); w = 0.5 * w0
    mon = q[:, None] ** powers[None, :]
    dmon = np.zeros_like(mon)
    if p:
        dmon[:, 1:] = powers[1:][None, :] * q[:, None] ** (powers[1:][None, :] - 1)
    return nodes, mon @ coeff, dmon @ coeff, q, w, coeff


def solve_lagrange_1d(n_elements, degree, A_func, f_func=lambda x: 1.0, n_quad=20):
    n_elements, p = int(n_elements), int(degree)
    _, Nq, dNq, qref, wref, coeff_ref = _lagrange_reference_1d(p, n_quad)
    n_nodes = n_elements * p + 1
    nodes = np.linspace(0.0, 1.0, n_nodes)
    rows, cols, data = [], [], []
    F = np.zeros(n_nodes)
    t_asm = _time.perf_counter()
    for e in range(n_elements):
        a, b = e / n_elements, (e + 1) / n_elements
        h = b - a
        xq = a + h * qref
        Aq = np.asarray(jax.vmap(A_func)(jnp.asarray(xq)), dtype=float)
        fq = (np.asarray(jax.vmap(f_func)(jnp.asarray(xq)), dtype=float)
              if callable(f_func) else np.full_like(xq, float(f_func)))
        G = dNq / h
        Ke = (G.T * (wref * h * Aq)) @ G
        Fe = Nq.T @ (wref * h * fq)
        ids = e * p + np.arange(p + 1)
        F[ids] += Fe
        ii, jj = np.meshgrid(ids, ids, indexing='ij')
        rows.extend(ii.ravel()); cols.extend(jj.ravel()); data.extend(Ke.ravel())
    K = _coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
    assembly_time = _time.perf_counter() - t_asm
    free = np.arange(1, n_nodes - 1)
    coeff = np.zeros(n_nodes)
    t_solve = _time.perf_counter()
    coeff[free] = _spsolve_sparse(K[free][:, free], F[free])
    solve_time = _time.perf_counter() - t_solve
    return dict(degree=p, n_elements=n_elements, nodes=nodes, coeff=coeff,
                free=free, n_dofs=len(free), ref_coeff=coeff_ref,
                assembly_time=assembly_time, solve_time=solve_time)


def eval_lagrange_1d(bundle, points, derivative=False):
    x = np.asarray(points, dtype=float)
    p, ne = int(bundle['degree']), int(bundle['n_elements'])
    e = np.clip((x * ne).astype(int), 0, ne - 1)
    xi = x * ne - e
    powers = np.arange(p + 1)
    mon = xi[:, None] ** powers[None, :]
    if derivative:
        dmon = np.zeros_like(mon)
        if p:
            dmon[:, 1:] = powers[1:][None, :] * xi[:, None] ** (powers[1:][None, :] - 1)
        basis = (dmon @ bundle['ref_coeff']) * ne
    else:
        basis = mon @ bundle['ref_coeff']
    ids = e[:, None] * p + np.arange(p + 1)[None, :]
    return np.sum(basis * bundle['coeff'][ids], axis=1)


# ============================================================
# 2D conforming triangular P1-P4 FEM (Sections 2-4)
# ============================================================
def make_unit_square_mesh(nx, ny=None):
    ny = nx if ny is None else ny
    xs = np.linspace(0.0, 1.0, nx + 1)
    ys = np.linspace(0.0, 1.0, ny + 1)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    nodes = np.stack([X.ravel(), Y.ravel()], 1); Npy = ny + 1
    ix, iy = np.mgrid[0:nx, 0:ny]; ix, iy = ix.ravel(), iy.ravel()
    n00 = ix * Npy + iy; n10 = (ix + 1) * Npy + iy
    n11 = (ix + 1) * Npy + (iy + 1); n01 = ix * Npy + (iy + 1)
    even = ((ix + iy) % 2 == 0)
    t1 = np.where(even[:, None], np.stack([n00, n10, n11], 1), np.stack([n00, n10, n01], 1))
    t2 = np.where(even[:, None], np.stack([n00, n11, n01], 1), np.stack([n10, n11, n01], 1))
    tris = np.concatenate([t1, t2], 0)
    bmask = (np.isclose(nodes[:, 0], 0) | np.isclose(nodes[:, 0], 1)
             | np.isclose(nodes[:, 1], 0) | np.isclose(nodes[:, 1], 1))
    return nodes, tris.astype(np.int32), bmask


def make_lshape_corner_mesh(n_radial=16, n_angular=48, beta=2.0, R_out=1.0,
                             grading_mode="power", q=0.7):
    seg = [0.0, np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 1.5 * np.pi]
    lens = np.diff(seg)
    counts = np.maximum(1, np.round(n_angular * lens / lens.sum()).astype(int))
    thetas = np.concatenate([np.linspace(seg[k], seg[k + 1], counts[k] + 1)[:-1]
                             for k in range(4)] + [np.array([1.5 * np.pi])])
    Nth, M = thetas.size, n_radial
    j = np.arange(0, M + 1)
    if grading_mode == "power":
        t = (j / M) ** beta
    elif grading_mode == "geometric":
        q = float(q)
        t = (q ** (M - j) - q ** M) / (1.0 - q ** M)
    else:
        raise ValueError(grading_mode)
    c, s = np.cos(thetas), np.sin(thetas)
    rho = R_out / np.maximum(np.abs(c), np.abs(s))
    ring = np.stack([t[1:, None] * rho[None, :] * c[None, :],
                     t[1:, None] * rho[None, :] * s[None, :]], axis=-1)
    nodes = np.concatenate([np.array([[0.0, 0.0]]), ring.reshape(-1, 2)], axis=0)
    def idx(i, j): return 1 + i * Nth + j
    tris = [[0, idx(0, j), idx(0, j + 1)] for j in range(Nth - 1)]
    for i in range(M - 1):
        for j in range(Nth - 1):
            a, b, cc, d = idx(i, j), idx(i, j + 1), idx(i + 1, j + 1), idx(i + 1, j)
            tris += [[a, b, cc], [a, cc, d]]
    elements = np.array(tris)
    bmask = np.zeros(nodes.shape[0], dtype=bool); bmask[0] = True
    for i in range(M):
        for j in range(Nth):
            if i == M - 1 or j == 0 or j == Nth - 1:
                bmask[idx(i, j)] = True
    return nodes, elements, bmask


def locate_np(pts, nd, el):
    v = np.array(nd)[np.array(el)]; v0 = v[:, 0]
    e1 = v[:, 1] - v0; e2 = v[:, 2] - v0
    det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    d = np.array(pts)[:, None, :] - v0[None]
    l1 = (d[:, :, 0] * e2[None, :, 1] - d[:, :, 1] * e2[None, :, 0]) / det
    l2 = (-d[:, :, 0] * e1[None, :, 1] + d[:, :, 1] * e1[None, :, 0]) / det
    l0 = 1 - l1 - l2
    best = np.argmax(np.minimum(np.minimum(l0, l1), l2), axis=1)
    N = pts.shape[0]
    return best, np.stack([l0[np.arange(N), best], l1[np.arange(N), best],
                           l2[np.arange(N), best]], 1)


def locate_np_chunked(pts, nd, el, chunk_size=8192):
    pts = np.asarray(pts)
    ids, bary = [], []
    for start in range(0, len(pts), int(chunk_size)):
        stop = min(start + int(chunk_size), len(pts))
        i, b = locate_np(pts[start:stop], nd, el)
        ids.append(i); bary.append(b)
    if not ids:
        return np.empty((0,), dtype=np.int32), np.empty((0, 3), dtype=float)
    return np.concatenate(ids), np.concatenate(bary)


def _triangle_duffy_rule(order):
    z, w = np.polynomial.legendre.leggauss(max(4, int(order)))
    u = 0.5 * (z + 1.0); wu = 0.5 * w
    U, V = np.meshgrid(u, u, indexing='ij')
    WU, WV = np.meshgrid(wu, wu, indexing='ij')
    xi = U.ravel(); eta = ((1.0 - U) * V).ravel()
    weights = (WU * WV * (1.0 - U)).ravel()
    return np.column_stack([xi, eta]), weights


def _pk_reference(degree, quadrature_order=None):
    p = int(degree)
    bary = []
    for i in range(p, -1, -1):
        for j in range(p - i, -1, -1):
            k = p - i - j
            bary.append((i / p, j / p, k / p) if p else (1.0, 0.0, 0.0))
    bary = np.asarray(bary, dtype=float)
    ref_nodes = np.column_stack([bary[:, 1], bary[:, 2]])
    exps = [(a, b) for a in range(p + 1) for b in range(p + 1 - a)]
    def mon(points):
        x, y = points[:, 0], points[:, 1]
        return np.column_stack([x**a * y**b for a, b in exps])
    V = mon(ref_nodes)
    coeff = np.linalg.inv(V)
    q, w = _triangle_duffy_rule(max(2 * p + 3, quadrature_order or 0))
    M = mon(q)
    dMx = np.column_stack([(a * q[:, 0]**(a-1) * q[:, 1]**b) if a else np.zeros(len(q)) for a, b in exps])
    dMy = np.column_stack([(b * q[:, 0]**a * q[:, 1]**(b-1)) if b else np.zeros(len(q)) for a, b in exps])
    Nq = M @ coeff
    Gq = np.stack([dMx @ coeff, dMy @ coeff], axis=-1)
    return dict(degree=p, bary=bary, ref_nodes=ref_nodes, exps=exps, coeff=coeff, q=q, w=w, Nq=Nq, Gq=Gq)


def _pk_values(ref, xi_eta):
    pts = np.atleast_2d(np.asarray(xi_eta, dtype=float))
    M = np.column_stack([pts[:, 0]**a * pts[:, 1]**b for a, b in ref['exps']])
    dMx = np.column_stack([(a * pts[:, 0]**(a-1) * pts[:, 1]**b) if a else np.zeros(len(pts)) for a, b in ref['exps']])
    dMy = np.column_stack([(b * pts[:, 0]**a * pts[:, 1]**(b-1)) if b else np.zeros(len(pts)) for a, b in ref['exps']])
    return M @ ref['coeff'], np.stack([dMx @ ref['coeff'], dMy @ ref['coeff']], axis=-1)


def enrich_pk_mesh(nodes, elements, boundary_mask, degree):
    nodes = np.asarray(nodes, dtype=float); elements = np.asarray(elements, dtype=np.int32)
    p = int(degree); ref = _pk_reference(p)
    edge_count = {}
    for tri in elements:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            e = tuple(sorted((int(a), int(b))))
            edge_count[e] = edge_count.get(e, 0) + 1
    global_nodes, global_boundary, key_to_id, conn = [], [], {}, []
    for tri in elements:
        verts = nodes[tri]
        local_ids = []
        for lam in ref['bary']:
            x = lam @ verts
            key = tuple(np.round(x, 13))
            is_bdry = False
            for opp, edge_loc in ((0, (1, 2)), (1, (0, 2)), (2, (0, 1))):
                if abs(lam[opp]) < 1e-12:
                    edge = tuple(sorted((int(tri[edge_loc[0]]), int(tri[edge_loc[1]]))))
                    is_bdry = is_bdry or edge_count.get(edge, 0) == 1
            if key not in key_to_id:
                key_to_id[key] = len(global_nodes)
                global_nodes.append(x.tolist()); global_boundary.append(bool(is_bdry))
            else:
                gid = key_to_id[key]
                global_boundary[gid] = global_boundary[gid] or bool(is_bdry)
            local_ids.append(key_to_id[key])
        conn.append(local_ids)
    return np.asarray(global_nodes), np.asarray(conn, dtype=np.int32), np.asarray(global_boundary), ref


def solve_pk_dirichlet(nodes, elements, boundary_mask, f_func, degree, *,
                        line_source=None, quadrature_order=None, A_func=None):
    t0 = _time.perf_counter()
    coarse_nodes = np.asarray(nodes, dtype=float); coarse_elements = np.asarray(elements, dtype=np.int32)
    pnodes, pelems, pbmask, ref = enrich_pk_mesh(coarse_nodes, coarse_elements, boundary_mask, degree)
    ref = _pk_reference(degree, quadrature_order)
    rows, cols, data = [], [], []
    F = np.zeros(len(pnodes))
    for e, tri in enumerate(coarse_elements):
        v = coarse_nodes[tri]
        J = np.column_stack([v[1] - v[0], v[2] - v[0]])
        detJ = float(np.linalg.det(J)); absdet = abs(detJ)
        if absdet <= 1e-15:
            raise ValueError(f'degenerate triangle {e}')
        invJ = np.linalg.inv(J)
        qphys = v[0] + ref['q'] @ J.T
        fq = np.asarray(jax.vmap(f_func)(jnp.asarray(qphys)), dtype=float)
        Aq = np.ones(len(qphys)) if A_func is None else np.asarray(jax.vmap(A_func)(jnp.asarray(qphys)), dtype=float)
        G = np.einsum('qia,ab->qib', ref['Gq'], invJ)
        Ke = np.einsum('q,q,qia,qja->ij', ref['w'] * absdet, Aq, G, G)
        Fe = ref['Nq'].T @ (ref['w'] * absdet * fq)
        ids = pelems[e]
        F[ids] += Fe
        ii, jj = np.meshgrid(ids, ids, indexing='ij')
        rows.extend(ii.ravel()); cols.extend(jj.ravel()); data.extend(Ke.ravel())
    K = _coo_matrix((data, (rows, cols)), shape=(len(pnodes), len(pnodes))).tocsr()
    if line_source is not None:
        lpts, lw = map(np.asarray, line_source)
        eid, bary = locate_np_chunked(lpts, coarse_nodes, coarse_elements)
        vals, _ = _pk_values(ref, np.column_stack([bary[:, 1], bary[:, 2]]))
        for qid, e in enumerate(eid):
            F[pelems[e]] += float(lw[qid]) * vals[qid]
    assembly_time = _time.perf_counter() - t0
    free = np.where(~pbmask)[0]
    coeff = np.zeros(len(pnodes))
    ts = _time.perf_counter()
    coeff[free] = _spsolve_sparse(K[free][:, free], F[free])
    solve_time = _time.perf_counter() - ts
    return dict(degree=int(degree), nodes=pnodes, elements=pelems, boundary=pbmask,
                coeff=coeff, free=free, n_dofs=len(free), coarse_nodes=coarse_nodes,
                coarse_elements=coarse_elements, ref=ref,
                assembly_time=assembly_time, solve_time=solve_time)


def eval_pk_solution(bundle, points, return_gradient=False, chunk_size=4096):
    points = np.asarray(points, dtype=float)
    eid, bary = locate_np_chunked(points, bundle['coarse_nodes'], bundle['coarse_elements'], chunk_size)
    xi = np.column_stack([bary[:, 1], bary[:, 2]])
    vals, grads_ref = _pk_values(bundle['ref'], xi)
    coeff_local = bundle['coeff'][bundle['elements'][eid]]
    values = np.einsum('ni,ni->n', vals, coeff_local)
    if not return_gradient:
        return values
    gradients = np.empty((len(points), 2))
    for e in np.unique(eid):
        mask = eid == e
        v = bundle['coarse_nodes'][bundle['coarse_elements'][e]]
        J = np.column_stack([v[1] - v[0], v[2] - v[0]])
        Gphys = np.einsum('nia,ab->nib', grads_ref[mask], np.linalg.inv(J))
        gradients[mask] = np.einsum('ni,nia->na', coeff_local[mask], Gphys)
    return values, gradients


# ============================================================
# Section-specific exact PDE data (verbatim formulas from the notebook)
# ============================================================
eps_1d = 0.5 ** 4
TWO_PI = 2.0 * jnp.pi

def A_eps_1d(x):
    return 1.0 / (2.0 + jnp.cos(TWO_PI * x / eps_1d))

def _pfun(x):
    return jnp.where(x < 0.5, x ** 2 / 2 - 3 * x / 8, (x - 1) / 8)

def f22_single(P):
    x, y = P[0], P[1]
    return -jnp.where(x < 0.5, 1.0, 0.0) * y * (1 - y) + 2.0 * _pfun(x)

A_one = lambda x: 1.0

def f24_reg_single(P):
    x, y = P[0], P[1]
    def _tent(z): return jnp.minimum(z, 1 - z)
    return 2.0 * _tent(x) + 4.0 * jnp.pi ** 2 * jnp.sin(2 * jnp.pi * x) * jnp.sin(2 * jnp.pi * y)

def _kink_line_source(n_quad=16):
    q0, w0 = np.polynomial.legendre.leggauss(n_quad)
    ly = 0.5 * (q0 + 1.0); lyw = 0.5 * w0
    pts = np.stack([np.full_like(ly, 0.5), ly], axis=1)
    density = lyw * 2.0 * ly * (1.0 - ly)
    return pts, density

kink_line_points, kink_line_weighted_density = _kink_line_source()

def fL_single(point):
    x, y = point
    r2 = x ** 2 + y ** 2
    th = jnp.where(jnp.arctan2(y, x) < 0.0, jnp.arctan2(y, x) + 2.0 * jnp.pi, jnp.arctan2(y, x))
    r_23 = r2 ** (1.0 / 3.0); r_m13 = r2 ** (-1.0 / 6.0)
    t1 = (4.0 - 2.0 * r2) * r_23 * jnp.sin(2.0 * th / 3.0)
    t2 = (8.0 / 3.0) * r_m13 * (y * (1.0 - x ** 2) * jnp.cos(th / 3.0)
                                - x * (1.0 - y ** 2) * jnp.sin(th / 3.0))
    return t1 + t2
