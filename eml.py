"""
eml — minimal library for tree-based differentiable symbolic computation (numpy only)

Core primitive: eml(x, y) = exp(x) - log|y|
Supports: +, *, /, sin, neg, eml, const, var, and tree composition (tree-as-atom).

Phase 1 additions:
    - TreeList: vector-valued trees (bundle of k scalar trees)
    - SIGReg: numpy-based anti-collapse regularizer
    - fit_ep: encoder + predictor joint training with SIGReg

This mirrors LeWorldModel's structure (encoder -> latent -> predictor)
but for static tabular data, with no actions and no temporal dimension.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Union

EXP_CLIP = 10.0
LN_EPS = 1e-6
DIV_EPS = 1e-6


# ======================================================================
# Tree data structure
# ======================================================================

@dataclass
class Node:
    op: str
    children: List['Node'] = field(default_factory=list)
    cid: Optional[int] = None
    init_val: float = 1.0
    var_idx: int = 0

    def __repr__(self):
        return _format(self)


# ======================================================================
# Builders
# ======================================================================

def var(i: int) -> Node:
    n = Node('var'); n.var_idx = int(i); return n

def const(v: float) -> Node:
    n = Node('const'); n.init_val = float(v); return n

def neg(a: Node) -> Node: return Node('neg', [a])
def sin(a: Node) -> Node: return Node('sin', [a])
def add(a: Node, b: Node) -> Node: return Node('+', [a, b])
def mul(a: Node, b: Node) -> Node: return Node('*', [a, b])
def div(a: Node, b: Node) -> Node: return Node('/', [a, b])
def eml(a: Node, b: Node) -> Node: return Node('eml', [a, b])
def tree_ref(inner: Node) -> Node: return Node('ref', [inner])


# ======================================================================
# Traversal / introspection
# ======================================================================

def _all_nodes(n: Node):
    yield n
    for c in n.children: yield from _all_nodes(c)

def size(n: Node) -> int:
    return sum(1 for _ in _all_nodes(n))

def clone(n: Node) -> Node:
    nn = Node(n.op, [clone(c) for c in n.children], n.cid, n.init_val)
    nn.var_idx = n.var_idx
    return nn


# ======================================================================
# Constant indexing
# ======================================================================

def _assign_cids(n: Node, theta_out: List[float]) -> None:
    if n.op == 'const':
        n.cid = len(theta_out)
        theta_out.append(n.init_val)
    for c in n.children:
        _assign_cids(c, theta_out)

def _writeback(n: Node, theta: np.ndarray) -> None:
    if n.op == 'const' and n.cid is not None:
        n.init_val = float(theta[n.cid])
    for c in n.children:
        _writeback(c, theta)


# ======================================================================
# Forward evaluation
# ======================================================================

def _eval(n: Node, X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    if n.op == 'var':
        return X[:, n.var_idx]
    if n.op == 'const':
        return np.full(X.shape[0],
                       theta[n.cid] if n.cid is not None else n.init_val)
    if n.op == 'ref':
        return _eval(n.children[0], X, theta)
    if n.op == 'neg':
        return -_eval(n.children[0], X, theta)
    if n.op == 'sin':
        return np.sin(_eval(n.children[0], X, theta))
    if n.op == '+':
        return _eval(n.children[0], X, theta) + _eval(n.children[1], X, theta)
    if n.op == '*':
        return _eval(n.children[0], X, theta) * _eval(n.children[1], X, theta)
    if n.op == '/':
        a = _eval(n.children[0], X, theta)
        b = _eval(n.children[1], X, theta)
        b_safe = np.sign(b) * (np.abs(b) + DIV_EPS)
        b_safe = np.where(b_safe == 0, DIV_EPS, b_safe)
        return a / b_safe
    if n.op == 'eml':
        a = _eval(n.children[0], X, theta)
        b = _eval(n.children[1], X, theta)
        a_c = np.clip(a, -EXP_CLIP, EXP_CLIP)
        return np.exp(a_c) - np.log(np.maximum(np.abs(b), LN_EPS))
    raise ValueError(f"unknown op: {n.op}")


# ======================================================================
# Gradient wrt constants
# ======================================================================

Grad = Dict[int, np.ndarray]

def _eval_grad(n: Node, X: np.ndarray,
               theta: np.ndarray) -> Tuple[np.ndarray, Grad]:
    if n.op == 'var':
        return X[:, n.var_idx], {}
    if n.op == 'const':
        return np.full(X.shape[0], theta[n.cid]), {n.cid: np.ones(X.shape[0])}
    if n.op == 'ref':
        return _eval_grad(n.children[0], X, theta)
    if n.op == 'neg':
        v, g = _eval_grad(n.children[0], X, theta)
        return -v, {k: -vv for k, vv in g.items()}
    if n.op == 'sin':
        v, g = _eval_grad(n.children[0], X, theta)
        cv = np.cos(v)
        return np.sin(v), {k: vv * cv for k, vv in g.items()}
    if n.op == '+':
        a, ga = _eval_grad(n.children[0], X, theta)
        b, gb = _eval_grad(n.children[1], X, theta)
        g = dict(ga)
        for k, vv in gb.items(): g[k] = g.get(k, 0.0) + vv
        return a + b, g
    if n.op == '*':
        a, ga = _eval_grad(n.children[0], X, theta)
        b, gb = _eval_grad(n.children[1], X, theta)
        g = {k: vv * b for k, vv in ga.items()}
        for k, vv in gb.items(): g[k] = g.get(k, 0.0) + a * vv
        return a * b, g
    if n.op == '/':
        a, ga = _eval_grad(n.children[0], X, theta)
        b, gb = _eval_grad(n.children[1], X, theta)
        b_safe = np.sign(b) * (np.abs(b) + DIV_EPS)
        b_safe = np.where(b_safe == 0, DIV_EPS, b_safe)
        inv_b = 1.0 / b_safe
        g = {k: vv * inv_b for k, vv in ga.items()}
        neg_over = -a / (b_safe * b_safe)
        for k, vv in gb.items(): g[k] = g.get(k, 0.0) + vv * neg_over
        return a / b_safe, g
    if n.op == 'eml':
        a, ga = _eval_grad(n.children[0], X, theta)
        b, gb = _eval_grad(n.children[1], X, theta)
        a_c = np.clip(a, -EXP_CLIP, EXP_CLIP)
        ev = np.exp(a_c)
        ab = np.maximum(np.abs(b), LN_EPS)
        mask = (np.abs(a) <= EXP_CLIP).astype(float)
        g = {k: vv * ev * mask for k, vv in ga.items()}
        inv = -np.sign(b) / ab
        for k, vv in gb.items(): g[k] = g.get(k, 0.0) + vv * inv
        return ev - np.log(ab), g
    raise ValueError(f"unknown op: {n.op}")


# ======================================================================
# Gradient wrt input variables (for chaining through encoder)
# ======================================================================

def _eval_var_grads(n: Node, X: np.ndarray,
                    theta: np.ndarray) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    """Returns (value, {var_idx: d value / d X[:, var_idx]})."""
    if n.op == 'var':
        return X[:, n.var_idx], {n.var_idx: np.ones(X.shape[0])}
    if n.op == 'const':
        return np.full(X.shape[0],
                       theta[n.cid] if n.cid is not None else n.init_val), {}
    if n.op == 'ref':
        return _eval_var_grads(n.children[0], X, theta)
    if n.op == 'neg':
        v, g = _eval_var_grads(n.children[0], X, theta)
        return -v, {k: -vv for k, vv in g.items()}
    if n.op == 'sin':
        v, g = _eval_var_grads(n.children[0], X, theta)
        cv = np.cos(v)
        return np.sin(v), {k: vv * cv for k, vv in g.items()}
    if n.op == '+':
        a, ga = _eval_var_grads(n.children[0], X, theta)
        b, gb = _eval_var_grads(n.children[1], X, theta)
        g = dict(ga)
        for k, vv in gb.items(): g[k] = g.get(k, 0.0) + vv
        return a + b, g
    if n.op == '*':
        a, ga = _eval_var_grads(n.children[0], X, theta)
        b, gb = _eval_var_grads(n.children[1], X, theta)
        g = {k: vv * b for k, vv in ga.items()}
        for k, vv in gb.items(): g[k] = g.get(k, 0.0) + a * vv
        return a * b, g
    if n.op == '/':
        a, ga = _eval_var_grads(n.children[0], X, theta)
        b, gb = _eval_var_grads(n.children[1], X, theta)
        b_safe = np.sign(b) * (np.abs(b) + DIV_EPS)
        b_safe = np.where(b_safe == 0, DIV_EPS, b_safe)
        inv_b = 1.0 / b_safe
        g = {k: vv * inv_b for k, vv in ga.items()}
        neg_over = -a / (b_safe * b_safe)
        for k, vv in gb.items(): g[k] = g.get(k, 0.0) + vv * neg_over
        return a / b_safe, g
    if n.op == 'eml':
        a, ga = _eval_var_grads(n.children[0], X, theta)
        b, gb = _eval_var_grads(n.children[1], X, theta)
        a_c = np.clip(a, -EXP_CLIP, EXP_CLIP)
        ev = np.exp(a_c)
        ab = np.maximum(np.abs(b), LN_EPS)
        mask = (np.abs(a) <= EXP_CLIP).astype(float)
        g = {k: vv * ev * mask for k, vv in ga.items()}
        inv = -np.sign(b) / ab
        for k, vv in gb.items(): g[k] = g.get(k, 0.0) + vv * inv
        return ev - np.log(ab), g
    raise ValueError(f"unknown op: {n.op}")


# ======================================================================
# Single-tree fit
# ======================================================================

def fit(tree: Node, X: np.ndarray, y: np.ndarray,
        steps: int = 500, lr: float = 0.05,
        verbose: bool = False) -> Node:
    tree = clone(tree)
    theta_list: List[float] = []
    _assign_cids(tree, theta_list)
    if not theta_list:
        return tree
    theta = np.array(theta_list, dtype=np.float64)
    m = np.zeros_like(theta); v = np.zeros_like(theta)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        pred, grads = _eval_grad(tree, X, theta)
        if not np.all(np.isfinite(pred)): break
        resid = pred - y
        g = np.zeros_like(theta)
        for cid, dp in grads.items():
            gv = float(np.mean(2.0 * resid * dp))
            g[cid] = np.clip(gv, -10.0, 10.0) if np.isfinite(gv) else 0.0
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        theta -= lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
        if verbose and (t % max(1, steps // 10) == 0):
            print(f"[fit] step {t:4d}  loss {float(np.mean(resid ** 2)):.6f}")
    _writeback(tree, theta)
    return tree


def predict(tree: Node, X: np.ndarray) -> np.ndarray:
    theta_list: List[float] = []
    _assign_cids(tree, theta_list)
    theta = np.array(theta_list, dtype=np.float64)
    return _eval(tree, X, theta)


def r2_score(tree: Node, X: np.ndarray, y: np.ndarray) -> float:
    pred = predict(tree, X)
    ss_res = float(np.mean((pred - y) ** 2))
    ss_tot = float(np.var(y)) + 1e-12
    return 1.0 - ss_res / ss_tot


def trainable_count(tree: Node) -> int:
    theta: List[float] = []
    _assign_cids(clone(tree), theta)
    return len(theta)


# ======================================================================
# Pretty-print
# ======================================================================

def _format(n: Node) -> str:
    if n.op == 'var':   return f"x{n.var_idx}"
    if n.op == 'const': return f"{n.init_val:.4g}"
    if n.op == 'ref':   return f"<{_format(n.children[0])}>"
    if n.op == 'neg':   return f"(-{_format(n.children[0])})"
    if n.op == 'sin':   return f"sin({_format(n.children[0])})"
    if n.op == 'eml':   return f"eml({_format(n.children[0])},{_format(n.children[1])})"
    if n.op in ('+', '*', '/'):
        return f"({_format(n.children[0])} {n.op} {_format(n.children[1])})"
    return f"?{n.op}?"


# ======================================================================
# Phase 1: TreeList (vector-valued trees)
# ======================================================================

class TreeList:
    """Bundle of k scalar trees sharing input X. Output: (N, k)."""
    def __init__(self, trees: List[Node]):
        if not trees:
            raise ValueError("TreeList needs at least one tree")
        self.trees = [clone(t) for t in trees]

    @property
    def dim(self) -> int:
        return len(self.trees)

    def __repr__(self):
        lines = [f"TreeList(dim={self.dim}):"]
        for i, t in enumerate(self.trees):
            lines.append(f"  [{i}] {_format(t)}")
        return "\n".join(lines)


def _eval_treelist(tl: TreeList, X: np.ndarray,
                   theta: np.ndarray) -> np.ndarray:
    return np.stack([_eval(t, X, theta) for t in tl.trees], axis=-1)


def _eval_grad_treelist(tl: TreeList, X: np.ndarray,
                        theta: np.ndarray) -> Tuple[np.ndarray, List[Grad]]:
    vals, grads = [], []
    for t in tl.trees:
        v, g = _eval_grad(t, X, theta)
        vals.append(v); grads.append(g)
    return np.stack(vals, axis=-1), grads


def _assign_cids_many(objs) -> np.ndarray:
    theta: List[float] = []
    for obj in objs:
        if isinstance(obj, TreeList):
            for t in obj.trees: _assign_cids(t, theta)
        else:
            _assign_cids(obj, theta)
    return np.array(theta, dtype=np.float64)


def _writeback_many(objs, theta: np.ndarray) -> None:
    for obj in objs:
        if isinstance(obj, TreeList):
            for t in obj.trees: _writeback(t, theta)
        else:
            _writeback(obj, theta)


# ======================================================================
# Phase 1: SIGReg (anti-collapse regularizer)
# ======================================================================
#
# Simplified SIGReg: projects Z onto M random 1D directions and pushes
# each projection toward standard normal via characteristic-function
# matching. This is a lightweight proxy for the full Epps-Pulley integral.

def _sigreg_value_and_grad(Z: np.ndarray, M: int = 64,
                           rng: Optional[np.random.Generator] = None
                           ) -> Tuple[float, np.ndarray]:
    """Returns (loss, d loss / d Z)."""
    N, d = Z.shape
    if rng is None:
        rng = np.random.default_rng(0)

    U = rng.standard_normal((d, M))
    U /= (np.linalg.norm(U, axis=0, keepdims=True) + 1e-12)
    H = Z @ U   # (N, M)

    ts = np.array([0.5, 1.0, 1.5, 2.0])
    total_loss = 0.0
    grad_H = np.zeros_like(H)

    for t in ts:
        cos_tH = np.cos(t * H)
        sin_tH = np.sin(t * H)
        emp_cos = np.mean(cos_tH, axis=0)
        target = np.exp(-0.5 * t * t)
        diff = emp_cos - target
        total_loss += np.mean(diff ** 2)
        grad_H += (2.0 / M) * diff[None, :] * (-t * sin_tH) / N

    total_loss /= len(ts)
    grad_H /= len(ts)
    grad_Z = grad_H @ U.T
    return float(total_loss), grad_Z


def sigreg(Z: np.ndarray, M: int = 64, seed: int = 0) -> float:
    """Value-only SIGReg for inspection."""
    rng = np.random.default_rng(seed)
    v, _ = _sigreg_value_and_grad(Z, M=M, rng=rng)
    return v


# ======================================================================
# Phase 1: fit_ep (encoder + predictor + SIGReg)
# ======================================================================

def fit_ep(encoder: TreeList,
           predictor: Node,
           X: np.ndarray, y: np.ndarray,
           steps: int = 500, lr: float = 0.05,
           lambd: float = 0.1, sigreg_M: int = 64,
           sigreg_seed: int = 0,
           verbose: bool = False
           ) -> Tuple[TreeList, Node]:
    """Train encoder + predictor jointly.
    
    Forward:  z = encoder(X),  y_pred = predictor(z)
    Loss:     MSE(y_pred, y) + lambd * SIGReg(z)
    """
    enc = TreeList([clone(t) for t in encoder.trees])
    pred = clone(predictor)

    theta = _assign_cids_many([enc, pred])
    if len(theta) == 0:
        return enc, pred

    rng = np.random.default_rng(sigreg_seed)
    m = np.zeros_like(theta); v = np.zeros_like(theta)
    b1, b2, eps = 0.9, 0.999, 1e-8

    for t in range(1, steps + 1):
        # Encoder forward + grads wrt encoder constants
        Z, enc_grads = _eval_grad_treelist(enc, X, theta)

        # Predictor forward + grads wrt predictor constants and wrt Z
        y_pred, pred_const_grads = _eval_grad(pred, Z, theta)
        _, pred_var_grads = _eval_var_grads(pred, Z, theta)

        if not np.all(np.isfinite(y_pred)): break

        resid = y_pred - y
        N = X.shape[0]
        pred_loss = float(np.mean(resid ** 2))

        grad_theta = np.zeros_like(theta)

        # Predictor's constants
        for cid, dp in pred_const_grads.items():
            gv = float(np.mean(2.0 * resid * dp))
            grad_theta[cid] += (np.clip(gv, -10.0, 10.0)
                                if np.isfinite(gv) else 0.0)

        # dL_pred / dZ
        dZ_pred = np.zeros((N, enc.dim))
        for j, dpdZj in pred_var_grads.items():
            if j < enc.dim:
                dZ_pred[:, j] = (2.0 / N) * resid * dpdZj

        # dL_sigreg / dZ
        sig_loss, dZ_sig = _sigreg_value_and_grad(Z, M=sigreg_M, rng=rng)

        dZ_total = dZ_pred + lambd * dZ_sig

        # Backprop dZ into encoder constants
        for j in range(enc.dim):
            gj = enc_grads[j]
            gZj = dZ_total[:, j]
            for cid, dp in gj.items():
                contribution = float(np.sum(gZj * dp))
                grad_theta[cid] += (np.clip(contribution, -10.0, 10.0)
                                    if np.isfinite(contribution) else 0.0)

        m = b1 * m + (1 - b1) * grad_theta
        v = b2 * v + (1 - b2) * grad_theta * grad_theta
        theta -= lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)

        if verbose and (t % max(1, steps // 10) == 0):
            total = pred_loss + lambd * sig_loss
            print(f"[fit_ep] step {t:4d}  pred {pred_loss:.4f}  "
                  f"sigreg {sig_loss:.4f}  total {total:.4f}")

    _writeback_many([enc, pred], theta)
    return enc, pred


def predict_ep(encoder: TreeList, predictor: Node,
               X: np.ndarray) -> np.ndarray:
    theta = _assign_cids_many([encoder, predictor])
    Z = _eval_treelist(encoder, X, theta)
    return _eval(predictor, Z, theta)


def encode(encoder: TreeList, X: np.ndarray) -> np.ndarray:
    theta = _assign_cids_many([encoder])
    return _eval_treelist(encoder, X, theta)


# ======================================================================
# Smoke test
# ======================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Demo 1: single tree (sigmoid recovery)")
    print("=" * 60)

    rng = np.random.default_rng(0)
    X1 = rng.uniform(-3, 3, size=(300, 1))
    y1 = 1.0 / (1.0 + np.exp(-X1[:, 0]))
    y1 += 0.02 * rng.standard_normal(X1.shape[0])

    x = var(0)
    inner = eml(neg(x), const(0.3))
    outer = div(const(0.5), add(const(2.0), tree_ref(inner)))

    print(f"  before: {outer}")
    print(f"  R^2    = {r2_score(outer, X1, y1):.4f}")
    trained = fit(outer, X1, y1, steps=500)
    print(f"  after : {trained}")
    print(f"  R^2    = {r2_score(trained, X1, y1):.4f}")

    print()
    print("=" * 60)
    print("Demo 2: encoder + predictor + SIGReg")
    print("=" * 60)

    rng = np.random.default_rng(1)
    N = 400
    Xm = rng.uniform(-1, 1, size=(N, 4))
    logits = Xm[:, 0] + Xm[:, 1] * Xm[:, 2]
    ym = 1.0 / (1.0 + np.exp(-logits)) + 0.5 * Xm[:, 3]
    ym += 0.05 * rng.standard_normal(N)

    enc = TreeList([
        add(mul(const(0.5), var(0)),
            mul(var(1), var(2))),
        mul(const(1.0), var(3)),
    ])

    sig_inner = eml(neg(var(0)), const(1.0))
    sig_term = div(const(1.0), add(const(1.0), tree_ref(sig_inner)))
    predictor = add(sig_term, mul(const(0.5), var(1)))

    print(f"  encoder (initial):")
    for i, t in enumerate(enc.trees):
        print(f"    z[{i}] = {_format(t)}")
    print(f"  predictor (initial): {_format(predictor)}")

    Z_before = encode(enc, Xm)
    y_pred_before = predict_ep(enc, predictor, Xm)
    mse_before = float(np.mean((y_pred_before - ym) ** 2))
    print(f"  latent var (before): {np.var(Z_before, axis=0).round(3)}")
    print(f"  SIGReg (before): {sigreg(Z_before):.4f}")
    print(f"  MSE (before): {mse_before:.4f}")

    enc_t, pred_t = fit_ep(enc, predictor, Xm, ym,
                           steps=600, lr=0.05, lambd=0.05,
                           verbose=True)

    print(f"\n  encoder (trained):")
    for i, t in enumerate(enc_t.trees):
        print(f"    z[{i}] = {_format(t)}")
    print(f"  predictor (trained): {_format(pred_t)}")

    Z_after = encode(enc_t, Xm)
    y_pred_after = predict_ep(enc_t, pred_t, Xm)
    mse_after = float(np.mean((y_pred_after - ym) ** 2))
    ss_tot = float(np.var(ym)) + 1e-12

    print(f"\n  latent var (after): {np.var(Z_after, axis=0).round(3)}")
    print(f"  SIGReg (after): {sigreg(Z_after):.4f}")
    print(f"  MSE (after): {mse_after:.4f}")
    print(f"  R^2: {1.0 - mse_after / ss_tot:.4f}")
