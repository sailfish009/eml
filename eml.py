"""
eml — minimal library for tree-based differentiable symbolic computation (numpy only)

Core primitive: eml(x, y) = exp(x) - log|y|
Supports: +, *, /, sin, neg, eml, const, var, and tree composition (tree-as-atom).

Design:
    - Functional API: T.add(a, b), not a.add(b)
    - Trees are (mostly) immutable; new tree returned from each op
    - Inner trees can be composed into outer trees as atoms (tree-as-atom)
    - Shared theta across all composed trees: single optimization problem
    - Fixed architecture + Adam for constants (GA lives in a separate module)

Typical usage:
    import eml as T
    x = T.var(0)
    inner = T.div(T.const(1.0),
                  T.add(T.const(1.0), T.eml(T.neg(x), T.const(1.0))))
    outer = T.mul(inner, T.add(x, T.const(0.1)))
    trained = T.fit(outer, X, y, steps=500)
    y_pred = T.predict(trained, X_test)
    print(trained)
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
    """Immutable-ish tree node.

    op: 'var' | 'const' | 'neg' | 'sin' | '+' | '*' | '/' | 'eml' | 'ref'
    'ref' means: children[0] is another Tree whose output becomes this atom's value.
    """
    op: str
    children: List['Node'] = field(default_factory=list)
    cid: Optional[int] = None   # index into global theta, assigned at fit time
    init_val: float = 1.0       # for const
    var_idx: int = 0            # for var

    def __repr__(self):
        return _format(self)


# ======================================================================
# Builder functions (public API)
# ======================================================================

def var(i: int) -> Node:
    """Variable atom. i-th column of input X."""
    n = Node('var'); n.var_idx = int(i); return n

def const(v: float) -> Node:
    """Trainable constant. Initial value v."""
    n = Node('const'); n.init_val = float(v); return n

def neg(a: Node) -> Node:
    return Node('neg', [a])

def sin(a: Node) -> Node:
    return Node('sin', [a])

def add(a: Node, b: Node) -> Node:
    return Node('+', [a, b])

def mul(a: Node, b: Node) -> Node:
    return Node('*', [a, b])

def div(a: Node, b: Node) -> Node:
    return Node('/', [a, b])

def eml(a: Node, b: Node) -> Node:
    """The core primitive: exp(a) - log|b|."""
    return Node('eml', [a, b])

def tree_ref(inner: Node) -> Node:
    """Use another tree as an atom. Inner tree sees the same input X as outer.

    Example:
        inner = T.eml(T.var(0), T.const(1.0))
        outer = T.mul(T.tree_ref(inner), T.var(0))
    """
    return Node('ref', [inner])


# ======================================================================
# Tree traversal / introspection
# ======================================================================

def _all_nodes(n: Node):
    """Yield every node including the composed-in 'ref' targets."""
    yield n
    for c in n.children:
        yield from _all_nodes(c)

def size(n: Node) -> int:
    """Count nodes, including inside any composed trees."""
    return sum(1 for _ in _all_nodes(n))

def clone(n: Node) -> Node:
    nn = Node(n.op, [clone(c) for c in n.children], n.cid, n.init_val)
    nn.var_idx = n.var_idx
    return nn


# ======================================================================
# Constant indexing (theta layout)
# ======================================================================

def _assign_cids(n: Node, theta_out: List[float]) -> None:
    """Walk entire tree (including refs) and assign each const a slot in theta."""
    if n.op == 'const':
        n.cid = len(theta_out)
        theta_out.append(n.init_val)
    for c in n.children:
        _assign_cids(c, theta_out)

def _writeback(n: Node, theta: np.ndarray) -> None:
    """Copy trained values from theta back into tree's init_val fields."""
    if n.op == 'const' and n.cid is not None:
        n.init_val = float(theta[n.cid])
    for c in n.children:
        _writeback(c, theta)


# ======================================================================
# Evaluation (forward only)
# ======================================================================

def _eval(n: Node, X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    if n.op == 'var':
        return X[:, n.var_idx]
    if n.op == 'const':
        return np.full(X.shape[0],
                       theta[n.cid] if n.cid is not None else n.init_val)
    if n.op == 'ref':
        # Composed tree: just evaluate its inner root with same X
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
        # Handle b == 0 exactly: sign is 0 so fall back
        b_safe = np.where(b_safe == 0, DIV_EPS, b_safe)
        return a / b_safe
    if n.op == 'eml':
        a = _eval(n.children[0], X, theta)
        b = _eval(n.children[1], X, theta)
        a_c = np.clip(a, -EXP_CLIP, EXP_CLIP)
        return np.exp(a_c) - np.log(np.maximum(np.abs(b), LN_EPS))
    raise ValueError(f"unknown op: {n.op}")


# ======================================================================
# Forward + gradient wrt theta (dict: cid -> (N,) per-sample partial)
# ======================================================================

Grad = Dict[int, np.ndarray]

def _eval_grad(n: Node, X: np.ndarray, theta: np.ndarray) -> Tuple[np.ndarray, Grad]:
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
        out = a + b
        g = dict(ga)
        for k, vv in gb.items(): g[k] = g.get(k, 0.0) + vv
        return out, g
    if n.op == '*':
        a, ga = _eval_grad(n.children[0], X, theta)
        b, gb = _eval_grad(n.children[1], X, theta)
        out = a * b
        g = {k: vv * b for k, vv in ga.items()}
        for k, vv in gb.items(): g[k] = g.get(k, 0.0) + a * vv
        return out, g
    if n.op == '/':
        a, ga = _eval_grad(n.children[0], X, theta)
        b, gb = _eval_grad(n.children[1], X, theta)
        b_safe = np.sign(b) * (np.abs(b) + DIV_EPS)
        b_safe = np.where(b_safe == 0, DIV_EPS, b_safe)
        out = a / b_safe
        inv_b = 1.0 / b_safe
        g = {k: vv * inv_b for k, vv in ga.items()}
        neg_over = -a / (b_safe * b_safe)
        for k, vv in gb.items(): g[k] = g.get(k, 0.0) + vv * neg_over
        return out, g
    if n.op == 'eml':
        a, ga = _eval_grad(n.children[0], X, theta)
        b, gb = _eval_grad(n.children[1], X, theta)
        a_c = np.clip(a, -EXP_CLIP, EXP_CLIP)
        ev = np.exp(a_c)
        ab = np.maximum(np.abs(b), LN_EPS)
        mask = (np.abs(a) <= EXP_CLIP).astype(float)
        out = ev - np.log(ab)
        g = {k: vv * ev * mask for k, vv in ga.items()}
        inv = -np.sign(b) / ab
        for k, vv in gb.items(): g[k] = g.get(k, 0.0) + vv * inv
        return out, g
    raise ValueError(f"unknown op: {n.op}")


# ======================================================================
# Training: fit constants with Adam
# ======================================================================

def fit(tree: Node, X: np.ndarray, y: np.ndarray,
        steps: int = 500, lr: float = 0.05,
        verbose: bool = False) -> Node:
    """Fit all const nodes (in tree and any composed trees) to minimize MSE.

    Returns the same tree with const init_vals updated. Does NOT modify structure.
    """
    tree = clone(tree)  # don't mutate caller's tree
    theta_list: List[float] = []
    _assign_cids(tree, theta_list)

    if not theta_list:
        return tree  # nothing trainable

    theta = np.array(theta_list, dtype=np.float64)
    m = np.zeros_like(theta); v = np.zeros_like(theta)
    b1, b2, eps = 0.9, 0.999, 1e-8

    for t in range(1, steps + 1):
        pred, grads = _eval_grad(tree, X, theta)
        if not np.all(np.isfinite(pred)):
            if verbose: print(f"[fit] step {t}: non-finite, stopping")
            break
        resid = pred - y
        g = np.zeros_like(theta)
        for cid, dp in grads.items():
            gv = float(np.mean(2.0 * resid * dp))
            g[cid] = np.clip(gv, -10.0, 10.0) if np.isfinite(gv) else 0.0
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        theta -= lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
        if verbose and (t % max(1, steps // 10) == 0):
            loss = float(np.mean(resid ** 2))
            print(f"[fit] step {t:4d}  loss {loss:.6f}")

    _writeback(tree, theta)
    return tree


# ======================================================================
# Inference
# ======================================================================

def predict(tree: Node, X: np.ndarray) -> np.ndarray:
    """Evaluate tree on X. Uses const.init_val (no external theta)."""
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
# Example / smoke test
# ======================================================================

if __name__ == "__main__":
    # Small demo: compose trees, fit, predict.
    rng = np.random.default_rng(0)
    X = rng.uniform(-3, 3, size=(300, 1))
    y = 1.0 / (1.0 + np.exp(-X[:, 0]))                   # sigmoid target
    y += 0.02 * rng.standard_normal(X.shape[0])

    # User-defined architecture: sigmoid structure, all constants trainable.
    x = var(0)
    inner = eml(neg(x), const(1.0))                       # exp(-x) - log|c|
    outer = div(const(1.0), add(const(1.0), tree_ref(inner)))

    print("Before fit:")
    print(f"  {outer}")
    print(f"  R^2 = {r2_score(outer, X, y):.4f}  (random init)")
    print(f"  trainable constants: {trainable_count(outer)}")

    trained = fit(outer, X, y, steps=500, verbose=False)

    print("\nAfter fit:")
    print(f"  {trained}")
    print(f"  R^2 = {r2_score(trained, X, y):.4f}")
