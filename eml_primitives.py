"""
eml_primitives — minimal primitive encoder library for eml

Five core primitives that serve as "elements" for building encoders:

  linear(i)       — scale * x_i + offset
  sigmoid(i)      — 1 / (1 + exp(-x_i))
  interact(i, j)  — x_i * x_j
  saturate(i)     — eml-based softplus-like bounded growth
  thresh_pos(i)   — exp-dominated activation for positive region

Each primitive has its own trainable constants. They are
composed (via + or *) into an encoder TreeList, and fit_ep
treats all constants uniformly.

Philosophy: structure comes from the user choosing which
primitives to combine, not from random search.
"""

import eml as T


# ----------------------------------------------------------------------
# Five minimum primitives
# ----------------------------------------------------------------------

def linear(i, scale=1.0, offset=0.0):
    """scale * x_i + offset. Trainable: scale, offset."""
    return T.add(
        T.mul(T.const(scale), T.var(i)),
        T.const(offset),
    )


def sigmoid(i, scale=1.0):
    """1 / (1 + exp(-scale * x_i)).
    
    Uses the eml(-scale*x, 1) = exp(-scale*x) - log(1) identity,
    wrapped as a bounded (0,1) output.
    """
    inner = T.eml(
        T.neg(T.mul(T.const(scale), T.var(i))),
        T.const(1.0),
    )
    return T.div(
        T.const(1.0),
        T.add(T.const(1.0), T.tree_ref(inner)),
    )


def interact(i, j, scale=1.0):
    """scale * x_i * x_j. Pairwise multiplicative interaction."""
    return T.mul(
        T.const(scale),
        T.mul(T.var(i), T.var(j)),
    )


def saturate(i, scale=1.0, offset=1.0):
    """eml(scale*x_i, offset) — softplus-like bounded growth.
    
    For small x: ~ constant; for large positive x: grows as exp(scale*x).
    An early-experiment idiom that recovered softplus in one node.
    """
    return T.eml(
        T.mul(T.const(scale), T.var(i)),
        T.const(offset),
    )


def thresh_pos(i, scale=1.0):
    """exp(scale * x_i) — large for positive x, small for negative.
    
    This is eml(scale*x, 1) stripped to its exp arm, useful as
    a soft-gate primitive.
    """
    return T.eml(
        T.mul(T.const(scale), T.var(i)),
        T.const(1.0),
    )


# ----------------------------------------------------------------------
# Three minimal composition operators
# ----------------------------------------------------------------------

def combine_add(a, b, weight=0.5):
    """weight * a + (1 - weight) * b. Weight itself is trainable."""
    w = T.const(weight)
    one_minus_w = T.add(T.const(1.0), T.neg(T.const(weight)))
    return T.add(
        T.mul(w, T.tree_ref(a)),
        T.mul(one_minus_w, T.tree_ref(b)),
    )


def combine_mul(a, b):
    """Multiplicative combination of two primitives."""
    return T.mul(T.tree_ref(a), T.tree_ref(b))


def combine_plain(a, b):
    """a + b, no weights (simpler, fewer constants)."""
    return T.add(T.tree_ref(a), T.tree_ref(b))
