"""
experiment_primitives — test primitive-based encoder composition

Synthetic target: y = sigmoid(x0 + x1*x2) + 0.5 * x3

Three encoder variants built from primitives:
  E1: raw linear features only (weak)
  E2: interact + linear mixed (medium)
  E3: matches target structure (expected strong)

Each variant is trained with fit_ep + SIGReg.
We compare R^2, latent structure, and readability.
"""

import numpy as np
import eml as T
from eml_primitives import (
    linear, sigmoid, interact, saturate, thresh_pos,
    combine_plain, combine_mul, combine_add,
)


# ----------------------------------------------------------------------
# Synthetic data
# ----------------------------------------------------------------------

def make_synthetic(n=400, seed=3, noise=0.05):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 4))
    y_clean = 1.0 / (1.0 + np.exp(-(X[:, 0] + X[:, 1] * X[:, 2]))) + 0.5 * X[:, 3]
    y = y_clean + noise * rng.standard_normal(n)
    return X, y, y_clean


# ----------------------------------------------------------------------
# Three encoder variants built from primitives
# ----------------------------------------------------------------------

def build_encoders():
    """Return a dict {name: encoder_TreeList} for the three variants."""

    # E1: naive — each latent is just a linear in one feature
    E1 = T.TreeList([linear(0), linear(1), linear(2)])

    # E2: mixed — some linears plus one interaction
    E2 = T.TreeList([
        linear(0),
        interact(1, 2),   # x1 * x2
        linear(3),
    ])

    # E3: structure-matching — the logit x0 + x1*x2 as one latent,
    #     x3 as another.  Target has two effective components; we
    #     give exactly two latents.
    logit = T.add(linear(0, scale=1.0, offset=0.0),
                  interact(1, 2))
    E3 = T.TreeList([
        logit,
        linear(3),
    ])

    return {'E1_naive': E1, 'E2_mixed': E2, 'E3_matched': E3}


# ----------------------------------------------------------------------
# Predictor: sigmoid on latent[0], linear on latent[1]
# ----------------------------------------------------------------------

def build_predictor_2d():
    """For 2-dim latent (E3)."""
    sig_inner = T.eml(T.neg(T.var(0)), T.const(1.0))
    sig_term = T.div(T.const(1.0),
                     T.add(T.const(1.0), T.tree_ref(sig_inner)))
    return T.add(sig_term, T.mul(T.const(0.5), T.var(1)))


def build_predictor_3d():
    """For 3-dim latent (E1, E2).  Uses all three latents; last is slack."""
    sig_inner = T.eml(T.neg(T.var(0)), T.const(1.0))
    sig_term = T.div(T.const(1.0),
                     T.add(T.const(1.0), T.tree_ref(sig_inner)))
    return T.add(
        T.add(sig_term, T.mul(T.const(0.5), T.var(1))),
        T.mul(T.const(0.1), T.var(2)),
    )


# ----------------------------------------------------------------------
# Run one variant
# ----------------------------------------------------------------------

def run_variant(name, encoder, predictor, X, y, y_clean,
                steps=800, lambd=0.05, seed=0):
    enc_t, pred_t = T.fit_ep(encoder, predictor, X, y,
                             steps=steps, lr=0.05,
                             lambd=lambd, sigreg_seed=seed)

    y_pred = T.predict_ep(enc_t, pred_t, X)
    mse = float(np.mean((y_pred - y_clean) ** 2))
    ss_tot = float(np.var(y_clean)) + 1e-12
    r2 = 1.0 - mse / ss_tot

    Z = T.encode(enc_t, X)
    latent_std = Z.std(axis=0)
    latent_corr = [float(np.corrcoef(Z[:, j], y_clean)[0, 1])
                   for j in range(Z.shape[1])]
    sig_val = T.sigreg(Z)

    print(f"\n{'=' * 60}")
    print(f"{name}")
    print(f"{'=' * 60}")
    print(f"  R^2 (vs clean)     : {r2:.4f}")
    print(f"  latent std per dim : {latent_std.round(3)}")
    print(f"  |corr(z, y_clean)| : {[round(abs(c), 3) for c in latent_corr]}")
    print(f"  SIGReg value       : {sig_val:.4f}")
    print(f"  trained encoder:")
    for i, t in enumerate(enc_t.trees):
        print(f"    z[{i}] = {t}")
    print(f"  trained predictor  : {pred_t}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

if __name__ == "__main__":
    X, y, y_clean = make_synthetic(n=400, seed=3, noise=0.05)

    print(f"Target: sigmoid(x0 + x1*x2) + 0.5 * x3")
    print(f"Data  : N={X.shape[0]}, features={X.shape[1]}")
    print(f"        y_clean var = {np.var(y_clean):.4f}")

    encoders = build_encoders()

    # E1, E2 use 3-dim predictor; E3 uses 2-dim
    run_variant('E1_naive',
                encoders['E1_naive'],
                build_predictor_3d(),
                X, y, y_clean)

    run_variant('E2_mixed',
                encoders['E2_mixed'],
                build_predictor_3d(),
                X, y, y_clean)

    run_variant('E3_matched',
                encoders['E3_matched'],
                build_predictor_2d(),
                X, y, y_clean)
