
# eml

Minimal tree-based differentiable symbolic computation in pure numpy.

A single primitive operator — `eml(x, y) = exp(x) − log|y|` — composed with basic arithmetic into trees that can be both evaluated and learned. No PyTorch. No GPU. Just numpy.

## Why

Neural networks hide their function inside opaque weight tensors. Symbolic regression produces readable formulas but is mostly decoupled from gradient-based learning. `eml` sits in between: small computational trees whose structure is explicit and whose constants are learned by Adam.

The `eml` primitive is interesting because shallow compositions of it can approximate many elementary functions — sigmoid, softplus, relu-like shapes, absolute value, piecewise branches — often without needing dedicated operators. Branching behavior emerges from the arithmetic of `exp` and `log` rather than from built-in `if` nodes.

## Install

No package yet. Drop `eml.py` into your project.

```bash
curl -O https://raw.githubusercontent.com/sailfish009/eml/main/eml.py
```

Requires numpy only.

## Quick start

```python
import eml as T
import numpy as np

# Target: sigmoid
X = np.random.uniform(-3, 3, size=(300, 1))
y = 1.0 / (1.0 + np.exp(-X[:, 0]))

# Define architecture: 1 / (1 + eml(-x, c)) with non-ideal initial constants
x = T.var(0)
inner = T.eml(T.neg(x), T.const(0.3))
model = T.div(T.const(0.5), T.add(T.const(2.0), T.tree_ref(inner)))

# Fit constants with Adam
trained = T.fit(model, X, y, steps=500)

print(trained)
print(f"R^2 = {T.r2_score(trained, X, y):.4f}")
```

Output:

```
(1.006 / (1.206 + <eml((-x0),1.22)>))
R^2 = 0.9963
```

The three `const` nodes converge to values that reconstruct `sigmoid(x)` from the `eml` primitive.

## Core concepts

### The `eml` primitive

```
eml(x, y) = exp(x) − log|y|
```

Two arguments, one real output. Combined with `+`, `*`, `/`, it can express:

- `exp(x) = eml(x, 1)`
- `log|x| = 1 − eml(0, x)`
- Sigmoid, softplus, Gaussian, Lorentzian shapes
- Soft branching behaviour (the `exp` arm dominates or the `log` arm dominates depending on the sign of `x`)

### Trees are data

Every expression is a `Node`. Constants (`const`) are trainable slots in a global parameter vector; variables (`var`) read from input columns; binary and unary operators compose subtrees.

```python
a = T.const(0.5)           # trainable scalar
x = T.var(0)               # reads X[:, 0]
expr = T.add(T.mul(a, x), T.const(1.0))   # 0.5 * x + 1.0
```

### Tree as atom

The distinctive feature: a tree can appear as an atom inside another tree.

```python
# Inner computation — any eml tree
activation = T.eml(T.var(0), T.const(1.0))

# Outer uses the inner as if it were a leaf
model = T.mul(T.tree_ref(activation), T.var(0))
```

When `fit` runs, it collects constants from every composed tree into one `theta` vector and optimises them jointly. There is one optimisation problem, not many.

Pretty-printed, composed trees appear inside `< >`:

```
(<eml(x0,1)> * x0)
```

### Encoder + predictor (vector-valued trees)

For building small models in the spirit of world models — an encoder that maps inputs to a latent representation, and a predictor that consumes the latent — the library provides `TreeList` and `fit_ep`.

```python
# Encoder: 4 input features → 2-dim latent
encoder = T.TreeList([
    T.add(T.mul(T.const(0.5), T.var(0)),
          T.mul(T.var(1), T.var(2))),     # latent[0]
    T.mul(T.const(1.0), T.var(3)),         # latent[1]
])

# Predictor: latent → scalar.  var(j) inside predictor reads latent[j].
sig_inner = T.eml(T.neg(T.var(0)), T.const(1.0))
sig_term  = T.div(T.const(1.0), T.add(T.const(1.0), T.tree_ref(sig_inner)))
predictor = T.add(sig_term, T.mul(T.const(0.5), T.var(1)))

# Joint training with anti-collapse regularization on the latent
enc_t, pred_t = T.fit_ep(encoder, predictor, X, y,
                         steps=600, lr=0.05, lambd=0.05)

y_pred = T.predict_ep(enc_t, pred_t, X)
Z      = T.encode(enc_t, X)   # latent only
```

The encoder is a `TreeList` (a bundle of k scalar trees, producing a k-dim vector). The predictor is a single scalar tree whose `var(j)` reads the j-th latent dimension. Training minimises

```
MSE(predictor(encoder(X)), y)  +  lambd * SIGReg(encoder(X))
```

where `SIGReg` pushes the latent distribution toward a standard normal via characteristic-function matching on random 1D projections. This prevents the encoder from collapsing to a constant or low-rank representation.

### Primitives (optional companion library)

Trees can be composed directly, but for repeated use the companion module
`eml_primitives` provides a small set of named building blocks — in the
spirit of chemical elements: a few irreducible pieces that combine into
a vast space of encoders.

```python
from eml_primitives import linear, sigmoid, interact, saturate, thresh_pos

# Build an encoder by composing primitives
encoder = T.TreeList([
    T.add(linear(0), interact(1, 2)),   # z[0] = scale*x0 + x1*x2
    linear(3),                           # z[1] = scale*x3 + offset
])
```

Each primitive carries its own trainable constants. The resulting
encoder is a regular `TreeList` and trains exactly like any other
with `fit_ep`. The advantage is readability: after training, the
learned encoder reads back as a composition of named concepts rather
than an arbitrary tree.

`eml_primitives` is entirely optional and lives in a separate file.
It depends on `eml` but `eml` does not depend on it.

## API

### Builders

| Function | Meaning |
|---|---|
| `var(i)` | i-th input column |
| `const(v)` | trainable scalar, initial value `v` |
| `neg(a)` | `−a` |
| `sin(a)` | `sin(a)` |
| `add(a, b)` | `a + b` |
| `mul(a, b)` | `a * b` |
| `div(a, b)` | `a / b` (with small denominator guard) |
| `eml(a, b)` | `exp(a) − log|b|` |
| `tree_ref(inner)` | use `inner` as an atom |
| `TreeList(trees)` | bundle of k scalar trees with vector output |

### Training and inference

| Function | Meaning |
|---|---|
| `fit(tree, X, y, steps, lr, verbose)` | Adam on all constants, returns new tree |
| `predict(tree, X)` | forward evaluation |
| `r2_score(tree, X, y)` | coefficient of determination |
| `trainable_count(tree)` | number of `const` nodes |
| `fit_ep(encoder, predictor, X, y, steps, lr, lambd)` | joint training with SIGReg |
| `predict_ep(encoder, predictor, X)` | encoder → predictor forward |
| `encode(encoder, X)` | encoder forward only (returns latent) |
| `sigreg(Z, M, seed)` | SIGReg value on a latent batch |

`fit` and `fit_ep` return new trees; the caller's trees are not mutated.

### Primitives (in `eml_primitives.py`)

| Function | Meaning |
|---|---|
| `linear(i, scale, offset)` | `scale * x_i + offset` |
| `sigmoid(i, scale)` | bounded `1 / (1 + exp(-scale * x_i))` |
| `interact(i, j, scale)` | pairwise product `scale * x_i * x_j` |
| `saturate(i, scale, offset)` | `eml(scale*x_i, offset)` — softplus-like |
| `thresh_pos(i, scale)` | `exp(scale * x_i)` — positive-region activation |

Composition helpers: `combine_add(a, b, w)`, `combine_mul(a, b)`,
`combine_plain(a, b)`.

## Examples

### Emergent sigmoid

Given a sigmoid-shaped target and only `{eml, +, *, /, const, var}`, the constants converge so that `eml(−x, 1)` plays the role of `exp(−x)` inside a `1 / (1 + ·)` envelope. The primitive didn't "know" it was a sigmoid; the arithmetic made it one.

### Softplus in one node

```python
# Target: log(1 + exp(x))
model = T.eml(T.mul(T.const(0.4), T.var(0)), T.const(1.3))
trained = T.fit(model, X, y, steps=500)
# R^2 ≈ 0.996
```

A single `eml` node, two constants, approximates softplus over a practical range.

### Two-branch piecewise

With the same primitive set, `x² if x > 0 else −x` is recoverable as an `eml(c·x, ...) · x` idiom, where the `eml` factor dominates in one half and is suppressed in the other. Branch selection is implicit in the arithmetic.

### Encoder + predictor with non-degenerate latent

On synthetic data where the target is `sigmoid(x₀ + x₁·x₂) + 0.5·x₃`, a 4→2→1 encoder+predictor configuration trained with `fit_ep` reaches R² ≈ 0.98. During training the SIGReg term decreases while latent variance grows — the opposite of collapse.

### Primitive-based encoder

The same target `sigmoid(x₀ + x₁·x₂) + 0.5·x₃` can be approached by
composing primitives rather than writing trees by hand:

```python
from eml_primitives import linear, interact

encoder = T.TreeList([
    T.add(linear(0), interact(1, 2)),   # captures the logit
    linear(3),                           # captures the additive term
])
```

After `fit_ep`, this reaches R² ≈ 0.9997. The learned encoder reads
back as the composition of named primitives with fitted constants,
keeping the model self-documenting.

## Design notes

- **Functional API.** Builders return new nodes. Trees are treated as mostly immutable; `fit` and `fit_ep` clone before modifying.
- **Shared theta.** All constants across composed trees (including TreeLists and nested refs) live in a single 1-D `theta`. Encoder and predictor train together in one optimisation problem.
- **Gradient chain through the latent.** `fit_ep` propagates both the prediction error and the SIGReg gradient back through the encoder's constants. This requires a small "variable-gradient" pass in addition to the usual "constant-gradient" pass.
- **Safe numerics.** `exp` is clipped, `log` uses `|·|` plus small `eps`, `/` guards its denominator. These guards preserve gradient flow; they are not exact mathematical definitions. If you want pure `eml(x, y) = exp(x) − log y`, restrict `y > 0` in your data and remove the safeguards.
- **Structure vs constants.** `fit` and `fit_ep` train constants only. Structure is whatever the user builds. Automatic structure search (genetic algorithms over tree space) belongs in a separate module, not here.

## What this is not

- Not a production deep-learning framework. No GPU, no batching tricks, no fused kernels.
- Not a symbolic-math library. There is no `simplify`, no symbolic differentiation, no `sympy` integration here. Just numerical gradient and evaluation.
- Not a genetic-programming library. GA-based structure search is a natural companion but lives elsewhere.
- Not a fixed architecture framework. Encoders and predictors are
  user-defined trees, optionally built from the `eml_primitives`
  companion library.

The goal is a clean substrate for experimenting with the `eml` primitive and tree composition, not to compete with PyTorch or SymPy.

## Background

The `eml` primitive is introduced in [this paper](https://arxiv.org/pdf/2603.21852v2), which proposes that a single two-argument real function can play a role for continuous mathematics analogous to what NAND plays for Boolean logic. This repository is one concrete, minimal playground for exploring that claim through learning.

The encoder + predictor design is inspired by [LeWorldModel](https://arxiv.org/pdf/2603.19312v1): an encoder maps observations to a compact latent, a predictor models dynamics in that latent, and an anti-collapse regularizer prevents the trivial constant solution. This library adapts the same skeleton to static tabular data, replacing Vision Transformer + full SIGReg with tree-based encoder + tree-based predictor + a lightweight characteristic-function SIGReg.

## License

Apache-2.0

## Demo

```
============================================================
Demo 1: single tree (sigmoid recovery)
============================================================
  before: (0.5 / (2 + <eml((-x0),0.3)>))
  R^2    = -1.5974
  after : (1.006 / (1.206 + <eml((-x0),1.22)>))
  R^2    = 0.9963

============================================================
Demo 2: encoder + predictor + SIGReg
============================================================
  encoder (initial):
    z[0] = ((0.5 * x0) + (x1 * x2))
    z[1] = (1 * x3)
  predictor (initial): ((1 / (1 + <eml((-x0),1)>)) + (0.5 * x1))
  latent var (before): [0.207 0.337]
  SIGReg (before): 0.1047
  MSE (before): 0.0067
[fit_ep] step   60  pred 0.0026  sigreg 0.0079  total 0.0030
[fit_ep] step  120  pred 0.0026  sigreg 0.0071  total 0.0030
[fit_ep] step  180  pred 0.0026  sigreg 0.0076  total 0.0030
[fit_ep] step  240  pred 0.0026  sigreg 0.0079  total 0.0030
[fit_ep] step  300  pred 0.0027  sigreg 0.0059  total 0.0030
[fit_ep] step  360  pred 0.0026  sigreg 0.0078  total 0.0030
[fit_ep] step  420  pred 0.0027  sigreg 0.0074  total 0.0030
[fit_ep] step  480  pred 0.0026  sigreg 0.0077  total 0.0030
[fit_ep] step  540  pred 0.0026  sigreg 0.0073  total 0.0030
[fit_ep] step  600  pred 0.0026  sigreg 0.0066  total 0.0030

  encoder (trained):
    z[0] = ((1.192 * x0) + (x1 * x2))
    z[1] = (1.582 * x3)
  predictor (trained): ((1.12 / (1.111 + <eml((-x0),0.9084)>)) + (0.3074 * x1))

  latent var (after): [0.579 0.843]
  SIGReg (after): 0.0072
  MSE (after): 0.0026
  R^2: 0.9757
```
