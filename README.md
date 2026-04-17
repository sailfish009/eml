# eml

Minimal tree-based differentiable symbolic computation in pure numpy.

A single primitive operator — `eml(x, y) = exp(x) − log|y|` — composed with basic arithmetic into trees that can be both evaluated and learned. No PyTorch. No GPU. Just numpy.

## Why

Neural networks hide their function inside opaque weight tensors. Symbolic regression produces readable formulas but is mostly decoupled from gradient-based learning. `eml` sits in between: small computational trees whose structure is explicit and whose constants are learned by Adam.

The `eml` primitive is interesting because shallow compositions of it can approximate many elementary functions — sigmoid, softplus, relu-like shapes, absolute value, piecewise branches — often without needing dedicated operators. Branching behavior emerges from the arithmetic of `exp` and `log` rather than from built-in `if` nodes.

## Install

No package yet. Drop `eml.py` into your project.

```bash
curl -O https://raw.githubusercontent.com/<user>/<repo>/main/eml.py
```

Requires numpy only.

## Quick start

```python
import eml as T
import numpy as np

# Target: sigmoid
X = np.random.uniform(-3, 3, size=(300, 1))
y = 1.0 / (1.0 + np.exp(-X[:, 0]))

# Define architecture: 1 / (1 + eml(-x, c))
x = T.var(0)
model = T.div(
    T.const(1.0),
    T.add(T.const(1.0), T.eml(T.neg(x), T.const(1.0)))
)

# Fit constants with Adam
trained = T.fit(model, X, y, steps=500)

print(trained)
print(f"R^2 = {T.r2_score(trained, X, y):.4f}")
```

Output:

```
(1.006 / (1.017 + eml((-x0),1.01)))
R^2 = 0.9963
```

The three `const` nodes moved from their initial `1.0` values to `1.006`, `1.017`, `1.01` — reconstructing `sigmoid(x)` from the `eml` primitive.

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

### Training and inference

| Function | Meaning |
|---|---|
| `fit(tree, X, y, steps, lr, verbose)` | Adam on all constants, returns new tree |
| `predict(tree, X)` | forward evaluation |
| `r2_score(tree, X, y)` | coefficient of determination |
| `trainable_count(tree)` | number of `const` nodes |

`fit` returns a new tree; the caller's tree is not mutated.

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

## Design notes

- **Functional API.** Builders return new nodes. Trees are treated as mostly immutable; `fit` clones before modifying.
- **Shared theta.** All constants across composed trees live in a single 1-D `theta`. This is what makes tree-as-atom coherent — inner and outer are trained together.
- **Safe numerics.** `exp` is clipped, `log` uses `|·|` plus small `eps`, `/` guards its denominator. These guards preserve gradient flow; they are not exact mathematical definitions. If you want pure `eml(x, y) = exp(x) − log y`, restrict `y > 0` in your data and remove the safeguards.
- **Structure vs constants.** `fit` trains constants only. Structure is whatever the user builds. Automatic structure search (genetic algorithms over tree space) belongs in a separate module, not here.

## What this is not

- Not a production deep-learning framework. No GPU, no batching tricks, no fused kernels.
- Not a symbolic-math library. There is no `simplify`, no symbolic differentiation, no `sympy` integration here. Just numerical gradient and evaluation.
- Not a genetic-programming library. GA-based structure search is a natural companion but lives elsewhere.

The goal is a clean substrate for experimenting with the `eml` primitive and tree composition, not to compete with PyTorch or SymPy.

## Background

## Background

The `eml` primitive is introduced and motivated in [this paper](https://arxiv.org/pdf/2603.21852v2), which proposes that a single two-argument real function can play a role for continuous mathematics analogous to what NAND plays for Boolean logic. This repository is one concrete, minimal playground for exploring that claim through learning.

## License

Apache-2.0
