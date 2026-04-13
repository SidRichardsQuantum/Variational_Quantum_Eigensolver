# VQE Optimizers

This document summarizes the classical optimizers available through `vqe.optimizer` and used by the VQE workflows in this repository.

The implemented optimizer names are:

- `Adam`
- `GradientDescent`
- `Momentum`
- `NesterovMomentum`
- `RMSProp`
- `Adagrad`

These are exposed through the registry `OPTIMIZERS`, where each canonical name maps
to:

- the PennyLane optimizer factory
- a calibrated default `stepsize`
- accepted aliases

The calibrated defaults currently used by `run_vqe()` when `stepsize` is omitted are:

- `Adam`: `0.15`
- `GradientDescent`: `0.10`
- `Momentum`: `0.10`
- `NesterovMomentum`: `0.20`
- `RMSProp`: `0.01`
- `Adagrad`: `0.10`

The main helpers are:

```python
get_optimizer(name: str = "Adam", stepsize: float | None = None)
get_optimizer_stepsize(name: str = "Adam") -> float
```

and are used inside the main VQE loop via a unified PennyLane interface.

---

## Scope and Optimization Objective

In VQE, the classical optimization problem is

$$
\min_{\theta \in \mathbb{R}^d} E(\theta),
$$

where

$$
E(\theta) = \langle \psi(\theta) | H | \psi(\theta) \rangle.
$$

Definitions:

- $\theta = (\theta_1, \dots, \theta_d)^\top \in \mathbb{R}^d$ is the vector of variational circuit parameters
- $d$ is the number of trainable parameters in the ansatz
- $|\psi(\theta)\rangle$ is the parameterized trial state prepared by the ansatz
- $H$ is the qubit Hamiltonian for the chosen molecule, basis, and fermion-to-qubit mapping
- $E(\theta)$ is the VQE energy objective in Hartree (Ha)

At optimization step $t$, let:

- $\theta_t$ denote the current parameter vector
- $g_t = \nabla_\theta E(\theta_t)$ denote the gradient of the energy with respect to the parameters
- $\eta > 0$ denote the stepsize (learning rate), passed in this repository as `stepsize`

A generic first-order optimizer updates parameters according to

$$
\theta_{t+1} = \theta_t + \Delta \theta_t,
$$

where the update $\Delta \theta_t$ depends on the optimizer-specific rule.

---

## Repository Context

In `vqe.core.run_vqe`, the optimizer is constructed as

```python
resolved_stepsize = (
    get_optimizer_stepsize(str(optimizer_name))
    if stepsize is None
    else float(stepsize)
)
opt = engine_build_optimizer(str(optimizer_name), stepsize=resolved_stepsize)
```

and then applied through either

```python
params, cost = opt.step_and_cost(energy_qnode, params)
```

or, if unavailable,

```python
params = opt.step(energy_qnode, params)
```

So the role of the optimizer is purely classical:

1. evaluate the VQE objective $E(\theta_t)$
2. compute or use gradient information
3. update the parameter vector $\theta_t \to \theta_{t+1}$

The quantum part prepares states and measures energies; the optimizer determines how the parameters move across iterations.

---

## Common Notation

The formulas below use the following symbols.

- $t \in {0,1,2,\dots}$: optimization iteration index
- $\theta_t \in \mathbb{R}^d$: parameter vector at iteration $t$
- $g_t = \nabla_\theta E(\theta_t) \in \mathbb{R}^d$: gradient at iteration $t$
- $\eta > 0$: global stepsize / learning rate
- $\odot$: elementwise (Hadamard) product
- $g_t^2$: elementwise square of the gradient vector
- $\sqrt{v_t}$: elementwise square root of a vector $v_t$
- $\epsilon > 0$: small numerical stabilizer to avoid division by zero
- $\beta, \beta_1, \beta_2 \in [0,1)$: decay or momentum hyperparameters
- $m_t \in \mathbb{R}^d$: first-moment / momentum accumulator
- $v_t \in \mathbb{R}^d$: second-moment / squared-gradient accumulator

Unless otherwise stated, vector divisions are elementwise.

---

## 1. Gradient Descent

### Update rule

Gradient Descent uses the negative gradient direction directly:

$$
\theta_{t+1} = \theta_t - \eta g_t.
$$

### Variables

- $\theta_t$: current parameters
- $g_t$: gradient of the VQE energy at the current parameters
- $\eta$: fixed stepsize

### Interpretation

This is the simplest optimizer in the repository. It moves in the steepest local descent direction with a constant learning rate.

### Strengths

- simplest and most interpretable baseline
- useful for pedagogical comparisons
- minimal internal state

### Limitations

- highly sensitive to the choice of $\eta$
- can zig-zag in narrow valleys
- can be slow on ill-conditioned landscapes
- may stall or oscillate if the energy landscape has very different curvature scales across parameters

### Typical use in this repo

Gradient Descent is mainly a baseline for optimizer comparisons rather than the default production choice.

---

## 2. Momentum

### Update rule

Momentum augments Gradient Descent with a velocity-like running average of past gradients. A standard form is

$$
m_t = \beta m_{t-1} + g_t,
$$

$$
\theta_{t+1} = \theta_t - \eta m_t.
$$

### Variables

- $m_t$: momentum accumulator at iteration $t$
- $m_{t-1}$: previous momentum accumulator
- $\beta \in [0,1)$: momentum coefficient
- $g_t$: current gradient
- $\eta$: stepsize

### Interpretation

Momentum smooths the update sequence by retaining part of the previous descent direction. This can accelerate motion along persistent downhill directions and suppress some oscillations.

### Strengths

- often faster than plain Gradient Descent
- can reduce oscillatory behaviour
- useful on elongated or shallow valleys

### Limitations

- still requires tuning of $\eta$
- can overshoot minima if momentum is too strong
- less robust than adaptive methods on heterogeneous parameter scales

### Typical use in this repo

Momentum is useful in optimizer-comparison studies where one wants to see whether simple inertial updates improve VQE convergence relative to plain Gradient Descent.

---

## 3. Nesterov Momentum

### Update rule

Nesterov momentum is a momentum-based method with a look-ahead correction. A standard conceptual form is:

1. build a look-ahead point
   $$
   \tilde{\theta}_t = \theta_t - \eta \beta m_{t-1}
   $$

2. evaluate the gradient at the look-ahead point
   $$
   \tilde{g}_t = \nabla_\theta E(\tilde{\theta}_t)
   $$

3. update momentum and parameters
   $$
   m_t = \beta m_{t-1} + \tilde{g}_t
   $$

$$
\theta_{t+1} = \theta_t - \eta m_t
$$

### Variables

- $\tilde{\theta}_t$: look-ahead parameter point
- $\tilde{g}_t$: gradient evaluated at the look-ahead point
- $m_t$: momentum accumulator
- $\beta$: momentum coefficient
- $\eta$: stepsize

### Interpretation

Instead of evaluating the gradient exactly at the current point, Nesterov momentum evaluates it after a partial extrapolation in the current momentum direction. This often gives a more anticipatory update.

### Strengths

- can converge faster than standard momentum in smooth problems
- often improves directional correction
- reduces some forms of overshooting relative to naive momentum

### Limitations

- behaviour still depends on step-size tuning
- benefit can be modest on noisy or irregular objective landscapes
- more difficult to reason about than plain Gradient Descent

### Typical use in this repo

`NesterovMomentum` is a useful intermediate option between simple momentum methods and more adaptive optimizers such as Adam.

---

## 4. Adagrad

### Update rule

Adagrad rescales each parameter update using the history of squared gradients:

$$
v_t = v_{t-1} + g_t^2,
$$

$$
\theta_{t+1} = \theta_t - \eta \frac{g_t}{\sqrt{v_t} + \epsilon}.
$$

### Variables

- $v_t$: accumulated elementwise sum of squared gradients up to step $t$
- $g_t^2$: elementwise square of the current gradient
- $\epsilon$: small positive stabilizer
- $\eta$: base stepsize

### Interpretation

Parameters that repeatedly experience large gradients receive smaller future effective step sizes, while parameters with smaller accumulated gradients receive relatively larger effective steps.

### Strengths

- automatically adapts per-parameter learning rates
- useful when different parameters evolve on very different scales
- can stabilize optimization early in training

### Limitations

- $v_t$ grows monotonically, so effective learning rates continually shrink
- may become overly conservative in longer optimization runs
- can stop making meaningful progress if the accumulated denominator becomes too large

### Typical use in this repo

Adagrad can be informative in VQE studies where parameter sensitivities are highly nonuniform, but it is usually not the first default choice for sustained optimization.

---

## 5. RMSProp

### Update rule

RMSProp modifies Adagrad by replacing the cumulative squared-gradient sum with an exponentially weighted moving average:

$$
v_t = \beta v_{t-1} + (1-\beta) g_t^2,
$$

$$
\theta_{t+1} = \theta_t - \eta \frac{g_t}{\sqrt{v_t} + \epsilon}.
$$

### Variables

- $v_t$: exponential moving average of squared gradients
- $\beta \in [0,1)$: decay rate for the second-moment estimate
- $g_t$: current gradient
- $\eta$: base stepsize
- $\epsilon$: numerical stabilizer

### Interpretation

RMSProp keeps adaptive per-parameter scaling like Adagrad, but avoids the permanently shrinking learning-rate problem by discounting old gradient information.

### Strengths

- more stable than plain Gradient Descent on uneven landscapes
- often better behaved than Adagrad over longer runs
- adaptive scaling can help when gradients vary strongly across parameters

### Limitations

- still requires stepsize tuning
- can be less robust than Adam in practice
- does not include an explicit first-moment momentum term in the basic formulation

### Typical use in this repo

RMSProp is a reasonable adaptive alternative when one wants per-parameter learning-rate normalization without moving to the fuller Adam update.

---

## 6. Adam

### Update rule

Adam combines momentum-like first-moment tracking with RMSProp-style second-moment adaptation.

First moment estimate:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
$$

Second moment estimate:

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

Bias-corrected estimates:

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t},
\qquad
\hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$

Parameter update:

$$
\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}.
$$

### Variables

- $m_t$: exponential moving average of gradients (first moment)
- $v_t$: exponential moving average of squared gradients (second moment)
- $\beta_1 \in [0,1)$: first-moment decay rate
- $\beta_2 \in [0,1)$: second-moment decay rate
- $\hat{m}_t$: bias-corrected first-moment estimate
- $\hat{v}_t$: bias-corrected second-moment estimate
- $\epsilon$: small stabilizer
- $\eta$: learning rate

### Interpretation

Adam combines three useful ideas:

1. **momentum-like smoothing** through $m_t$
2. **adaptive per-parameter scaling** through $v_t$
3. **bias correction** to compensate for zero initialization of the moment estimates at early iterations

### Strengths

- often robust across a wide range of VQE settings
- tends to work well without extensive tuning
- handles heterogeneous gradient scales better than plain Gradient Descent
- commonly a strong default for small and medium variational problems

### Limitations

- can sometimes converge to slightly less clean final minima than carefully tuned simpler methods
- still sensitive to overly aggressive stepsizes
- the most complex update rule among the implemented first-order optimizers here

### Typical use in this repo

Adam is the default optimizer in this repository and is generally the first choice for standard VQE runs.

---

## Practical Comparison

The optimizers in this repository can be grouped roughly as follows.

### Fixed-step methods

- `GradientDescent`
- `Momentum`
- `NesterovMomentum`

These methods use a global learning rate $\eta$ without adaptive per-parameter normalization. They are simple and interpretable, but usually more sensitive to hyperparameter tuning.

### Adaptive methods

- `Adagrad`
- `RMSProp`
- `Adam`

These methods rescale updates using gradient-history information. They are often more robust when the variational parameters have different sensitivities or when the local energy landscape is poorly conditioned.

---

## Which Optimizer Should Be Tried First?

For most standard VQE experiments in this repository:

1. start with `Adam`
2. use `GradientDescent` as a baseline if you want a clean reference
3. try `Momentum` or `NesterovMomentum` if you want simple inertial alternatives
4. try `RMSProp` or `Adagrad` when parameter scales appear uneven

A practical rule of thumb is:

- **Adam** for general-purpose default use
- **GradientDescent** for interpretability and baseline studies
- **Momentum / NesterovMomentum** for simple acceleration over GD
- **RMSProp / Adagrad** for stronger per-parameter adaptation

---

## Relation to Noise and Ansatz Choice

The optimizer acts on the classical objective produced by the quantum circuit, so its behaviour depends indirectly on:

- the ansatz family
- the number of trainable parameters
- the molecular Hamiltonian
- the fermion-to-qubit mapping
- whether the circuit execution is noiseless or noisy

In particular:

- a more expressive ansatz may give a lower reachable minimum but a harder optimization landscape
- noisy execution can distort gradients or make convergence less smooth
- different mappings can change Pauli structure and therefore the measured objective landscape
- different optimizers may respond differently to the same VQE instance

This is why the repository includes dedicated optimizer-comparison workflows such as `run_vqe_optimizer_comparison`.

---

## Notes on Exact Hyperparameters

This repository passes only the user-facing `stepsize` explicitly through the optimizer factory:

```python
get_optimizer(name, stepsize)
```

Other internal optimizer hyperparameters (e.g. momentum or decay coefficients) are not exposed in this repository and are inherited directly from the underlying PennyLane implementations.

So:

- the formulas in this document describe the mathematical update structure
- the concrete default hyperparameter values beyond `stepsize` are determined by PennyLane unless the implementation is extended in future

---

## Implemented Name Mapping

The current optimizer registry is:

```python
OPTIMIZERS = {
    "Adam": {"factory": qml.AdamOptimizer, "stepsize": 0.15, "aliases": ("adam",)},
    "GradientDescent": {
        "factory": qml.GradientDescentOptimizer,
        "stepsize": 0.10,
        "aliases": ("gradientdescent", "gradient_descent", "gd"),
    },
    "Momentum": {
        "factory": qml.MomentumOptimizer,
        "stepsize": 0.10,
        "aliases": ("momentum",),
    },
    "NesterovMomentum": {
        "factory": qml.NesterovMomentumOptimizer,
        "stepsize": 0.20,
        "aliases": ("nesterov", "nesterovmomentum"),
    },
    "RMSProp": {"factory": qml.RMSPropOptimizer, "stepsize": 0.01, "aliases": ("rmsprop",)},
    "Adagrad": {"factory": qml.AdagradOptimizer, "stepsize": 0.10, "aliases": ("adagrad",)},
}
```

So the canonical user-facing names are:

- `Adam`
- `GradientDescent`
- `Momentum`
- `NesterovMomentum`
- `RMSProp`
- `Adagrad`

Accepted aliases also include:

- `adam`
- `gd`
- `nesterov`

---

## Summary Table

| Optimizer         | Core idea                                           | Uses momentum? | Adaptive per-parameter scaling? | Main trade-off                                        |
| ----------------- | --------------------------------------------------- | -------------: | ------------------------------: | ----------------------------------------------------- |
| `GradientDescent` | direct negative-gradient update                     |             No |                              No | simple but stepsize-sensitive                         |
| `Momentum`        | gradient descent with velocity accumulation         |            Yes |                              No | faster than GD, can overshoot                         |
| `NesterovMomentum` | momentum with look-ahead gradient                  |            Yes |                              No | often sharper updates, still tuning-sensitive         |
| `Adagrad`         | cumulative squared-gradient scaling                 |             No |                             Yes | adapts well early, can become too conservative        |
| `RMSProp`         | moving-average squared-gradient scaling             |             No |                             Yes | adaptive and stable, but less full-featured than Adam |
| `Adam`            | momentum + adaptive second moment + bias correction |            Yes |                             Yes | robust default, but more complex                      |

---

## See Also

- [`THEORY.md`](../../THEORY.md) — high-level VQE theory and algorithm context
- `vqe/optimizer.py` — optimizer factory and supported names
- `vqe/core.py` — main VQE loop using the optimizer
- `USAGE.md` — practical usage and CLI examples
