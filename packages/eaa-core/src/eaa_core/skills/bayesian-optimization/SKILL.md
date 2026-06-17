---
name: bayesian-optimization
description: Design and run Bayesian optimization or adaptive sampling loops for expensive, noisy, multivariate, or non-convex measurement problems.
---

## Overview

Use Bayesian optimization (BO) when the user needs to optimize or adaptively sample
an expensive black-box process and each measurement has meaningful cost. Good uses
include microscope focus optimization, adaptive scanning of a sample, searching for
a region with a target property, tuning instrument parameters, or optimizing a
non-convex response over several controllable variables.

Do not default to BO for a simple deterministic one-dimensional problem unless the
user asks for it or measurements are expensive enough to justify the overhead. For
simple cases, suggest a direct scan, line search, grid search, or local optimizer if
that would be clearer and cheaper.

## First Understand the Problem

Before designing or launching BO, identify the problem type and the constraints.
Keep asking the user until the required information is clear.

Required information:

- Objective: what scalar quantity should be maximized, minimized, mapped, or
  balanced. If there are multiple objectives, ask how tradeoffs should be handled.
- Problem type: optimization, adaptive sampling, active learning, contour finding,
  threshold discovery, or constrained optimization.
- Search variables: names, units, continuous/discrete/categorical type, valid
  ranges, forbidden regions, and any safety limits.
- Measurement procedure: exactly how a suggested point is evaluated and how the
  response value is returned.
- Measurement cost and budget: expected time per measurement, maximum number of
  measurements, and whether batching is possible.
- Noise and repeatability: expected noise level, heteroscedasticity, drift, outlier
  behavior, and whether repeated measurements are allowed.
- Prior knowledge: known smoothness, periodicity, anisotropy, monotonicity,
  physically plausible length scales, important regions, or historical data.
- Stopping rule: measurement budget, wall-clock time, convergence criterion,
  posterior uncertainty threshold, expected improvement threshold, or user review
  checkpoint.

If the task is actually adaptive sampling, do not optimize blindly. Adaptive
sampling usually aims to learn important sample structure under a point budget,
not merely to find the highest observed value.

## Choose the BO Design

Explain the design choices to the user before launching the loop. At minimum,
state the model, kernel, noise treatment, acquisition function, startup design, and
stopping rule.

Model and kernel:

- Start with a Gaussian process surrogate unless there is a strong reason not to.
- Use a Matern 5/2 kernel with automatic relevance determination as a robust
  default for physical measurements. It assumes smooth but not perfectly analytic
  responses and can learn separate length scales per variable.
- Use an RBF kernel only when the response is expected to be very smooth.
- Add periodic, linear, additive, or custom kernels only when the user's prior
  knowledge clearly supports them.
- Normalize continuous inputs to `[0, 1]` internally and standardize outputs before
  fitting. Report suggested points in the user's original units.

Noise:

- Ask for expected noise. If a known standard deviation is available, use it as
  fixed observation noise or measurement-specific variance.
- If noise is unknown, let the GP infer it, but watch for pathologies such as
  unrealistically tiny noise, excessive length scales, or over-smoothed data.
- For drift or heteroscedastic noise, consider repeated measurements, time as an
  input, local noise estimates, or a more explicit model.

Acquisition:

- For optimization, use expected improvement (EI) or a numerically stable EI
  variant as the default. EI usually needs less manual tuning than UCB.
- Use UCB or Thompson sampling when explicit exploration is desired, but explain
  and tune the exploration parameter.
- For adaptive sampling, use uncertainty-driven acquisition, integrated variance
  reduction, entropy-based methods, or a task-specific acquisition. If high-value
  regions are more important, combine uncertainty with a posterior mean or
  probability-of-interest term.
- For constrained problems, model feasibility separately and acquire points with
  high utility and acceptable feasibility probability.

Startup data:

- Use a space-filling design such as Sobol or Latin hypercube points before relying
  on the acquisition function.
- As a rough default, collect at least `max(5, 2 * d)` initial observations for
  `d` continuous dimensions. Increase this when the space is noisy, highly
  anisotropic, or constrained.
- Include any trustworthy historical observations as initial data after checking
  units, bounds, and objective sign.

Stopping:

- Always define a stopping criterion before launching.
- Common criteria are a fixed measurement budget, no meaningful EI for several
  iterations, stable incumbent value, posterior uncertainty below a threshold in
  the region of interest, or an explicit user review after each batch.

## Communicate Before Execution

Before running the first suggestion, tell the user:

- The objective and whether it is being maximized or minimized.
- The search variables and bounds.
- The number and source of initial measurements.
- The GP model, kernel, and noise assumption.
- The acquisition function and why it matches the task.
- How observations will be logged and how the script will be reused.
- The stopping rule and any safety checks.

If the user asks for a fully automated measurement loop, also state how failed
measurements, out-of-bounds suggestions, instrument limits, and user interrupts
will be handled.

## Implementation Procedure

1. Create or reuse one Python BO script for the task. Do not rewrite the whole BO
   implementation after every observation. Evolve the same script only when a
   design change is needed.
2. Store observations in a durable state file such as JSON, CSV, SQLite, or a
   domain-specific log. Include variable names, units when relevant, response,
   noise estimate, timestamp if useful, and metadata for failed or repeated
   measurements.
3. Expose a command-line interface with at least:
   - `update`: append new observations and optional measurement noise.
   - `suggest`: fit/update the GP and return the next point or batch of points.
4. Add plotting or diagnostics early. Visualize observations, the incumbent
   optimum, posterior mean, posterior standard deviation, and acquisition values
   whenever dimensionality allows. For higher-dimensional problems, use slices,
   pair plots, marginal effects, or projections.
5. Fit GP hyperparameters by maximizing the marginal likelihood after each update
   unless the data volume or timing requires a less frequent schedule.
6. After every measurement or batch, inspect whether the observation is plausible,
   update the state, regenerate diagnostics, and decide whether the stopping rule
   has been met.

Use a high-level BO library unless the user explicitly requests an implementation
from scratch. BoTorch is recommended. If BoTorch is unavailable, do not silently
change the project environment. Ask before installing persistent dependencies, or
use a temporary command such as `uv run --with botorch --with gpytorch ...` when
that is acceptable for the task.

## Example Script

An example BoTorch CLI is available at:

`packages/eaa-core/src/eaa_core/skills/bayesian-optimization/examples/botorch_ei_cli.py`

It demonstrates a reusable state-file workflow:

```bash
uv run python packages/eaa-core/src/eaa_core/skills/bayesian-optimization/examples/botorch_ei_cli.py init \
  --state bo_state.json \
  --var x 0 1 \
  --var y 0 1 \
  --min-initial 6

uv run python packages/eaa-core/src/eaa_core/skills/bayesian-optimization/examples/botorch_ei_cli.py suggest \
  --state bo_state.json

uv run python packages/eaa-core/src/eaa_core/skills/bayesian-optimization/examples/botorch_ei_cli.py update \
  --state bo_state.json \
  --x 0.25 0.75 \
  --y 1.42 \
  --noise-std 0.05
```

Use this script as a starting point, then adapt only the objective-specific parts:
bounds, variable names, measurement interface, acquisition choice, constraints,
and diagnostics. Keep the state file as the source of truth during the run.

## Practical Checks

- Confirm the suggested point is inside the physical and safety bounds before
  measuring.
- Track the incumbent best observation and the model-predicted best point.
- Watch for repeated suggestions, boundary fixation, unrealistic length scales,
  or posterior uncertainty collapse. These often indicate bad bounds, incorrect
  noise assumptions, duplicated data, or a misspecified kernel.
- If observations contradict the GP assumptions, pause and revise the model rather
  than continuing blindly.
- For instrument or sample scans, keep raw data and derived scalar objectives
  linked so the objective can be audited later.

