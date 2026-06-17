"""Small stateful BoTorch example using a Matern GP and EI acquisition.

This script is intentionally generic. It keeps a JSON state file, appends
observations with ``update``, and proposes points with ``suggest``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from botorch.acquisition.analytic import ExpectedImprovement, LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood


DTYPE = torch.double


def load_state(path: Path) -> dict[str, Any]:
    """Load a BO state JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: dict[str, Any]) -> None:
    """Write a BO state JSON file."""
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def bounds_tensor(state: dict[str, Any]) -> torch.Tensor:
    """Return physical bounds as a ``2 x d`` tensor."""
    bounds = [[var["lower"], var["upper"]] for var in state["variables"]]
    return torch.tensor(bounds, dtype=DTYPE).transpose(0, 1)


def normalize_x(x: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
    """Map physical coordinates to the unit cube."""
    return (x - bounds[0]) / (bounds[1] - bounds[0])


def unnormalize_x(x: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
    """Map unit-cube coordinates to physical coordinates."""
    return bounds[0] + x * (bounds[1] - bounds[0])


def observation_tensors(
    state: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Build BoTorch training tensors from state observations."""
    observations = state.get("observations", [])
    train_x = torch.tensor([obs["x"] for obs in observations], dtype=DTYPE)
    train_y = torch.tensor([[obs["y"]] for obs in observations], dtype=DTYPE)
    noise_values = [obs.get("noise_std") for obs in observations]
    if all(value is not None for value in noise_values):
        train_yvar = torch.tensor(
            [[float(value) ** 2] for value in noise_values],
            dtype=DTYPE,
        )
    else:
        train_yvar = None
    return train_x, train_y, train_yvar


def fit_model(state: dict[str, Any]) -> SingleTaskGP:
    """Fit a Matern 5/2 GP with ARD length scales."""
    physical_bounds = bounds_tensor(state)
    train_x, train_y, train_yvar = observation_tensors(state)
    normalized_x = normalize_x(train_x, physical_bounds)
    dimension = normalized_x.shape[-1]
    covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=dimension))
    model = SingleTaskGP(
        normalized_x,
        train_y,
        train_Yvar=train_yvar,
        covar_module=covar_module,
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


def sobol_suggestion(state: dict[str, Any], n: int) -> list[float]:
    """Return the next deterministic Sobol startup point in physical units."""
    dimension = len(state["variables"])
    seed = int(state.get("seed", 0))
    index = len(state.get("observations", [])) + n
    engine = torch.quasirandom.SobolEngine(dimension=dimension, scramble=True, seed=seed)
    unit_points = engine.draw(index + 1)
    physical_bounds = bounds_tensor(state)
    point = unnormalize_x(unit_points[index].to(dtype=DTYPE), physical_bounds)
    return point.tolist()


def suggest_point(state: dict[str, Any], acq: str) -> dict[str, Any]:
    """Suggest one next point using Sobol startup or EI."""
    observations = state.get("observations", [])
    default_min_initial = max(5, 2 * len(state["variables"]))
    min_initial = int(state.get("min_initial") or default_min_initial)
    if len(observations) < min_initial:
        return {
            "method": "sobol_initial",
            "x": sobol_suggestion(state, 0),
            "reason": f"{len(observations)} observations available; need {min_initial}",
        }

    model = fit_model(state)
    train_y = observation_tensors(state)[1]
    maximize = bool(state.get("maximize", True))
    best_value = train_y.max() if maximize else train_y.min()

    if acq == "ei":
        acquisition = ExpectedImprovement(
            model=model,
            best_f=best_value,
            maximize=maximize,
        )
    else:
        acquisition = LogExpectedImprovement(
            model=model,
            best_f=best_value,
            maximize=maximize,
        )

    dimension = len(state["variables"])
    unit_bounds = torch.stack([torch.zeros(dimension), torch.ones(dimension)]).to(DTYPE)
    candidate, value = optimize_acqf(
        acq_function=acquisition,
        bounds=unit_bounds,
        q=1,
        num_restarts=int(state.get("num_restarts", 10)),
        raw_samples=int(state.get("raw_samples", 256)),
    )
    physical_x = unnormalize_x(candidate.squeeze(0), bounds_tensor(state))
    return {
        "method": acq,
        "x": physical_x.tolist(),
        "acquisition_value": float(value.item()),
    }


def init_command(args: argparse.Namespace) -> None:
    """Initialize a new state file."""
    variables = [
        {"name": name, "lower": float(lower), "upper": float(upper)}
        for name, lower, upper in args.var
    ]
    for variable in variables:
        if variable["lower"] >= variable["upper"]:
            raise ValueError(f"Invalid bounds for {variable['name']}.")
    state = {
        "variables": variables,
        "maximize": not args.minimize,
        "min_initial": args.min_initial or max(5, 2 * len(variables)),
        "seed": args.seed,
        "observations": [],
    }
    save_state(args.state, state)
    print(json.dumps(state, indent=2))


def update_command(args: argparse.Namespace) -> None:
    """Append one observation to the state file."""
    state = load_state(args.state)
    dimension = len(state["variables"])
    if len(args.x) != dimension:
        raise ValueError(f"Expected {dimension} x values, received {len(args.x)}.")
    lower = bounds_tensor(state)[0]
    upper = bounds_tensor(state)[1]
    x = torch.tensor(args.x, dtype=DTYPE)
    if torch.any(x < lower) or torch.any(x > upper):
        raise ValueError("Observation is outside configured bounds.")
    observation = {"x": [float(value) for value in args.x], "y": float(args.y)}
    if args.noise_std is not None:
        observation["noise_std"] = float(args.noise_std)
    if args.metadata:
        observation["metadata"] = json.loads(args.metadata)
    state.setdefault("observations", []).append(observation)
    save_state(args.state, state)
    print(json.dumps(observation, indent=2))


def suggest_command(args: argparse.Namespace) -> None:
    """Print the next suggested point as JSON."""
    state = load_state(args.state)
    suggestion = suggest_point(state, args.acq)
    print(json.dumps(suggestion, indent=2))


def plot_command(args: argparse.Namespace) -> None:
    """Plot observations for one- or two-dimensional states."""
    import matplotlib.pyplot as plt

    state = load_state(args.state)
    train_x, train_y, _ = observation_tensors(state)
    dimension = train_x.shape[-1]
    if dimension == 1:
        plt.scatter(train_x[:, 0].numpy(), train_y[:, 0].numpy(), label="observed")
        plt.xlabel(state["variables"][0]["name"])
        plt.ylabel("objective")
    elif dimension == 2:
        scatter = plt.scatter(
            train_x[:, 0].numpy(),
            train_x[:, 1].numpy(),
            c=train_y[:, 0].numpy(),
            cmap="viridis",
        )
        plt.xlabel(state["variables"][0]["name"])
        plt.ylabel(state["variables"][1]["name"])
        plt.colorbar(scatter, label="objective")
    else:
        raise ValueError("Plotting is implemented only for 1D or 2D states.")
    plt.tight_layout()
    if args.output:
        plt.savefig(args.output, dpi=150)
    else:
        plt.show()


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(required=True)

    init_parser = subparsers.add_parser("init", help="Create a BO state file.")
    init_parser.add_argument("--state", type=Path, required=True)
    init_parser.add_argument(
        "--var",
        nargs=3,
        action="append",
        metavar=("NAME", "LOWER", "UPPER"),
        required=True,
        help="Add one continuous variable and its physical bounds.",
    )
    init_parser.add_argument("--minimize", action="store_true")
    init_parser.add_argument("--min-initial", type=int, default=None)
    init_parser.add_argument("--seed", type=int, default=0)
    init_parser.set_defaults(func=init_command)

    update_parser = subparsers.add_parser("update", help="Append one observation.")
    update_parser.add_argument("--state", type=Path, required=True)
    update_parser.add_argument("--x", nargs="+", type=float, required=True)
    update_parser.add_argument("--y", type=float, required=True)
    update_parser.add_argument("--noise-std", type=float)
    update_parser.add_argument("--metadata", help="Optional JSON metadata object.")
    update_parser.set_defaults(func=update_command)

    suggest_parser = subparsers.add_parser("suggest", help="Suggest the next point.")
    suggest_parser.add_argument("--state", type=Path, required=True)
    suggest_parser.add_argument(
        "--acq",
        choices=["logei", "ei"],
        default="logei",
        help="Use log-EI by default for numerical stability.",
    )
    suggest_parser.set_defaults(func=suggest_command)

    plot_parser = subparsers.add_parser("plot", help="Plot 1D or 2D observations.")
    plot_parser.add_argument("--state", type=Path, required=True)
    plot_parser.add_argument("--output", type=Path)
    plot_parser.set_defaults(func=plot_command)

    return parser


def main() -> None:
    """Run the CLI."""
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
