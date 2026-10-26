"""
assignment1.py
State Estimation and Filtering - Assignment 1

Run:
    python assignment1.py

Requirements:
    numpy
    scipy
    matplotlib

Important:
    Check nonlinear_rhs() and nonlinear_jacobian() against ass1_framework.py
    supplied with the exercise. The PDF text available here does not render
    the complete nonlinear equation unambiguously.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp


# =============================================================================
# Configuration
# =============================================================================

OUTPUT_DIR = Path("figures")
FIGURE_FORMAT = "pdf"

T_FINAL = 100.0
T_REFERENCE = np.linspace(0.0, T_FINAL, 10001)  # smooth numerical reference
TAYLOR_STEP = 1.0
EULER_STEPS = (0.1, 1.0, 5.0)

N_ENSEMBLE = 1000
RANDOM_SEED = 2026

# Initial condition and initial covariance required in Assignment 1.
X0 = np.array([10.0, 0.0, 0.0])
P0 = np.diag([0.1, 0.2, 0.1])

# Linear-system parameters.
K1 = 0.1
K2 = 0.2
K3 = 0.01

# Nonlinear-system parameters.
K1_F = 1.0
K2_F = 2.0
K1_R = 4.0
K2_R = 3.0

# Colours for the three states.
STATE_COLORS = ["tab:blue", "tab:orange", "tab:green"]


# =============================================================================
# System models
# =============================================================================

# For the reactions S1 -> S2 -> S3 -> S1:
#
# x1_dot = -k1*x1 + k3*x3
# x2_dot =  k1*x1 - k2*x2
# x3_dot =            k2*x2 - k3*x3
#
# Hence:
A_LINEAR = np.array(
    [
        [-K1, 0.0, K3],
        [K1, -K2, 0.0],
        [0.0, K2, -K3],
    ],
    dtype=float,
)


def linear_rhs(t: float, x: np.ndarray) -> np.ndarray:
    """Linear right-hand side: x_dot = A x."""
    del t
    return A_LINEAR @ x


def linear_jacobian(x: np.ndarray) -> np.ndarray:
    """Jacobian of the linear system."""
    del x
    return A_LINEAR.copy()


# -------------------------------------------------------------------------
# Nonlinear model
# -------------------------------------------------------------------------
# IMPORTANT:
# Compare this exact implementation with ass1_framework.py. If the
# framework contains a different Michaelis-Menten convention, modify
# only nonlinear_rhs() and nonlinear_jacobian().
#
# Implemented rate:
#     q(s; k) = s / (1 + s/k) = k*s/(k+s)
#
# Reversible reactions:
#     r1 = q(x1; k1f) - q(x2; k1r)
#     r2 = q(x2; k2f) - q(x3; k2r)
#
# State equations:
#     x1_dot = -r1
#     x2_dot =  r1 - r2
#     x3_dot =  r2
# -------------------------------------------------------------------------


def mm_rate(s: float, k: float) -> float:
    """Michaelis-Menten-type rate q(s;k) = k*s/(k+s)."""
    return (k * s) / (k + s)


def mm_rate_derivative(s: float, k: float) -> float:
    """
    Derivative of q(s;k) = k*s/(k+s):
        dq/ds = k^2/(k+s)^2
    """
    return (k**2) / (k + s) ** 2


def nonlinear_rhs(t: float, x: np.ndarray) -> np.ndarray:
    """Nonlinear reversible enzyme-kinetics system."""
    del t

    x1, x2, x3 = x

    r1 = mm_rate(x1, K1_F) - mm_rate(x2, K1_R)
    r2 = mm_rate(x2, K2_F) - mm_rate(x3, K2_R)

    return np.array(
        [
            -r1,
            r1 - r2,
            r2,
        ],
        dtype=float,
    )


def nonlinear_jacobian(x: np.ndarray) -> np.ndarray:
    """
    Analytic Jacobian of nonlinear_rhs().

    Let:
        a = d q(x1;k1f) / dx1
        b = d q(x2;k1r) / dx2
        c = d q(x2;k2f) / dx2
        d = d q(x3;k2r) / dx3

    Then:
        J = [ -a,       b,       0 ]
            [  a,  -b - c,       d ]
            [  0,       c,      -d ]
    """
    x1, x2, x3 = x

    a = mm_rate_derivative(x1, K1_F)
    b = mm_rate_derivative(x2, K1_R)
    c = mm_rate_derivative(x2, K2_F)
    d = mm_rate_derivative(x3, K2_R)

    return np.array(
        [
            [-a, b, 0.0],
            [a, -b - c, d],
            [0.0, c, -d],
        ],
        dtype=float,
    )


# =============================================================================
# Numerical methods
# =============================================================================

Rhs = Callable[[float, np.ndarray], np.ndarray]
Jacobian = Callable[[np.ndarray], np.ndarray]


def make_time_grid(dt: float) -> np.ndarray:
    """Create a grid that includes both 0 and T_FINAL exactly."""
    number_of_steps = int(round(T_FINAL / dt))
    if not np.isclose(number_of_steps * dt, T_FINAL):
        raise ValueError(f"T_FINAL={T_FINAL} is not divisible by dt={dt}.")
    return np.linspace(0.0, T_FINAL, number_of_steps + 1)


def numerical_solution(
    rhs: Rhs,
    x_initial: np.ndarray,
    time_grid: np.ndarray,
) -> np.ndarray:
    """
    Accurate numerical reference solution using scipy.integrate.solve_ivp.
    DOP853 is used with strict tolerances so that it can serve as reference.
    """
    result = solve_ivp(
        fun=rhs,
        t_span=(float(time_grid[0]), float(time_grid[-1])),
        y0=np.asarray(x_initial, dtype=float),
        t_eval=time_grid,
        method="DOP853",
        rtol=1e-10,
        atol=1e-12,
    )

    if not result.success:
        raise RuntimeError(f"Numerical integration failed: {result.message}")

    return result.y.T


def explicit_euler(
    rhs: Rhs,
    x_initial: np.ndarray,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Explicit forward Euler method:
        x_{k+1} = x_k + dt f(x_k).
    """
    time_grid = make_time_grid(dt)
    states = np.empty((len(time_grid), len(x_initial)), dtype=float)
    states[0] = x_initial

    for k in range(len(time_grid) - 1):
        states[k + 1] = states[k] + dt * rhs(time_grid[k], states[k])

    return time_grid, states


def propagate_taylor_covariance(
    rhs: Rhs,
    jacobian: Jacobian,
    x_initial: np.ndarray,
    p_initial: np.ndarray,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Propagate mean and covariance using first-order Taylor linearisation.

    Nominal state:
        xbar_{k+1} = xbar_k + dt f(xbar_k)

    Linearised discrete transition:
        F_k = I + dt J_f(xbar_k)

    Covariance:
        P_{k+1} = F_k P_k F_k^T
    """
    time_grid = make_time_grid(dt)
    n_states = len(x_initial)

    mean_states = np.empty((len(time_grid), n_states), dtype=float)
    covariances = np.empty((len(time_grid), n_states, n_states), dtype=float)

    mean_states[0] = x_initial
    covariances[0] = p_initial

    identity = np.eye(n_states)

    for k in range(len(time_grid) - 1):
        x_nominal = mean_states[k]
        jac = jacobian(x_nominal)
        transition = identity + dt * jac

        mean_states[k + 1] = x_nominal + dt * rhs(time_grid[k], x_nominal)
        covariances[k + 1] = transition @ covariances[k] @ transition.T

        # Remove numerical asymmetry caused by floating-point arithmetic.
        covariances[k + 1] = 0.5 * (
            covariances[k + 1] + covariances[k + 1].T
        )

    return time_grid, mean_states, covariances


# =============================================================================
# Sampling and Monte Carlo simulation
# =============================================================================


def sample_initial_states(
    rng: np.random.Generator,
    distribution: str,
    n_samples: int,
) -> np.ndarray:
    """
    Draw samples with mean X0 and marginal covariance diag(P0).

    Gaussian:
        X ~ N(X0, P0)

    Laplace:
        Var(Laplace(0,b)) = 2 b^2
        -> b_i = sqrt(P0_ii / 2)

    Uniform:
        Var(U(-a,a)) = a^2 / 3
        -> a_i = sqrt(3 P0_ii)

    P0 is diagonal, so independent component-wise samples are appropriate.
    """
    variances = np.diag(P0)
    standard_deviations = np.sqrt(variances)

    if distribution == "Gaussian":
        return rng.multivariate_normal(mean=X0, cov=P0, size=n_samples)

    if distribution == "Laplace":
        scales = standard_deviations / np.sqrt(2.0)
        noise = rng.laplace(
            loc=0.0,
            scale=scales,
            size=(n_samples, len(X0)),
        )
        return X0 + noise

    if distribution == "Uniform":
        half_widths = np.sqrt(3.0) * standard_deviations
        noise = rng.uniform(
            low=-half_widths,
            high=half_widths,
            size=(n_samples, len(X0)),
        )
        return X0 + noise

    raise ValueError(f"Unknown distribution: {distribution}")


def simulate_ensemble(
    rhs: Rhs,
    initial_states: np.ndarray,
    time_grid: np.ndarray,
    system_name: str,
    distribution_name: str,
) -> np.ndarray:
    """
    Simulate every Monte-Carlo trajectory using solve_ivp.

    Output shape:
        (number of samples, number of time points, number of states)
    """
    n_samples = initial_states.shape[0]
    n_time_points = len(time_grid)
    n_states = initial_states.shape[1]

    trajectories = np.empty((n_samples, n_time_points, n_states), dtype=float)

    for sample_index, x_initial in enumerate(initial_states):
        if (sample_index + 1) % 100 == 0 or sample_index == 0:
            print(
                f"  {system_name:9s} | {distribution_name:8s} | "
                f"trajectory {sample_index + 1:4d}/{n_samples}"
            )

        trajectories[sample_index] = numerical_solution(
            rhs=rhs,
            x_initial=x_initial,
            time_grid=time_grid,
        )

    return trajectories


# =============================================================================
# Plotting utilities
# =============================================================================


def save_figure(fig: plt.Figure, filename: str) -> None:
    """Save and close a Matplotlib figure."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUTPUT_DIR / filename,
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_task1_euler_comparison(
    system_name: str,
    rhs: Rhs,
) -> List[Dict[str, float]]:
    """
    Create Task 1 Euler/reference plots and return maximum-error rows.
    """
    error_rows: List[Dict[str, float]] = []

    fig, axes = plt.subplots(
        1,
        len(EULER_STEPS),
        figsize=(17, 4.8),
        sharey=True,
    )

    if len(EULER_STEPS) == 1:
        axes = [axes]

    for axis, dt in zip(axes, EULER_STEPS):
        time_euler, x_euler = explicit_euler(rhs, X0, dt)
        x_reference = numerical_solution(rhs, X0, time_euler)

        absolute_error = np.abs(x_euler - x_reference)
        max_error_per_state = np.max(absolute_error, axis=0)

        error_rows.append(
            {
                "system": system_name,
                "Tp": dt,
                "max_abs_error_x1": max_error_per_state[0],
                "max_abs_error_x2": max_error_per_state[1],
                "max_abs_error_x3": max_error_per_state[2],
                "max_abs_error_all_states": float(np.max(absolute_error)),
            }
        )

        for state_index in range(3):
            axis.plot(
                time_euler,
                x_reference[:, state_index],
                color=STATE_COLORS[state_index],
                linewidth=2.0,
            )
            axis.plot(
                time_euler,
                x_euler[:, state_index],
                linestyle="None",
                marker="o",
                markersize=3.5,
                markevery=max(1, len(time_euler) // 20),
                markerfacecolor="none",
                markeredgewidth=1.0,
                color=STATE_COLORS[state_index],
            )

        axis.set_title(rf"$T_p={dt:g}\,\mathrm{{h}}$")
        axis.set_xlabel(r"$t\,[\mathrm{h}]$")
        axis.grid(alpha=0.3)

    axes[0].set_ylabel(r"concentration")

    legend_handles = [
        Line2D([0], [0], color=STATE_COLORS[0], lw=2, label=r"$x_1$"),
        Line2D([0], [0], color=STATE_COLORS[1], lw=2, label=r"$x_2$"),
        Line2D([0], [0], color=STATE_COLORS[2], lw=2, label=r"$x_3$"),
        Line2D([0], [0], color="black", lw=2, label="numerical reference"),
        Line2D(
            [0],
            [0],
            color="black",
            marker="o",
            markerfacecolor="none",
            linestyle="None",
            label="explicit Euler",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, -0.05),
    )
    fig.suptitle(
        f"Task 1: Explicit Euler versus numerical reference ({system_name} system)",
        y=1.02,
    )
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.95))

    save_figure(fig, f"task1_{system_name.lower()}.{FIGURE_FORMAT}")

    return error_rows


def plot_taylor_uncertainty(
    system_name: str,
    time_grid: np.ndarray,
    mean_states: np.ndarray,
    covariances: np.ndarray,
) -> None:
    """
    Plot nominal state and first-order uncertainty bands mean +/- one sigma.
    """
    standard_deviations = np.sqrt(
        np.maximum(np.diagonal(covariances, axis1=1, axis2=2), 0.0)
    )

    fig, axes = plt.subplots(3, 1, figsize=(9, 8.5), sharex=True)

    for state_index, axis in enumerate(axes):
        mean = mean_states[:, state_index]
        sigma = standard_deviations[:, state_index]

        axis.plot(
            time_grid,
            mean,
            color=STATE_COLORS[state_index],
            linewidth=2.0,
            label=rf"$\bar x_{state_index + 1}$",
        )
        axis.fill_between(
            time_grid,
            mean - sigma,
            mean + sigma,
            color=STATE_COLORS[state_index],
            alpha=0.25,
            label=r"$\bar x_i\pm 1\sigma_i$",
        )
        axis.set_ylabel(rf"$x_{state_index + 1}$")
        axis.grid(alpha=0.3)
        axis.legend(loc="best")

    axes[-1].set_xlabel(r"$t\,[\mathrm{h}]$")
    fig.suptitle(
        f"Task 2: First-order uncertainty propagation ({system_name} system)",
        y=0.995,
    )
    fig.tight_layout()

    save_figure(
        fig,
        f"task2_{system_name.lower()}_taylor_uncertainty.{FIGURE_FORMAT}",
    )


def plot_variance_comparison(
    system_name: str,
    time_grid: np.ndarray,
    covariances: np.ndarray,
    gaussian_trajectories: np.ndarray,
) -> None:
    """
    Compare Taylor covariance diagonal against Gaussian ensemble variance.
    """
    predicted_variance = np.diagonal(covariances, axis1=1, axis2=2)
    monte_carlo_variance = np.var(
        gaussian_trajectories,
        axis=0,
        ddof=1,
    )

    fig, axes = plt.subplots(3, 1, figsize=(9, 8.5), sharex=True)

    for state_index, axis in enumerate(axes):
        axis.plot(
            time_grid,
            predicted_variance[:, state_index],
            color="black",
            linewidth=2.0,
            label="Taylor approximation",
        )
        axis.plot(
            time_grid,
            monte_carlo_variance[:, state_index],
            color=STATE_COLORS[state_index],
            linestyle="--",
            linewidth=2.0,
            label="Monte-Carlo variance",
        )
        axis.set_ylabel(rf"$\mathrm{{Var}}(x_{state_index + 1})$")
        axis.grid(alpha=0.3)
        axis.legend(loc="best")

    axes[-1].set_xlabel(r"$t\,[\mathrm{h}]$")
    fig.suptitle(
        f"Task 2: Variance comparison for Gaussian initial values "
        f"({system_name} system)",
        y=0.995,
    )
    fig.tight_layout()

    save_figure(
        fig,
        f"task2_{system_name.lower()}_variance_comparison.{FIGURE_FORMAT}",
    )


def plot_x3_histograms(
    system_name: str,
    initial_samples: Dict[str, np.ndarray],
    ensemble_results: Dict[str, np.ndarray],
) -> None:
    """
    Plot x3 histograms at t=0 and t=100 for all three initial distributions.
    """
    distributions = ["Gaussian", "Laplace", "Uniform"]

    fig, axes = plt.subplots(
        len(distributions),
        2,
        figsize=(11, 10),
        sharex=False,
        sharey=False,
    )

    for row, distribution in enumerate(distributions):
        x3_at_zero = initial_samples[distribution][:, 2]
        x3_at_final = ensemble_results[distribution][:, -1, 2]

        combined = np.concatenate([x3_at_zero, x3_at_final])
        bin_edges = np.histogram_bin_edges(combined, bins=40)

        axes[row, 0].hist(
            x3_at_zero,
            bins=bin_edges,
            density=True,
            color=STATE_COLORS[2],
            alpha=0.75,
            edgecolor="black",
            linewidth=0.35,
        )
        axes[row, 1].hist(
            x3_at_final,
            bins=bin_edges,
            density=True,
            color=STATE_COLORS[2],
            alpha=0.75,
            edgecolor="black",
            linewidth=0.35,
        )

        axes[row, 0].set_ylabel(f"{distribution}\ndensity")
        axes[row, 0].grid(alpha=0.25)
        axes[row, 1].grid(alpha=0.25)

    axes[0, 0].set_title(r"$x_3$ at $t=0$")
    axes[0, 1].set_title(r"$x_3$ at $t=100\,\mathrm{h}$")

    for axis in axes[-1, :]:
        axis.set_xlabel(r"$x_3$")

    fig.suptitle(
        f"Task 2: Histograms of $x_3$ ({system_name} system)",
        y=0.995,
    )
    fig.tight_layout()

    save_figure(
        fig,
        f"hist_{system_name.lower()}_x3.{FIGURE_FORMAT}",
    )


# =============================================================================
# CSV output
# =============================================================================


def write_dict_rows(filename: str, rows: List[Dict[str, float]]) -> None:
    """Write a list of dictionaries to a CSV file."""
    if not rows:
        return

    path = OUTPUT_DIR / filename

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_variance_curves(
    system_name: str,
    time_grid: np.ndarray,
    covariances: np.ndarray,
    ensemble_results: Dict[str, np.ndarray],
) -> None:
    """Write predicted and sample variance curves to CSV."""
    predicted_variance = np.diagonal(covariances, axis1=1, axis2=2)

    sample_variances = {
        distribution: np.var(trajectories, axis=0, ddof=1)
        for distribution, trajectories in ensemble_results.items()
    }

    rows: List[Dict[str, float]] = []

    for time_index, time_value in enumerate(time_grid):
        row: Dict[str, float] = {
            "time": float(time_value),
            "taylor_var_x1": float(predicted_variance[time_index, 0]),
            "taylor_var_x2": float(predicted_variance[time_index, 1]),
            "taylor_var_x3": float(predicted_variance[time_index, 2]),
        }

        for distribution, variance in sample_variances.items():
            row[f"{distribution.lower()}_var_x1"] = float(
                variance[time_index, 0]
            )
            row[f"{distribution.lower()}_var_x2"] = float(
                variance[time_index, 1]
            )
            row[f"{distribution.lower()}_var_x3"] = float(
                variance[time_index, 2]
            )

        rows.append(row)

    write_dict_rows(f"variance_curves_{system_name.lower()}.csv", rows)


def write_task2_summary(
    taylor_results: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    initial_samples: Dict[str, np.ndarray],
    all_ensemble_results: Dict[str, Dict[str, np.ndarray]],
) -> None:
    """
    Write concise final-time statistics and variance-RMSE values to CSV.
    """
    rows: List[Dict[str, float]] = []

    for system_name, (_, _, covariances) in taylor_results.items():
        predicted_variance = np.diagonal(covariances, axis1=1, axis2=2)

        for distribution, trajectories in all_ensemble_results[
            system_name
        ].items():
            monte_carlo_mean = np.mean(trajectories, axis=0)
            monte_carlo_variance = np.var(trajectories, axis=0, ddof=1)

            variance_rmse = np.sqrt(
                np.mean((monte_carlo_variance - predicted_variance) ** 2, axis=0)
            )

            rows.append(
                {
                    "system": system_name,
                    "distribution": distribution,
                    "initial_sample_mean_x3": float(
                        np.mean(initial_samples[distribution][:, 2])
                    ),
                    "initial_sample_var_x3": float(
                        np.var(
                            initial_samples[distribution][:, 2],
                            ddof=1,
                        )
                    ),
                    "final_mc_mean_x3": float(monte_carlo_mean[-1, 2]),
                    "final_mc_var_x3": float(monte_carlo_variance[-1, 2]),
                    "final_taylor_var_x3": float(predicted_variance[-1, 2]),
                    "variance_rmse_x1": float(variance_rmse[0]),
                    "variance_rmse_x2": float(variance_rmse[1]),
                    "variance_rmse_x3": float(variance_rmse[2]),
                }
            )

    write_dict_rows("task2_summary.csv", rows)


# =============================================================================
# Main program
# =============================================================================


def main() -> None:
    """Run all Assignment 1 tasks."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    systems: Dict[str, Tuple[Rhs, Jacobian]] = {
        "linear": (linear_rhs, linear_jacobian),
        "nonlinear": (nonlinear_rhs, nonlinear_jacobian),
    }

    # Conservation checks:
    # For both models, x1 + x2 + x3 must be conserved.
    if not np.allclose(np.ones(3) @ A_LINEAR, np.zeros(3)):
        raise RuntimeError("The linear system does not conserve total mass.")

    test_state = np.array([3.0, 1.0, 2.0])
    if not np.isclose(np.sum(nonlinear_rhs(0.0, test_state)), 0.0):
        raise RuntimeError("The nonlinear system does not conserve total mass.")

    # -------------------------------------------------------------------------
    # Task 1: Numerical simulation and explicit Euler comparison.
    # -------------------------------------------------------------------------
    print("\nTask 1: Euler/reference comparisons")
    task1_error_rows: List[Dict[str, float]] = []

    for system_name, (rhs, _) in systems.items():
        print(f"  Creating Task 1 plot for the {system_name} system.")
        task1_error_rows.extend(
            plot_task1_euler_comparison(system_name, rhs)
        )

    write_dict_rows("task1_errors.csv", task1_error_rows)

    # -------------------------------------------------------------------------
    # Task 2: Taylor covariance propagation.
    # -------------------------------------------------------------------------
    print("\nTask 2: First-order covariance propagation")
    taylor_results: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for system_name, (rhs, jacobian) in systems.items():
        time_grid, mean_states, covariances = propagate_taylor_covariance(
            rhs=rhs,
            jacobian=jacobian,
            x_initial=X0,
            p_initial=P0,
            dt=TAYLOR_STEP,
        )

        taylor_results[system_name] = (
            time_grid,
            mean_states,
            covariances,
        )

        plot_taylor_uncertainty(
            system_name=system_name,
            time_grid=time_grid,
            mean_states=mean_states,
            covariances=covariances,
        )

    # -------------------------------------------------------------------------
    # Task 2: Initial ensembles.
    # -------------------------------------------------------------------------
    rng = np.random.default_rng(RANDOM_SEED)

    initial_samples = {
        "Gaussian": sample_initial_states(rng, "Gaussian", N_ENSEMBLE),
        "Laplace": sample_initial_states(rng, "Laplace", N_ENSEMBLE),
        "Uniform": sample_initial_states(rng, "Uniform", N_ENSEMBLE),
    }

    # -------------------------------------------------------------------------
    # Task 2: Monte-Carlo simulations for all distributions and systems.
    # -------------------------------------------------------------------------
    print("\nTask 2: Monte-Carlo ensemble simulations")
    all_ensemble_results: Dict[str, Dict[str, np.ndarray]] = {}

    for system_name, (rhs, _) in systems.items():
        time_grid, _, covariances = taylor_results[system_name]

        system_ensemble_results: Dict[str, np.ndarray] = {}

        for distribution_name, samples in initial_samples.items():
            system_ensemble_results[distribution_name] = simulate_ensemble(
                rhs=rhs,
                initial_states=samples,
                time_grid=time_grid,
                system_name=system_name,
                distribution_name=distribution_name,
            )

        all_ensemble_results[system_name] = system_ensemble_results

        # Gaussian variance comparison required in the first Monte-Carlo task.
        plot_variance_comparison(
            system_name=system_name,
            time_grid=time_grid,
            covariances=covariances,
            gaussian_trajectories=system_ensemble_results["Gaussian"],
        )

        # Histograms required for Gaussian, Laplace and Uniform initial values.
        plot_x3_histograms(
            system_name=system_name,
            initial_samples=initial_samples,
            ensemble_results=system_ensemble_results,
        )

        # CSV containing complete variance curves.
        write_variance_curves(
            system_name=system_name,
            time_grid=time_grid,
            covariances=covariances,
            ensemble_results=system_ensemble_results,
        )

    write_task2_summary(
        taylor_results=taylor_results,
        initial_samples=initial_samples,
        all_ensemble_results=all_ensemble_results,
    )

    # -------------------------------------------------------------------------
    # Console summary.
    # -------------------------------------------------------------------------
    print("\nFinished.")
    print(f"All figures and CSV files were written to: {OUTPUT_DIR.resolve()}")
    print("\nThe most useful numerical files for the report are:")
    print("  figures/task1_errors.csv")
    print("  figures/task2_summary.csv")
    print("  figures/variance_curves_linear.csv")
    print("  figures/variance_curves_nonlinear.csv")


if __name__ == "__main__":
    main()