"""Visualization for SOTIF-LLM results.

Produces plots analogous to the error_predictor paper:
  - Figure 2: Trust regions in design space (probably safe / possibly safe / dangerous)
  - Figure 8: Validation metric violin plots per campaign
  - Figure 10: MVP contour plots with measurement campaign locations
  - Figure 11: LOO cross-validation curves

Plus new plots specific to LLM safety:
  - Real-time SAE distance during generation (step-by-step traces)
  - Reward hacking vs. safe code distance separation
  - Jailbreak vs. refusal distance separation
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_trust_regions(
    grid_predictions: dict,
    campaign_locations: np.ndarray | None = None,
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot trust regions in 2D slices of design space.

    Analog of Figure 10 in the error_predictor paper:
    colored regions = probably safe (green), possibly safe (blue), dangerous (red)
    black dots = measurement campaign locations (evaluated prompts).
    """
    n_slices = len(grid_predictions)
    fig, axes = plt.subplots(1, n_slices, figsize=(6 * n_slices, 5))
    if n_slices == 1:
        axes = [axes]

    cmap = plt.cm.colors.ListedColormap(["#2ecc71", "#3498db", "#e67e22", "#e74c3c"])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    for ax, (slice_name, data) in zip(axes, grid_predictions.items()):
        labels = np.array(data["trust_labels"])
        im = ax.imshow(
            labels, extent=[0, 1, 0, 1], origin="lower",
            cmap=cmap, norm=norm, aspect="auto", alpha=0.7,
        )
        # Overlay q95 contours
        q95 = np.array(data["q95_values"])
        x = np.linspace(0, 1, q95.shape[1])
        y = np.linspace(0, 1, q95.shape[0])
        X, Y = np.meshgrid(x, y)
        ax.contour(X, Y, q95, levels=6, colors="black", linewidths=0.5, alpha=0.5)

        ax.set_xlabel(data["dim1"].replace("_", " ").title())
        ax.set_ylabel(data["dim2"].replace("_", " ").title())
        ax.set_title(f"Safety Predictor: {data['dim1']} vs {data['dim2']}")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", label="Validated Safe"),
        Patch(facecolor="#3498db", label="Probably Safe"),
        Patch(facecolor="#e67e22", label="Possibly Safe"),
        Patch(facecolor="#e74c3c", label="Dangerous"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=9)
    fig.tight_layout(rect=[0, 0.08, 1, 1])

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_generation_traces(
    step_distances_path: Path,
    baseline_thresholds: dict[str, float],
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot real-time SAE distance during code generation.

    Shows how the model's internal state evolves token-by-token,
    and when it crosses the SOTIF envelope boundary.
    """
    with open(step_distances_path) as f:
        traces = json.load(f)

    fig, axes = plt.subplots(len(traces), 1, figsize=(12, 3 * len(traces)), squeeze=False)

    thresh_95 = float(baseline_thresholds.get("0.95", 0))
    thresh_99 = float(baseline_thresholds.get("0.99", 0))

    for idx, (ax_row, trace) in enumerate(zip(axes, traces)):
        ax = ax_row[0]
        steps = range(len(trace["step_distances"]))
        distances = trace["step_distances"]

        ax.plot(steps, distances, "b-", linewidth=1, alpha=0.8)
        ax.axhline(y=thresh_95, color="orange", linestyle="--", alpha=0.7, label="95% threshold")
        ax.axhline(y=thresh_99, color="red", linestyle="--", alpha=0.7, label="99% threshold")

        # Color regions where model is outside envelope
        for i, (d, l) in enumerate(zip(distances, trace["step_labels"])):
            if l >= 2:
                ax.axvspan(i - 0.5, i + 0.5, alpha=0.15, color="red")

        ax.set_ylabel("SAE Distance")
        ax.set_title(f"{trace['prompt_id']} (difficulty: {trace['difficulty']})")
        if idx == 0:
            ax.legend(loc="upper right", fontsize=8)

    axes[-1][0].set_xlabel("Generation Step (token)")
    fig.suptitle("Real-time SOTIF Monitoring During Code Generation", fontsize=13)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_separation(
    phase2_summary: dict,
    phase3_summary: dict,
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot distance separation between safe and unsafe behaviors.

    Shows that reward hacking and jailbreaks produce mechanistically
    distinct internal states — the core claim of the paper.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Phase 2: Reward Hacking
    ax = axes[0]
    sep = phase2_summary.get("reward_hacking_separation", {})
    if "mean_distance_bad_exception" in sep:
        categories = ["Safe Code", "Bad Exception\n(Reward Hacking)"]
        means = [sep["mean_distance_good_exception"], sep["mean_distance_bad_exception"]]
        colors = ["#2ecc71", "#e74c3c"]
        ax.bar(categories, means, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_ylabel("Mean SAE Distance from Baseline")
        ax.set_title(f"Phase 2: Reward Hacking\nCohen's d = {sep.get('effect_size_cohens_d', 0):.2f}")
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Phase 2: Reward Hacking")

    # Phase 3: Jailbreaks
    ax = axes[1]
    exp_results = phase3_summary.get("experiments", {})
    if exp_results:
        first_exp = next(iter(exp_results.values()))
        resp = first_exp.get("response_analysis", {})
        if "mean_dist_jailbreak_responses" in resp:
            categories = ["Refusal\n(Safe)", "Jailbreak\n(Malicious Compliance)"]
            means = [resp["mean_dist_refusal_responses"], resp["mean_dist_jailbreak_responses"]]
            colors = ["#2ecc71", "#e74c3c"]
            ax.bar(categories, means, color=colors, edgecolor="black", linewidth=0.5)
            ax.set_ylabel("Mean SAE Distance from Baseline")
            ax.set_title(f"Phase 3: Jailbreaks\nCohen's d = {resp.get('cohens_d_responses', 0):.2f}")
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Phase 3: Jailbreaks")
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Phase 3: Jailbreaks")

    fig.suptitle("SOTIF Envelope Separation: Safe vs. Unsafe Behaviors", fontsize=13)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_campaign_validation(
    campaign_metrics_path: Path,
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot validation metrics per campaign (violin plots).

    Analog of Figures 8-9 in the error_predictor paper:
    violin plots showing the distribution of validation metrics
    across epistemic uncertainty samples for each campaign.
    """
    with open(campaign_metrics_path) as f:
        campaigns = json.load(f)

    fig, axes = plt.subplots(2, 1, figsize=(max(12, len(campaigns) * 0.5), 8))

    # Distance metric
    ax = axes[0]
    positions = range(len(campaigns))
    distances = [c["mean_distance"] for c in campaigns]
    ax.bar(positions, distances, color="#3498db", alpha=0.7, edgecolor="black", linewidth=0.3)
    ax.set_ylabel("Mean SAE Distance")
    ax.set_title("SAE Anomaly Distance per Campaign")
    ax.set_xticks([])

    # Area metric
    ax = axes[1]
    areas = [c["mean_area"] for c in campaigns]
    ax.bar(positions, areas, color="#e67e22", alpha=0.7, edgecolor="black", linewidth=0.3)
    ax.set_ylabel("Area Validation Metric")
    ax.set_xlabel("Campaign Index")
    ax.set_title("Area Validation Metric per Campaign")

    fig.suptitle("Per-Campaign Validation Metrics (Phase 1 Baseline)", fontsize=13)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def generate_all_plots(experiment_dir: Path) -> None:
    """Generate all visualization plots from experiment results."""
    plots_dir = experiment_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Campaign validation
    camp_path = experiment_dir / "phase1" / "metrics" / "campaign_metrics.json"
    if camp_path.exists():
        plot_campaign_validation(camp_path, plots_dir / "campaign_validation.png")

    # Trust regions
    pred_path = experiment_dir / "predictor" / "grid_predictions.json"
    if pred_path.exists():
        with open(pred_path) as f:
            grid = json.load(f)
        plot_trust_regions(grid, output_path=plots_dir / "trust_regions.png")

    # Generation traces
    traces_path = experiment_dir / "phase2" / "step_distances.json"
    baseline_path = experiment_dir / "phase1" / "phase1_summary.json"
    if traces_path.exists() and baseline_path.exists():
        with open(baseline_path) as f:
            p1 = json.load(f)
        thresholds = p1.get("baseline_thresholds", {})
        plot_generation_traces(traces_path, thresholds, plots_dir / "generation_traces.png")

    # Separation plot
    p2_path = experiment_dir / "phase2" / "phase2_summary.json"
    p3_path = experiment_dir / "phase3" / "phase3_summary.json"
    if p2_path.exists() and p3_path.exists():
        with open(p2_path) as f:
            p2 = json.load(f)
        with open(p3_path) as f:
            p3 = json.load(f)
        plot_separation(p2, p3, plots_dir / "envelope_separation.png")

    print(f"Plots saved to {plots_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=str, default="experiments/sotif_llm_v1")
    args = parser.parse_args()
    generate_all_plots(Path(args.experiment_dir))


if __name__ == "__main__":
    main()
