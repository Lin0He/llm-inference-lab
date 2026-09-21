import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_summaries(
    path: Path,
) -> list[dict]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    return [
        result["summary"]
        for result in data["results"]
    ]


def save_figure(
    fig,
    output_dir: Path,
    filename: str,
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir / filename
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {output_path}"
    )


def configure_batch_axis(
    ax,
    batch_sizes: list[int],
) -> None:
    ax.set_xticks(
        batch_sizes
    )

    ax.set_xticklabels(
        [
            str(batch_size)
            for batch_size
            in batch_sizes
        ]
    )


def plot_latency(
    summaries: list[dict],
    output_dir: Path,
) -> None:
    batch_sizes = [
        summary["batch_size"]
        for summary in summaries
    ]

    latencies = [
        summary["mean_latency_ms"]
        for summary in summaries
    ]

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        batch_sizes,
        latencies,
        marker="o",
    )

    configure_batch_axis(
        ax,
        batch_sizes,
    )

    ax.set_xlabel(
        "Batch size"
    )

    ax.set_ylabel(
        "E2E latency (ms)"
    )

    ax.set_title(
        "Batch Scaling: E2E Latency"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    save_figure(
        fig,
        output_dir,
        "batch_scaling_latency.png",
    )


def plot_throughput(
    summaries: list[dict],
    output_dir: Path,
) -> None:
    batch_sizes = [
        summary["batch_size"]
        for summary in summaries
    ]

    throughputs = [
        summary[
            "mean_e2e_tokens_per_second"
        ]
        for summary in summaries
    ]

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        batch_sizes,
        throughputs,
        marker="o",
    )

    configure_batch_axis(
        ax,
        batch_sizes,
    )

    ax.set_xlabel(
        "Batch size"
    )

    ax.set_ylabel(
        "Aggregate throughput (tokens/s)"
    )

    ax.set_title(
        "Batch Scaling: Aggregate Throughput"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    save_figure(
        fig,
        output_dir,
        "batch_scaling_throughput.png",
    )


def plot_vram(
    summaries: list[dict],
    output_dir: Path,
) -> None:
    batch_sizes = [
        summary["batch_size"]
        for summary in summaries
    ]

    peak_vram = [
        summary[
            "mean_peak_vram_mb"
        ]
        for summary in summaries
    ]

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        batch_sizes,
        peak_vram,
        marker="o",
    )

    configure_batch_axis(
        ax,
        batch_sizes,
    )

    ax.set_xlabel(
        "Batch size"
    )

    ax.set_ylabel(
        "Peak VRAM (MB)"
    )

    ax.set_title(
        "Batch Scaling: Peak VRAM"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    save_figure(
        fig,
        output_dir,
        "batch_scaling_vram.png",
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=(
            "Path to batch scaling JSON."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/figures"
        ),
    )

    args = parser.parse_args()

    summaries = load_summaries(
        args.input
    )

    summaries.sort(
        key=lambda summary:
        summary["batch_size"]
    )

    plot_latency(
        summaries,
        args.output_dir,
    )

    plot_throughput(
        summaries,
        args.output_dir,
    )

    plot_vram(
        summaries,
        args.output_dir,
    )


if __name__ == "__main__":
    main()