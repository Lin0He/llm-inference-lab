import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_summaries(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return [
        result["summary"]
        for result in data["results"]
    ]


def configure_log2_x_axis(
    ax,
    x_values: list[int],
) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(x_values)
    ax.set_xticklabels(
        [str(value) for value in x_values]
    )


def save_figure(
    fig,
    output_dir: Path,
    filename: str,
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_dir / filename

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output_path}")


def plot_latency_ttft(
    summaries: list[dict],
    output_dir: Path,
) -> None:
    output_tokens = [
        summary["max_new_tokens"]
        for summary in summaries
    ]

    ttft = [
        summary["mean_ttft_ms"]
        for summary in summaries
    ]

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        output_tokens,
        ttft,
        marker="o",
        label="TTFT",
    )

    configure_log2_x_axis(
        ax,
        output_tokens,
    )

    ax.set_xlabel(
        "Output tokens"
    )

    ax.set_ylabel(
        "Latency (ms)"
    )

    ax.set_title(
        "Output Length Scaling: TTFT Latency"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    save_figure(
        fig,
        output_dir,
        "output_length_latency_ttft.png",
    )

def plot_latency_e2e_latency(
    summaries: list[dict],
    output_dir: Path,
    ) -> None:

    output_tokens = [
        summary["max_new_tokens"]
        for summary in summaries
    ]

    e2e_latency = [
        summary["mean_latency_ms"]
        for summary in summaries
    ]

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        output_tokens,
        e2e_latency,
        marker="o",
        label="E2E latency",
    )

    configure_log2_x_axis(
        ax,
        output_tokens,
    )

    ax.set_xlabel(
        "Output tokens"
    )

    ax.set_ylabel(
        "Latency (ms)"
    )

    ax.set_title(
        "Output Length Scaling: E2E Latency"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    save_figure(
        fig,
        output_dir,
        "output_length_latency_e2e_latency.png",
    )


def plot_tpot(
    summaries: list[dict],
    output_dir: Path,
) -> None:
    output_tokens = [
        summary["max_new_tokens"]
        for summary in summaries
    ]

    tpot = [
        summary["mean_tpot_ms"]
        for summary in summaries
    ]

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        output_tokens,
        tpot,
        marker="o",
    )

    configure_log2_x_axis(
        ax,
        output_tokens,
    )

    ax.set_xlabel(
        "Output tokens"
    )

    ax.set_ylabel(
        "TPOT (ms/token)"
    )

    ax.set_title(
        "Output Length Scaling: TPOT"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    save_figure(
        fig,
        output_dir,
        "output_length_tpot.png",
    )


def plot_throughput(
    summaries: list[dict],
    output_dir: Path,
) -> None:
    output_tokens = [
        summary["max_new_tokens"]
        for summary in summaries
    ]

    e2e_tps = [
        summary[
            "mean_e2e_tokens_per_second"
        ]
        for summary in summaries
    ]

    decode_tps = [
        summary[
            "mean_decode_tokens_per_second"
        ]
        for summary in summaries
    ]

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        output_tokens,
        e2e_tps,
        marker="o",
        label="E2E throughput",
    )

    ax.plot(
        output_tokens,
        decode_tps,
        marker="o",
        label="Decode throughput",
    )

    configure_log2_x_axis(
        ax,
        output_tokens,
    )

    ax.set_xlabel(
        "Output tokens"
    )

    ax.set_ylabel(
        "Throughput (tokens/s)"
    )

    ax.set_title(
        "Output Length Scaling: Throughput"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    save_figure(
        fig,
        output_dir,
        "output_length_throughput.png",
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to output-length scaling JSON.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures"),
    )

    args = parser.parse_args()

    summaries = load_summaries(
        args.input
    )

    summaries.sort(
        key=lambda summary:
        summary["max_new_tokens"]
    )

    plot_latency_e2e_latency(
        summaries,
        args.output_dir,
    )

    plot_latency_ttft(
        summaries,
        args.output_dir,
    )

    plot_tpot(
        summaries,
        args.output_dir,
    )

    plot_throughput(
        summaries,
        args.output_dir,
    )


if __name__ == "__main__":
    main()