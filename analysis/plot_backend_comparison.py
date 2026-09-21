# analysis/plot_backend_comparison.py

import json
from pathlib import Path

import matplotlib.pyplot as plt


RESULTS_DIR = Path("../results/raw")
FIGURES_DIR = Path("../results/figures")


def load_latest_result() -> dict:
    result_files = sorted(
        RESULTS_DIR.glob("backend_comparison_*.json")
    )

    if not result_files:
        raise FileNotFoundError(
            "No backend comparison result found."
        )

    latest_file = result_files[-1]

    with latest_file.open("r") as f:
        data = json.load(f)

    print(f"Loaded: {latest_file}")
    return data


def extract_summary(data: dict):
    backends = []
    mean_latency = []
    mean_tps = []

    for result in data["results"]:
        summary = result["summary"]

        backends.append(summary["backend"])
        mean_latency.append(summary["mean_latency_ms"])
        mean_tps.append(
            summary["mean_e2e_tokens_per_second"]
        )

    return backends, mean_latency, mean_tps


def plot_latency(
    backends,
    mean_latency,
):
    fig, ax = plt.subplots()

    bars = ax.bar(
        backends,
        mean_latency,
    )

    ax.set_title("Backend Comparison — E2E Latency")
    ax.set_ylabel("Mean E2E latency (ms)")
    ax.set_xlabel("Backend")

    for bar, value in zip(bars, mean_latency):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1f} ms",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()

    output_path = (
        FIGURES_DIR
        / "backend_comparison_latency.png"
    )
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f"Saved: {output_path}")


def plot_throughput(
    backends,
    mean_tps,
):
    fig, ax = plt.subplots()

    bars = ax.bar(
        backends,
        mean_tps,
    )

    ax.set_title(
        "Backend Comparison — E2E Throughput"
    )
    ax.set_ylabel("Mean throughput (tokens/s)")
    ax.set_xlabel("Backend")

    for bar, value in zip(bars, mean_tps):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1f} tok/s",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()

    output_path = (
        FIGURES_DIR
        / "backend_comparison_throughput.png"
    )
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f"Saved: {output_path}")


def main():
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = load_latest_result()

    (
        backends,
        mean_latency,
        mean_tps,
    ) = extract_summary(data)

    plot_latency(
        backends,
        mean_latency,
    )

    plot_throughput(
        backends,
        mean_tps,
    )


if __name__ == "__main__":
    main()