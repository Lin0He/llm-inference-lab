from dataclasses import dataclass
from statistics import mean, median, pstdev
from typing import Sequence


@dataclass
class RunResult:
    run_id: int

    # Experimental conditions
    model_id: str
    dtype: str
    backend: str

    batch_size: int
    input_tokens: int
    max_new_tokens: int

    # Environment metadata
    gpu_name: str
    torch_version: str
    backend_version: str
    cuda_version: str

    # Direct observations
    e2e_latency_ms: float
    ttft_ms: float | None
    generated_tokens_per_sequence: int
    total_generated_tokens: int
    peak_vram_mb: float | None

    # Derived metrics
    tpot_ms: float | None
    e2e_tokens_per_second: float
    decode_tokens_per_second: float | None


@dataclass
class ExperimentSummary:
    model_id: str
    dtype: str
    backend: str
    batch_size: int
    input_tokens: int
    max_new_tokens: int

    num_runs: int

    # E2E latency
    mean_latency_ms: float
    median_latency_ms: float
    std_latency_ms: float

    # TTFT
    mean_ttft_ms: float | None
    median_ttft_ms: float | None
    std_ttft_ms: float | None

    # TPOT
    mean_tpot_ms: float | None
    median_tpot_ms: float | None
    std_tpot_ms: float | None

    # Throughput
    mean_e2e_tokens_per_second: float
    median_e2e_tokens_per_second: float

    mean_decode_tokens_per_second: float | None
    median_decode_tokens_per_second: float | None

    # Memory
    mean_peak_vram_mb: float | None
    max_peak_vram_mb: float | None


def summarize_results(
    results: Sequence[RunResult],
) -> ExperimentSummary:
    if not results:
        raise ValueError("Cannot summarize an empty result list.")

    first = results[0]

    latencies = [
        r.e2e_latency_ms
        for r in results
    ]

    ttfts = [
        r.ttft_ms
        for r in results
        if r.ttft_ms is not None
    ]

    tpots = [
        r.tpot_ms
        for r in results
        if r.tpot_ms is not None
    ]

    e2e_throughputs = [
        r.e2e_tokens_per_second
        for r in results
    ]

    decode_throughputs = [
        r.decode_tokens_per_second
        for r in results
        if r.decode_tokens_per_second is not None
    ]

    peak_vram = [
    r.peak_vram_mb
    for r in results
    if r.peak_vram_mb is not None
    ]

    # if not tpots:
    #     raise ValueError(
    #         "No valid TPOT values found in results."
    #     )

    # if not decode_throughputs:
    #     raise ValueError(
    #         "No valid decode throughput values found in results."
    #     )

    return ExperimentSummary(
        model_id=first.model_id,
        dtype=first.dtype,
        backend=first.backend,
        batch_size=first.batch_size,
        input_tokens=first.input_tokens,
        max_new_tokens=first.max_new_tokens,

        num_runs=len(results),

        mean_latency_ms=mean(latencies),
        median_latency_ms=median(latencies),
        std_latency_ms=pstdev(latencies),

        mean_ttft_ms=mean(ttfts) if ttfts else None,
        median_ttft_ms=median(ttfts) if ttfts else None,
        std_ttft_ms=pstdev(ttfts) if ttfts else None,

        mean_tpot_ms=mean(tpots) if tpots else None,
        median_tpot_ms=median(tpots) if tpots else None,
        std_tpot_ms=pstdev(tpots) if tpots else None,


        mean_e2e_tokens_per_second=mean(
            e2e_throughputs
        ),
        median_e2e_tokens_per_second=median(
            e2e_throughputs
        ),

        mean_decode_tokens_per_second=(
            mean(decode_throughputs)
            if decode_throughputs
            else None
        ),

        median_decode_tokens_per_second=(
            median(decode_throughputs)
            if decode_throughputs
            else None
        ),

        mean_peak_vram_mb=(
            mean(peak_vram)
            if peak_vram
            else None
        ),

        max_peak_vram_mb=(
            max(peak_vram)
            if peak_vram
            else None
        ),
    )


def calculate_tpot_ms(
    e2e_latency_ms: float,
    ttft_ms: float,
    output_tokens: int,
) -> float | None:
    if output_tokens <= 1:
        return None

    decode_latency_ms = (
        e2e_latency_ms - ttft_ms
    )

    if decode_latency_ms <= 0:
        return None

    return (
        decode_latency_ms
        / (output_tokens - 1)
    )


def calculate_e2e_throughput(
    output_tokens: int,
    e2e_latency_ms: float,
) -> float:
    if e2e_latency_ms <= 0:
        raise ValueError(
            "e2e_latency_ms must be > 0."
        )

    return (
        output_tokens
        / (e2e_latency_ms / 1000)
    )


def calculate_decode_throughput(
    output_tokens: int,
    e2e_latency_ms: float,
    ttft_ms: float,
) -> float | None:
    if output_tokens <= 1:
        return None

    decode_latency_ms = (
        e2e_latency_ms - ttft_ms
    )

    if decode_latency_ms <= 0:
        return None

    decode_seconds = (
        decode_latency_ms / 1000
    )

    return (
        output_tokens - 1
    ) / decode_seconds