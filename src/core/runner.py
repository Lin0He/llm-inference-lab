# core/runner.py

from dataclasses import dataclass
from typing import Callable, Sequence

from core.config import ExperimentConfig
from core.metrics import (
    RunResult,
    ExperimentSummary,
    summarize_results,
)

@dataclass
class ConfigResult:
    config: ExperimentConfig
    runs: list[RunResult]
    summary: ExperimentSummary

def backend_cache_key(
    config: ExperimentConfig,
    ) -> tuple:
        return (
            config.backend,
            config.model_id,
            config.dtype,
            config.enable_prefix_caching,
        )

class ExperimentRunner:
    def __init__(
        self,
        workload_generator,
        backend_factory: Callable,
    ):
        self.workload_generator = workload_generator
        self.backend_factory = backend_factory

        # Same model + same dtype can reuse the loaded backend.
        self._backend_cache = {}

    def _get_backend(
        self,
        config: ExperimentConfig,
    ):
        backend_key = backend_cache_key(config)

        if backend_key not in self._backend_cache:
            backend = self.backend_factory(config)
            backend.load()

            self._backend_cache[backend_key] = backend

        return self._backend_cache[backend_key]

    def run_config(
        self,
        config: ExperimentConfig,
    ) -> ConfigResult:
        config.validate()

        workload = self.workload_generator.build(
            batch_size=config.batch_size,
            sequence_length=config.input_tokens,
            seed=config.seed,
        )

        backend = self._get_backend(config)

        print(
            f"\nRunning config: "
            f"backend={config.backend},"
            f"dtype={config.dtype}, "
            f"batch={config.batch_size}, "
            f"input_tokens={config.input_tokens}, "
            f"max_new_tokens={config.max_new_tokens}"
        )

        # Warmup
        for _ in range(config.warmup_runs):
            backend.run(
                workload=workload,
                config=config,
                run_id=-1,
            )

        # Measurement
        runs: list[RunResult] = []

        for run_id in range(config.measure_runs):
            result = backend.run(
                workload=workload,
                config=config,
                run_id=run_id,
            )

            runs.append(result)

            print(
                f"  Run {run_id}: "
                f"E2E={result.e2e_latency_ms:.2f} ms | "
                f"TTFT={self._format_optional(result.ttft_ms, 'ms')} | "
                f"TPOT={self._format_optional(result.tpot_ms, 'ms/token')} | "
                f"E2E TPS={result.e2e_tokens_per_second:.2f} tok/s | "
                f"Decode TPS={self._format_optional(result.decode_tokens_per_second, 'tok/s')} | "
                f"Output/seq={result.generated_tokens_per_sequence} | "
                f"Total output={result.total_generated_tokens} | "
                f"VRAM={self._format_optional(result.peak_vram_mb, 'MB')}"
            )

        summary = summarize_results(runs)

        return ConfigResult(
            config=config,
            runs=runs,
            summary=summary,
        )

    def clear_backend_cache(self) -> None:
        for backend in self._backend_cache.values():
            backend.close()

        self._backend_cache.clear()
    
    def close(self) -> None:
        self.clear_backend_cache()

    def run_configs(
        self,
        configs: Sequence[ExperimentConfig],
    ) -> list[ConfigResult]:
        results = []

        for config in configs:
            results.append(
                self.run_config(config)
            )

        return results

    @staticmethod
    def _format_optional(
        value: float | None,
        unit: str,
    ) -> str:
        if value is None:
            return "N/A"

        return f"{value:.2f} {unit}"