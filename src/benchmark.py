# benchmark.py

from pathlib import Path

from transformers import AutoTokenizer

from core.config import ExperimentConfig
from core.runner import ExperimentRunner
from core.result_store import save_experiment_results
from core.workload import SyntheticTokenWorkloadGenerator

from backends.hf_backend import HFBackend
from backends.vllm_backend import vLLMBackend
# from backends.sglang_backend import SGLangBackend

from experiments.context_scaling import build_context_scaling_configs
from experiments.output_length_scaling import build_output_length_scaling_configs
from experiments.batch_scaling import build_batch_scaling_configs
from experiments.backend_comparison import build_backend_comparison_configs

RESULT_DIR = Path("../results/raw")


# ---------------------------------------------------------
# System initialization
# ---------------------------------------------------------

def init_system():
    base_config = ExperimentConfig()
    base_config.validate()

    tokenizer = AutoTokenizer.from_pretrained(
        base_config.model_id
    )

    workload_generator = (
        SyntheticTokenWorkloadGenerator(
            tokenizer
        )
    )

    runner = ExperimentRunner(
        workload_generator=workload_generator,
        backend_factory=create_backend,
    )

    return base_config, runner


def create_backend(
    config: ExperimentConfig,
):
    if config.backend == "hf":
        return HFBackend(
            model_id=config.model_id,
            dtype=config.dtype,
        )

    if config.backend == "vllm":
        return vLLMBackend(
            model_id=config.model_id,
            dtype=config.dtype,
        )

    if config.backend == "sglang":
        return SGLangBackend(
            model_id=config.model_id,
            dtype=config.dtype,
        )

    raise ValueError(
        f"Unsupported backend: {config.backend}"
    )


# ---------------------------------------------------------
# Experiment workflows
# ---------------------------------------------------------

def run_context_scaling_workflow(
    base_config: ExperimentConfig,
    runner: ExperimentRunner,
) -> None:

    print("\n==============================")
    print("Context Scaling Experiment")
    print("==============================")

    configs = build_context_scaling_configs(
        base_config
    )

    results = runner.run_configs(
        configs
    )

    output_path = save_experiment_results(
        experiment_name="context_scaling",
        results=results,
        output_dir=RESULT_DIR,
    )

    print(
        f"\nContext scaling results saved to:"
        f"\n{output_path}"
    )

def run_output_length_scaling_workflow(
    base_config,
    runner,
):
    print("\n==============================")
    print("Output Length Scaling Experiment")
    print("==============================")

    configs = build_output_length_scaling_configs(
        base_config
    )

    results = runner.run_configs(configs)

    output_path = save_experiment_results(
        experiment_name="output_length_scaling",
        results=results,
        output_dir=RESULT_DIR,
    )

    print(
        f"\nOutput-length results saved to:"
        f"\n{output_path}"
    )

def run_batch_scaling_workflow(
    base_config,
    runner,
):
    print("\n==============================")
    print("Batch Scaling Experiment")
    print("==============================")

    configs = build_batch_scaling_configs(
        base_config
    )

    results = runner.run_configs(
        configs
    )

    output_path = save_experiment_results(
        experiment_name="batch_scaling",
        results=results,
        output_dir=RESULT_DIR,
    )

    print(
        f"\nBatch-scaling results saved to:"
        f"\n{output_path}"
    )

def run_backend_comparison_workflow(
    base_config,
    runner,
):
    print("\n==============================")
    print("Backend Comparison Experiment")
    print("==============================")

    configs_by_backend = build_backend_comparison_configs(
        base_config
    )

    all_results = []

    for backend, configs in configs_by_backend.items():
        print(f"\nRunning backend: {backend}")

        try:
            results = runner.run_configs(configs)
            all_results.extend(results)

        finally:
            runner.clear_backend_cache()

    output_path = save_experiment_results(
        experiment_name="backend_comparison",
        results=all_results,
        output_dir=RESULT_DIR,
    )

    print(
        f"\nBackend comparison results saved to:"
        f"\n{output_path}"
    )
# ---------------------------------------------------------
# Benchmark orchestration
# ---------------------------------------------------------

def orchestration(base_config, runner) -> None:
    # run_context_scaling_workflow(
    #     base_config=base_config,
    #     runner=runner,
    # )

    # run_output_length_scaling_workflow(
    #     base_config,
    #     runner,
    # )

    # run_batch_scaling_workflow(
    #     base_config,
    #     runner,
    # )

    run_backend_comparison_workflow(
        base_config,
        runner,
    )


def main():
    runner = None

    try:
        base_config, runner = init_system()
        orchestration(
            base_config,
            runner,
        )

    finally:
        if runner is not None:
            runner.close()


if __name__ == "__main__":
    main()