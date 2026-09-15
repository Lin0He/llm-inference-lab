# benchmark.py

from pathlib import Path

from transformers import AutoTokenizer

from core.config import ExperimentConfig
from core.runner import ExperimentRunner
from core.result_store import save_experiment_results
from core.workload import SyntheticTokenWorkloadGenerator

from backends.hf_backend import HFBackend

from experiments.context_scaling import build_context_scaling_configs
from experiments.output_length_scaling import build_output_length_scaling_configs


RESULT_DIR = Path("results/raw")


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
        backend_factory=lambda config: HFBackend(
        model_id=config.model_id,
        dtype=config.dtype,
        ),
    )

    return base_config, runner


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

# ---------------------------------------------------------
# Benchmark orchestration
# ---------------------------------------------------------

def orchestration() -> None:
    base_config, runner = init_system()

    run_context_scaling_workflow(
        base_config=base_config,
        runner=runner,
    )

    # run_output_length_scaling_workflow(
    #     base_config,
    #     runner,
    # )


def main() -> None:
    orchestration()


if __name__ == "__main__":
    main()