from dataclasses import replace

from core.config import ExperimentConfig


BACKENDS = [
    "hf",
    "vllm",
    # "sglang",
]


def build_backend_comparison_configs(
    base_config: ExperimentConfig,
) -> dict[str, list[ExperimentConfig]]:
    configs_by_backend = {}

    for backend in BACKENDS:
        configs_by_backend[backend] = [
            replace(
                base_config,
                backend=backend,
                dtype="bfloat16",
                batch_size=1,
                input_tokens=512,
                max_new_tokens=128,
            )
        ]

    return configs_by_backend