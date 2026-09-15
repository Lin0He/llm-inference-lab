# experiments/context_scaling.py

from dataclasses import replace

from core.config import ExperimentConfig


CONTEXT_LENGTHS = [
    128,
    512,
    1024,
    2048,
]


def build_context_scaling_configs(
    base_config: ExperimentConfig,
) -> list[ExperimentConfig]:

    return [
        replace(
            base_config,
            input_tokens=context_length,
        )
        for context_length in CONTEXT_LENGTHS
    ]