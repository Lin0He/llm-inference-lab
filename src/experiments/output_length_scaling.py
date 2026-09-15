from dataclasses import replace

from core.config import ExperimentConfig


OUTPUT_LENGTHS = [
    16,
    32,
    64,
    128,
    256,
]


def build_output_length_scaling_configs(
    base_config: ExperimentConfig,
) -> list[ExperimentConfig]:

    return [
        replace(
            base_config,
            input_tokens=512,
            batch_size=1,
            max_new_tokens=output_length,
        )
        for output_length in OUTPUT_LENGTHS
    ]