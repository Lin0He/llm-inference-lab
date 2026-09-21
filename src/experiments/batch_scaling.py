from dataclasses import replace

from core.config import ExperimentConfig


BATCH_SIZES = [1, 2, 4, 8]


def build_batch_scaling_configs(
    base_config: ExperimentConfig,
) -> list[ExperimentConfig]:

    return [
        replace(
            base_config,
            batch_size=batch_size,
            input_tokens=512,
            max_new_tokens=128,
        )
        for batch_size in BATCH_SIZES
    ]