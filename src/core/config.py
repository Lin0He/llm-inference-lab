from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    model_id: str = "Qwen/Qwen3.5-4B"
    dtype: str = "bfloat16"

    batch_size: int = 1
    input_tokens: int = 512
    max_new_tokens: int = 128

    warmup_runs: int = 2
    measure_runs: int = 5

    seed: int = 42

    def validate(self) -> None:
        if self.dtype not in {"float16", "bfloat16"}:
            raise ValueError(
                f"Unsupported dtype: {self.dtype}. "
                "Expected 'float16' or 'bfloat16'."
            )

        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0.")

        if self.input_tokens <= 0:
            raise ValueError("input_tokens must be > 0.")

        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be > 0.")

        if self.warmup_runs < 0:
            raise ValueError("warmup_runs must be >= 0.")

        if self.measure_runs <= 0:
            raise ValueError("measure_runs must be > 0.")