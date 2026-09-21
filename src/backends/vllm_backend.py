import time
import gc

import torch
import vllm
from vllm import LLM, SamplingParams, TokensPrompt

from core.config import ExperimentConfig
from core.metrics import (
    RunResult,
    calculate_e2e_throughput,
)
from core.workload import TokenWorkload


class vLLMBackend:
    def __init__(
        self,
        model_id: str,
        dtype: str,
    ):
        self.model_id = model_id
        self.dtype = dtype
        self.model = None

    def load(self) -> None:
        self.model = LLM(
            model=self.model_id, 
            dtype=self.dtype)


    def close(self) -> None:
        if self.model is not None:
            self.model = None

        gc.collect()
        torch.cuda.empty_cache()
        

    def run(self,
        workload: TokenWorkload, 
        config: ExperimentConfig,
        run_id: int,
        ) -> RunResult:

        if self.model is None:
            raise RuntimeError(
                "Model must be loaded before running inference."
            )

        if config.model_id != self.model_id:
            raise ValueError(
                "Experiment model_id does not match loaded backend."
            )

        if config.dtype != self.dtype:
            raise ValueError(
                "Experiment dtype does not match loaded backend."
            )

        sampling_params = SamplingParams(
            temperature=0, 
            max_tokens=config.max_new_tokens,
            ignore_eos=True,
        )

        inputs = []
        for s in workload.input_ids:
            token_id = TokensPrompt(prompt_token_ids=s)
            inputs.append(token_id)
        
        t0 = time.perf_counter()
        outputs = self.model.generate(inputs, sampling_params)
        t2 = time.perf_counter()

        generated_token_ids = []
        for output in outputs:
            generated_token_ids.append(output.outputs[0].token_ids)
        generated_lengths = [
            len(tokens)
            for tokens in generated_token_ids
        ]

        if not generated_lengths:
            raise RuntimeError(
                "No generated outputs were returned."
            )

        if len(set(generated_lengths)) != 1:
            raise RuntimeError(
                f"Inconsistent generated lengths: {generated_lengths}"
            )

        if len(outputs) != workload.batch_size:
            raise RuntimeError(
            "Number of outputs does not match workload batch size."
            )

        generated_tokens_per_sequence = generated_lengths[0]
        total_generated_tokens = sum(generated_lengths)

        e2e_latency_ms = (t2 - t0) * 1000
        e2e_tps = calculate_e2e_throughput(
            output_tokens=total_generated_tokens,
            e2e_latency_ms=e2e_latency_ms,
        )

        return RunResult(
            run_id=run_id,
            backend=config.backend,
            model_id=config.model_id,
            dtype=config.dtype,
            batch_size=config.batch_size,
            input_tokens=config.input_tokens,
            max_new_tokens=config.max_new_tokens,

            gpu_name=torch.cuda.get_device_name(),
            torch_version=torch.__version__,
            backend_version=vllm.__version__,
            cuda_version=torch.version.cuda or "unknown",

            e2e_latency_ms=e2e_latency_ms,
            ttft_ms=None,
            generated_tokens_per_sequence=generated_tokens_per_sequence,
            total_generated_tokens=total_generated_tokens,
            peak_vram_mb=None,

            tpot_ms=None,
            e2e_tokens_per_second=e2e_tps,
            decode_tokens_per_second=None,
        )