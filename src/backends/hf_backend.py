import time

import torch
import transformers
from transformers import AutoModelForMultimodalLM

from core.config import ExperimentConfig
from core.metrics import (
    RunResult,
    calculate_e2e_throughput,
    calculate_decode_throughput,
    calculate_tpot_ms,
)
from core.workload import TokenWorkload
from core.timing import FirstTokenTimer


class HFBackend:
    def __init__(
        self,
        model_id: str,
        dtype: str,
    ):
        self.model_id = model_id
        self.dtype = dtype
        self.model = None

    def load(self) -> None:
        torch_dtype = self._resolve_dtype(self.dtype)

        self.model = AutoModelForMultimodalLM.from_pretrained(
            self.model_id,
            dtype=torch_dtype,
            device_map="cuda",
        )

        self.model.eval()

    def close(self) -> None:
        if self.model is not None:
            self.model = None

        torch.cuda.empty_cache()

    def run(
        self,
        workload: TokenWorkload,
        config: ExperimentConfig,
        run_id: int,
    ) -> RunResult:
        if self.model is None:
            raise RuntimeError(
                "Model must be loaded before running inference."
            )

        # if workload.batch_size != 1:
        #     raise ValueError(
        #         "V0.2 TTFT/TPOT measurement currently supports "
        #         "batch_size=1 only."
        #     )

        if config.model_id != self.model_id:
            raise ValueError(
                "Experiment model_id does not match loaded backend."
            )

        if config.dtype != self.dtype:
            raise ValueError(
                "Experiment dtype does not match loaded backend."
            )

        input_ids = torch.tensor(
            workload.input_ids,
            dtype=torch.long,
            device="cuda",
        )

        attention_mask = torch.tensor(
            workload.attention_mask,
            dtype=torch.long,
            device="cuda",
        )

        measure_fine_grained_latency = (
            workload.batch_size == 1
        )

        timer = (
            FirstTokenTimer()
            if measure_fine_grained_latency
            else None
        )

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        t0 = time.perf_counter()

        generation_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": config.max_new_tokens,
            "do_sample": False,
        }

        if timer is not None:
            generation_kwargs["streamer"] = timer

        with torch.inference_mode():
            outputs = self.model.generate(
                **generation_kwargs
            )

        torch.cuda.synchronize()
        t2 = time.perf_counter()

        # if timer.first_token_time is None:
        #     raise RuntimeError(
        #         "First generated token was not observed."
        #     )

        # t1 = timer.first_token_time

        # e2e_latency_ms = (t2 - t0) * 1000
        # ttft_ms = (t1 - t0) * 1000

        e2e_latency_ms = (t2 - t0) * 1000

        if timer is not None:
            if timer.first_token_time is None:
                raise RuntimeError(
                    "First generated token was not observed."
                )

            ttft_ms = (timer.first_token_time - t0) * 1000

        else:
            ttft_ms = None

        input_length = input_ids.shape[1]
        output_length = outputs.shape[1]

        generated_tokens_per_sequence = (
            output_length - input_length
        )

        total_generated_tokens = (
            generated_tokens_per_sequence
            * workload.batch_size
        )

        e2e_tps = calculate_e2e_throughput(
            output_tokens=total_generated_tokens,
            e2e_latency_ms=e2e_latency_ms,
        )

        # decode_tps = calculate_decode_throughput(
        #     output_tokens=actual_output_tokens,
        #     e2e_latency_ms=e2e_latency_ms,
        #     ttft_ms=ttft_ms,
        # )

        # tpot_ms = calculate_tpot_ms(
        #     e2e_latency_ms=e2e_latency_ms,
        #     ttft_ms=ttft_ms,
        #     output_tokens=actual_output_tokens,
        # )

        if ttft_ms is not None:
            tpot_ms = calculate_tpot_ms(
                e2e_latency_ms=e2e_latency_ms,
                ttft_ms=ttft_ms,
                output_tokens=generated_tokens_per_sequence,
            )

            decode_tps = calculate_decode_throughput(
                output_tokens=generated_tokens_per_sequence,
                e2e_latency_ms=e2e_latency_ms,
                ttft_ms=ttft_ms,
            )

        else:
            tpot_ms = None
            decode_tps = None

        peak_vram_mb = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 2)
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
            backend_version=transformers.__version__,
            cuda_version=torch.version.cuda or "unknown",

            e2e_latency_ms=e2e_latency_ms,
            ttft_ms=ttft_ms,
            generated_tokens_per_sequence=generated_tokens_per_sequence,
            total_generated_tokens=total_generated_tokens,
            peak_vram_mb=peak_vram_mb,

            tpot_ms=tpot_ms,
            e2e_tokens_per_second=e2e_tps,
            decode_tokens_per_second=decode_tps,
        )

    @staticmethod
    def _resolve_dtype(dtype: str) -> torch.dtype:
        mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }

        try:
            return mapping[dtype]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported dtype: {dtype}"
            ) from exc