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

        if workload.batch_size != 1:
            raise ValueError(
                "V0.2 TTFT/TPOT measurement currently supports "
                "batch_size=1 only."
            )

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

        timer = FirstTokenTimer()

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        t0 = time.perf_counter()

        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=config.max_new_tokens,
                do_sample=False,
                streamer=timer,
            )

        torch.cuda.synchronize()
        t2 = time.perf_counter()

        if timer.first_token_time is None:
            raise RuntimeError(
                "First generated token was not observed."
            )

        t1 = timer.first_token_time

        e2e_latency_ms = (t2 - t0) * 1000
        ttft_ms = (t1 - t0) * 1000

        input_length = input_ids.shape[1]
        output_length = outputs.shape[1]

        actual_output_tokens = (
            output_length - input_length
        )

        tpot_ms = calculate_tpot_ms(
            e2e_latency_ms=e2e_latency_ms,
            ttft_ms=ttft_ms,
            output_tokens=actual_output_tokens,
        )

        e2e_tps = calculate_e2e_throughput(
            output_tokens=actual_output_tokens,
            e2e_latency_ms=e2e_latency_ms,
        )

        decode_tps = calculate_decode_throughput(
            output_tokens=actual_output_tokens,
            e2e_latency_ms=e2e_latency_ms,
            ttft_ms=ttft_ms,
        )

        peak_vram_mb = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 2)
        )

        return RunResult(
            run_id=run_id,

            model_id=config.model_id,
            dtype=config.dtype,
            batch_size=config.batch_size,
            input_tokens=config.input_tokens,
            max_new_tokens=config.max_new_tokens,

            gpu_name=torch.cuda.get_device_name(),
            torch_version=torch.__version__,
            transformers_version=transformers.__version__,
            cuda_version=torch.version.cuda or "unknown",

            e2e_latency_ms=e2e_latency_ms,
            ttft_ms=ttft_ms,
            actual_output_tokens=actual_output_tokens,
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