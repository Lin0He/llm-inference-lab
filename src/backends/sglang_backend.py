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