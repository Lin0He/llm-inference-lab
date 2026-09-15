# core/result_store.py

import json

from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from core.runner import ConfigResult


def save_experiment_results(
    experiment_name: str,
    results: list[ConfigResult],
    output_dir: Path,
) -> Path:

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_path = (
        output_dir
        / f"{experiment_name}_{timestamp}.json"
    )

    payload = {
        "experiment": experiment_name,
        "timestamp": timestamp,
        "results": [
            asdict(result)
            for result in results
        ],
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            indent=2,
        )

    return output_path