# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0

from composer.core import Callback, State
from composer.loggers import Logger
from composer.utils import dist
import json

__all__ = ["DataLogger"]


class DataLogger(Callback):
    """Records the packing efficiency for each batch."""

    def __init__(self, log_file: str = "/tmp/datalogger_{rank}.jsonl", log_interval: int = 1):
        self.log_interval = log_interval
        self.accumulated_ids = []
        self.log_file = log_file.format(rank=dist.get_global_rank())

    def after_dataloader(self, state: State, logger: Logger) -> None:
        output = {
                "batch": state.timestamp.batch.value,
                "rank": dist.get_global_rank(),
                "sample_ids": state.batch["sample_ids"],
                }
        self.accumulated_ids.append(output)

        if state.timestamp.batch.value % self.log_interval == 0:
            with open(self.log_file, 'a') as out:
                for cached_output in self.accumulated_ids:
                    print(json.dumps(cached_output), file=out)

            self.accumulated_ids = []

