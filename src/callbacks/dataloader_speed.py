# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0

import time
from composer.core import Callback, State
from composer.loggers import Logger

__all__ = ["DataloaderSpeedMonitor"]


class DataloaderSpeedMonitor(Callback):
    """Measure how long it takes to return a batch from the dataloader."""

    def __init__(self, *args, **kwargs):
        self.dataloader_start_time = None

    def before_dataloader(self, state: State, logger: Logger) -> None:
        if self.dataloader_start_time is not None:
            total_time = time.time_ns() - self.dataloader_start_time
            logger.log_metrics(
                {
                    "throughput/total_time_ms": total_time / 1e6,
                }
            )
        self.dataloader_start_time = time.time_ns()

    def after_dataloader(self, state: State, logger: Logger) -> None:
        loader_time = time.time_ns() - self.dataloader_start_time
        logger.log_metrics(
            {
                "throughput/dataloader_serve_time_ms": loader_time / 1e6,
            }
        )


    def batch_start(self, state: State, logger: Logger) -> None:
        del logger  # unused
        self.batch_start_time = time.time_ns()

    def batch_end(self, state: State, logger: Logger) -> None:
        batch_time = time.time_ns() - self.batch_start_time
        logger.log_metrics(
            {
                "throughput/batch_time_ms": batch_time / 1e6,
            }
        )
