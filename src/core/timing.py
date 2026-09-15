import time

import torch


class FirstTokenTimer:
    """
    Minimal Transformers-compatible streamer used only for TTFT timing.

    The first `put()` from generate() contains the prompt tokens.
    The second `put()` corresponds to the first generated token.
    """

    def __init__(self) -> None:
        self._first_put = True
        self.first_token_time: float | None = None

    def put(self, value) -> None:
        # First call contains the original prompt.
        if self._first_put:
            self._first_put = False
            return

        # Record only the first generated token.
        if self.first_token_time is None:
            torch.cuda.synchronize()
            self.first_token_time = time.perf_counter()

    def end(self) -> None:
        pass