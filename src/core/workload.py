from dataclasses import dataclass
from typing import Protocol


@dataclass
class TokenWorkload:
    input_ids: list[list[int]]
    attention_mask: list[list[int]]

    batch_size: int
    sequence_length: int

    tokenizer_id: str
    workload_type: str
    seed: int

    def validate(self) -> None:
        if len(self.input_ids) != self.batch_size:
            raise ValueError(
                "Number of input sequences does not match batch_size."
            )

        if len(self.attention_mask) != self.batch_size:
            raise ValueError(
                "Number of attention masks does not match batch_size."
            )

        for sequence in self.input_ids:
            if len(sequence) != self.sequence_length:
                raise ValueError(
                    "Input sequence length does not match sequence_length."
                )

        for mask in self.attention_mask:
            if len(mask) != self.sequence_length:
                raise ValueError(
                    "Attention mask length does not match sequence_length."
                )


class TokenizerLike(Protocol):
    name_or_path: str
    all_special_ids: list[int]

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
    ) -> list[int]:
        ...


class SyntheticTokenWorkloadGenerator:
    def __init__(self, tokenizer: TokenizerLike):
        self.tokenizer = tokenizer

    def build(
        self,
        batch_size: int,
        sequence_length: int,
        seed: int,
    ) -> TokenWorkload:

        required_tokens = batch_size * sequence_length

        token_stream = self._build_token_stream(
            required_tokens=required_tokens
        )

        sequences = []

        for i in range(batch_size):
            start = i * sequence_length
            end = start + sequence_length

            sequence = token_stream[start:end]
            sequences.append(sequence)

        attention_mask = [
            [1] * sequence_length
            for _ in range(batch_size)
        ]

        workload = TokenWorkload(
            input_ids=sequences,
            attention_mask=attention_mask,
            batch_size=batch_size,
            sequence_length=sequence_length,
            tokenizer_id=self.tokenizer.name_or_path,
            workload_type="synthetic_tokens",
            seed=seed,
        )

        workload.validate()

        return workload

    def _build_token_stream(
        self,
        required_tokens: int,
    ) -> list[int]:

        base_text = (
            "Machine learning systems require careful measurement "
            "of latency, throughput, memory usage, and computational "
            "efficiency. Large language model inference consists of "
            "multiple computational stages whose performance depends "
            "on sequence length, batching, hardware utilization, and "
            "model architecture. "
        )

        text = base_text
        token_ids = self.tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        while len(token_ids) < required_tokens:
            text += base_text

            token_ids = self.tokenizer.encode(
                text,
                add_special_tokens=False,
            )

        special_ids = set(self.tokenizer.all_special_ids)

        token_ids = [
            token_id
            for token_id in token_ids
            if token_id not in special_ids
        ]

        if len(token_ids) < required_tokens:
            raise RuntimeError(
                "Not enough non-special tokens to construct workload."
            )

        return token_ids[:required_tokens]