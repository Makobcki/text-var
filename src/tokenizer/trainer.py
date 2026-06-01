from __future__ import annotations

import logging
from typing import Iterable

from rich.logging import RichHandler
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from transformers import PreTrainedTokenizerFast

# Setup Rich Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(show_path=False, rich_tracebacks=True)],
)
logger = logging.getLogger("tokenizer_trainer")


class BPETokenizerTrainer:
    """Trainer for a custom Byte-Level BPE tokenizer, adapted for text generation tasks."""

    def __init__(
        self,
        vocab_size: int = 32000,
        special_tokens: list[str] | None = None,
        min_frequency: int = 2,
    ) -> None:
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency

        self.special_tokens = special_tokens or [
            "<pad>",
            "<unk>",
            "<s>",
            "</s>",
            "<mask>",
        ]

        self.tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        self.tokenizer.decoder = decoders.ByteLevel()

    def train_from_files(self, files: str | list[str]) -> PreTrainedTokenizerFast:
        """Train the tokenizer on a list of text files."""
        if isinstance(files, str):
            files = [files]

        logger.info(f"Training tokenizer on {len(files)} files...")

        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens,
            min_frequency=self.min_frequency,
            show_progress=True,
        )

        self.tokenizer.train(files, trainer)
        self.tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

        logger.info("Training complete. Wrapping into PreTrainedTokenizerFast...")

        return PreTrainedTokenizerFast(
            tokenizer_object=self.tokenizer,
            unk_token="<unk>",
            pad_token="<pad>",
            bos_token="<s>",
            eos_token="</s>",
            mask_token="<mask>",
        )

    def train_from_iterator(
        self, iterator: Iterable[str], length: int | None = None
    ) -> PreTrainedTokenizerFast:
        """Train the tokenizer from a Python generator (e.g., HuggingFace Datasets)."""
        logger.info("Training tokenizer from an iterator...")

        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens,
            min_frequency=self.min_frequency,
            show_progress=True,
        )

        self.tokenizer.train_from_iterator(iterator, trainer=trainer, length=length)
        self.tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

        logger.info("Training complete. Wrapping into PreTrainedTokenizerFast...")

        return PreTrainedTokenizerFast(
            tokenizer_object=self.tokenizer,
            unk_token="<unk>",
            pad_token="<pad>",
            bos_token="<s>",
            eos_token="</s>",
            mask_token="<mask>",
        )
