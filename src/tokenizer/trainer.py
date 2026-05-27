from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from transformers import PreTrainedTokenizerFast


class BPETokenizerTrainer:
    """
    Класс для обучения кастомного Byte-Level BPE токенизатора.
    Адаптирован для задач текстовой генерации (VAR).
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        special_tokens: list[str] | None = None,
        min_frequency: int = 2,
    ):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency

        # Базовые специальные токены. Можно расширить маркерами масштабов для VAR.
        self.special_tokens = special_tokens or [
            "<pad>",
            "<unk>",
            "<s>",  # Начало последовательности (BOS)
            "</s>",  # Конец последовательности (EOS)
            "<mask>",
        ]

        # Инициализация BPE модели
        self.tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

        # ByteLevel позволяет обрабатывать любые символы без UNK-токенов
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        self.tokenizer.decoder = decoders.ByteLevel()

    def train_from_files(self, files: str | list[str]) -> PreTrainedTokenizerFast:
        """
        Обучает токенизатор на списке текстовых файлов.
        """
        if isinstance(files, str):
            files = [files]

        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens,
            min_frequency=self.min_frequency,
            show_progress=True,
        )

        self.tokenizer.train(files, trainer)

        # Настройка пост-процессора
        self.tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

        # Оборачиваем в transformers tokenizer для удобства интеграции в модель
        fast_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=self.tokenizer,
            unk_token="<unk>",
            pad_token="<pad>",
            bos_token="<s>",
            eos_token="</s>",
            mask_token="<mask>",
        )

        return fast_tokenizer

    def train_from_iterator(self, iterator, length: int | None = None) -> PreTrainedTokenizerFast:
        """
        Альтернативный метод обучения из Python-генератора (полезно для HuggingFace Datasets).
        """
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens,
            min_frequency=self.min_frequency,
            show_progress=True,
        )

        self.tokenizer.train_from_iterator(iterator, trainer=trainer, length=length)
        self.tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

        return PreTrainedTokenizerFast(
            tokenizer_object=self.tokenizer,
            unk_token="<unk>",
            pad_token="<pad>",
            bos_token="<s>",
            eos_token="</s>",
            mask_token="<mask>",
        )
