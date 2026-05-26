import argparse
import glob
import os

from src.tokenizer.trainer import BPETokenizerTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Обучение BPE токенизатора для text-var")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Путь к директории с текстовыми файлами для обучения (.txt)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Директория для сохранения обученного токенизатора",
    )
    parser.add_argument("--vocab_size", type=int, default=32000, help="Целевой размер словаря")
    parser.add_argument(
        "--min_frequency",
        type=int,
        default=2,
        help="Минимальная частота встречаемости токена для включения в словарь",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Сбор всех текстовых файлов
    files = glob.glob(os.path.join(args.data_dir, "**/*.txt"), recursive=True)
    if not files:
        print(f"Ошибка: В директории {args.data_dir} не найдено .txt файлов.")
        return

    print(f"Найдено файлов для обучения: {len(files)}")
    print(f"Начинается обучение токенизатора (vocab_size={args.vocab_size})...")

    trainer = BPETokenizerTrainer(vocab_size=args.vocab_size, min_frequency=args.min_frequency)

    # Запуск процесса
    tokenizer = trainer.train_from_files(files)

    # Сохранение (создаст tokenizer.json, tokenizer_config.json, special_tokens_map.json)
    os.makedirs(args.save_dir, exist_ok=True)
    tokenizer.save_pretrained(args.save_dir)
    print(f"Токенизатор успешно сохранен в директорию: {args.save_dir}")
    print("Теперь его можно загрузить в коде: PreTrainedTokenizerFast.from_pretrained('путь')")


if __name__ == "__main__":
    main()
