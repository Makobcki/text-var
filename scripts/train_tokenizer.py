import argparse
import glob
import os
import signal
import sys

from rich.console import Console
from rich.theme import Theme
from src.tokenizer.trainer import BPETokenizerTrainer

# Настройка красивого вывода
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
})
console = Console(theme=custom_theme)


# Восстанавливаем дефолтные C-обработчики сигналов.
# Это необходимо, потому что ядро библиотеки tokenizers (Rust) может надолго
# забирать GIL. Дефолтный обработчик завершит процесс мгновенно на уровне ОС.
signal.signal(signal.SIGINT, signal.SIG_DFL)
signal.signal(signal.SIGTERM, signal.SIG_DFL)


def parse_args():
    parser = argparse.ArgumentParser(description="Обучение BPE токенизатора для text-var")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Путь к директории с .txt файлами или к .jsonl датасету для обучения",
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
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Ограничить количество сэмплов для обучения (спасает от OOM в Rust на огромных датасетах, рекомендуемое значение: 1000000 - 5000000)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    trainer = BPETokenizerTrainer(vocab_size=args.vocab_size, min_frequency=args.min_frequency)

    if os.path.isdir(args.data):
        # Сбор всех текстовых файлов
        files = glob.glob(os.path.join(args.data, "**/*.txt"), recursive=True)
        if not files:
            console.print(f"[error]Ошибка:[/error] В директории {args.data} не найдено .txt файлов.")
            return

        console.print(f"[info]Найдено файлов для обучения:[/info] {len(files)}")
        console.print(
            f"[info]Начинается обучение токенизатора из текстовых файлов[/info] (vocab_size={args.vocab_size})..."
        )
        tokenizer = trainer.train_from_files(files)

    elif os.path.isfile(args.data) and args.data.endswith(".jsonl"):
        console.print(f"[info]Найдено .jsonl датасет:[/info] {args.data}")
        console.print(f"[info]Начинается обучение токенизатора из JSONL[/info] (vocab_size={args.vocab_size})...")

        def get_jsonl_iterator(filepath, max_samples=None):
            # 1. Попытка использовать cuDF для батчевого стриминга (SoA + CUDA)
            try:
                import cudf
                console.print(f"[success]Оптимизация:[/success] используем [bold]cuDF (CUDA)[/bold] для батчевой загрузки {filepath}...")
                def _cudf_iter():
                    import io
                    count = 0
                    chunk_bytes = 250 * 1024 * 1024  # 250MB
                    with open(filepath, "r", encoding="utf-8") as f:
                        while True:
                            # Читаем пачку строк (до ~250МБ). Чтение идет целыми строками.
                            lines = f.readlines(chunk_bytes)
                            if not lines:
                                break
                            
                            try:
                                # Отдаем текст в cuDF через StringIO (обход бага libcudf с byte_range)
                                # Указываем compression="uncompressed", чтобы избежать ворнинга от libcudf
                                chunk_df = cudf.read_json(io.StringIO("".join(lines)), lines=True, compression="uncompressed")
                                if chunk_df.empty:
                                    continue
                                
                                col = "content" if "content" in chunk_df.columns else "text"
                                batch = chunk_df[col].dropna().to_arrow().to_pylist()
                                for text in batch:
                                    if text:
                                        yield str(text)
                                        count += 1
                                        if max_samples and count >= max_samples:
                                            return
                            except Exception as e:
                                console.print(f"[warning]Предупреждение: ошибка чтения чанка cuDF:[/warning] {e}")
                                break
                return _cudf_iter()
            except ImportError:
                pass
            except Exception as e:
                console.print(f"[warning]cuDF fallback (ошибка: {e}). Переход к Pandas...[/warning]")

            # 2. Используем стриминг батчами через Pandas (CPU)
            try:
                import pandas as pd
                console.print(f"[success]Используем pandas[/success] для батчевого стриминга {filepath}...")
                def _pandas_iter():
                    count = 0
                    # Увеличен батч для современных CPU
                    for chunk_df in pd.read_json(filepath, lines=True, chunksize=100000):
                        col = "content" if "content" in chunk_df.columns else "text"
                        for text in chunk_df[col].dropna().astype(str):
                            if text:
                                yield text
                                count += 1
                                if max_samples and count >= max_samples:
                                    return
                return _pandas_iter()
            except ImportError:
                console.print(f"[info]Используем стриминг через стандартный json для загрузки {filepath}...[/info]")
                import json
                def _streaming_iter():
                    count = 0
                    with open(filepath, "r", encoding="utf-8") as f:
                        for line in f:
                            if not line.strip():
                                continue
                            payload = json.loads(line)
                            text = str(payload.get("content", payload.get("text", ""))).strip()
                            if text:
                                yield text
                                count += 1
                                if max_samples and count >= max_samples:
                                    return
                return _streaming_iter()

        tokenizer = trainer.train_from_iterator(
            get_jsonl_iterator(args.data, args.max_samples), length=args.max_samples
        )
    else:
        console.print(f"[error]Ошибка:[/error] Путь {args.data} должен быть либо директорией, либо .jsonl файлом.")
        return

    # Сохранение (создаст tokenizer.json, tokenizer_config.json, special_tokens_map.json)
    os.makedirs(args.save_dir, exist_ok=True)
    tokenizer.save_pretrained(args.save_dir)
    console.print(f"[success]Токенизатор успешно сохранен в директорию:[/success] {args.save_dir}")
    console.print("[info]Теперь его можно загрузить в коде:[/info] [bold]PreTrainedTokenizerFast.from_pretrained('путь')[/bold]")


if __name__ == "__main__":
    main()
