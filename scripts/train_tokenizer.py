import argparse
import glob
import os
import signal
import sys

from src.tokenizer.trainer import BPETokenizerTrainer

def _signal_handler(sig, frame):
    print(f"\n[!] Скрипт прерван сигналом {sig}. Принудительный выход...")
    os._exit(1)

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


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
            print(f"Ошибка: В директории {args.data} не найдено .txt файлов.")
            return

        print(f"Найдено файлов для обучения: {len(files)}")
        print(f"Начинается обучение токенизатора из текстовых файлов (vocab_size={args.vocab_size})...")
        tokenizer = trainer.train_from_files(files)
        
    elif os.path.isfile(args.data) and args.data.endswith(".jsonl"):
        print(f"Найдено .jsonl датасет: {args.data}")
        print(f"Начинается обучение токенизатора из JSONL (vocab_size={args.vocab_size})...")
        
        def get_jsonl_iterator(filepath, max_samples=None):
            file_size_gb = os.path.getsize(filepath) / (1024 ** 3)
            use_in_memory = file_size_gb < 2.0  # Только для файлов < 2GB
            
            if use_in_memory:
                # 1. Попытка использовать CUDA (cuDF) для GPU-ускоренного парсинга JSONL (SoA)
                try:
                    import cudf
                    print(f"Оптимизация: используем cuDF (CUDA) для загрузки {filepath}...")
                    df = cudf.read_json(filepath, lines=True)
                    col = "content" if "content" in df.columns else "text"
                    # to_arrow().to_pylist() эффективно конвертирует SoA в список строк
                    data = df[col].dropna().to_arrow().to_pylist()
                    return data[:max_samples] if max_samples else data
                except ImportError:
                    pass
                except Exception as e:
                    print(f"cuDF fallback (ошибка: {e}). Переход к CPU...")

                # 2. Попытка использовать PyArrow (CPU SoA)
                try:
                    from pyarrow import json as pa_json
                    import pyarrow.compute as pc
                    print(f"Оптимизация: используем PyArrow (SoA) для загрузки {filepath}...")
                    read_options = pa_json.ReadOptions(block_size=256 * 1024 * 1024)
                    table = pa_json.read_json(filepath, read_options=read_options)
                    col = "content" if "content" in table.column_names else "text"
                    data = pc.drop_null(table[col]).to_pylist()
                    return data[:max_samples] if max_samples else data
                except ImportError:
                    pass
                except Exception as e:
                    print(f"PyArrow fallback (ошибка: {e}). Переход к стандартному json...")
            else:
                print(f"Файл {filepath} слишком большой ({file_size_gb:.1f} GB). Отключаем in-memory SoA загрузку во избежание OOM.")

            # 3. Фолбек на потоковый генератор (AoS), который не съедает RAM
            print(f"Используем стриминг через стандартный json для загрузки {filepath}...")
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
                                break
            return _streaming_iter()

        tokenizer = trainer.train_from_iterator(get_jsonl_iterator(args.data, args.max_samples))
    else:
        print(f"Ошибка: Путь {args.data} должен быть либо директорией, либо .jsonl файлом.")
        return

    # Сохранение (создаст tokenizer.json, tokenizer_config.json, special_tokens_map.json)
    os.makedirs(args.save_dir, exist_ok=True)
    tokenizer.save_pretrained(args.save_dir)
    print(f"Токенизатор успешно сохранен в директорию: {args.save_dir}")
    print("Теперь его можно загрузить в коде: PreTrainedTokenizerFast.from_pretrained('путь')")


if __name__ == "__main__":
    main()
