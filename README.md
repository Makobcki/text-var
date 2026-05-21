# text-var

## Подготовка датасета

Добавлен скрипт `prepare_dataset.py`, который конвертирует `.txt` или `.jsonl` в кэш токенов (`.pt`) совместимый с `train.py` (`token_cache_path`).

### Пример

```bash
python prepare_dataset.py \
  --input data/train.txt \
  --output data/token_cache.pt
```

Для JSONL:

```bash
python prepare_dataset.py \
  --input data/train.jsonl \
  --field text \
  --output data/token_cache.pt
```

После этого укажи путь в `config/train.json`:

```json
{
  "token_cache_path": "data/token_cache.pt"
}
```
