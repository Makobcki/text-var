#!/usr/bin/env python

import argparse
import asyncio
import json
import logging
import os
import queue
import random
import re
import subprocess
import warnings
import zlib
from concurrent.futures import ThreadPoolExecutor

import orjson
import redis.asyncio as aioredis
import xxhash
from datasets import load_dataset
from rich.console import Console
from rich.live import Live
from rich.logging import RichHandler
from rich.table import Table

# --- Configuration & Setup ---
warnings.filterwarnings("ignore")
TOKEN_PATTERN = re.compile(r"\S+|\s+")
REDIS_QUEUE_NAME = "dataset_output_queue"

# Setup Rich Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(show_path=False, rich_tracebacks=True)],
)
logger = logging.getLogger("dataset_processor")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Distributed Dataset Processing (Async)")
    parser.add_argument(
        "--role", choices=["main", "remote"], required=True, help="Role of this node."
    )
    parser.add_argument(
        "--config",
        default="../configs/data_config.json",
        help="Path to dataset configuration file.",
    )
    parser.add_argument("--redis-host", default="localhost", help="Redis server host.")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis server port.")
    parser.add_argument(
        "--output-dir", default="./dataset_shards", help="Directory for temporary shards."
    )
    parser.add_argument(
        "--final-dir", default="./dataset_final", help="Directory for final merged shards."
    )
    parser.add_argument(
        "--num-final-shards", type=int, default=128, help="Number of final shards to generate."
    )
    parser.add_argument(
        "--local-shards", type=int, default=4, help="Number of temporary local shards per dataset."
    )
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"), help="HuggingFace API token.")
    parser.add_argument("--reset", action="store_true", help="Reset progress in Redis.")
    return parser.parse_args()


# --- Core Functions ---


def load_config(config_path):
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        exit(1)


def merge_and_shuffle(args):
    logger.info("Starting stream merge and shuffle of shards...")
    out_files = [
        os.path.join(args.output_dir, f)
        for f in os.listdir(args.output_dir)
        if f.endswith(".jsonl")
    ]

    if not out_files:
        logger.warning("No files found to merge.")
        return

    final_files = {
        i: open(os.path.join(args.final_dir, f"final_shard_{i:03d}.jsonl"), "a", encoding="utf-8")
        for i in range(args.num_final_shards)
    }

    total_lines = 0
    for temp_file in out_files:
        try:
            with open(temp_file, "r", encoding="utf-8") as f:
                for line in f:
                    shard_idx = random.randint(0, args.num_final_shards - 1)
                    final_files[shard_idx].write(line)
                    total_lines += 1
            os.remove(temp_file)
        except Exception as e:
            logger.error(f"Error merging file {temp_file}: {e}")

    for f in final_files.values():
        f.close()

    logger.info("Executing deep shuffle (shuf) inside each final shard...")
    for i in range(args.num_final_shards):
        filepath = os.path.join(args.final_dir, f"final_shard_{i:03d}.jsonl")
        if os.path.exists(filepath):
            cmd = f'shuf "{filepath}" -o "{filepath}.shuf" && mv "{filepath}.shuf" "{filepath}"'
            subprocess.run(cmd, shell=True)

    logger.info(
        f"Merge completed. {total_lines} documents distributed and shuffled across {args.num_final_shards} shards."
    )


async def process_single_dataset(r_client, config, args):
    name = config["name"]
    logger.info(f"[{name}] Starting processing...")

    exact_hash_key = "global_exact_hashes"
    added_count = int(await r_client.hget("dataset_added", name) or 0)
    skipped_count = int(await r_client.hget("dataset_skipped", name) or 0)
    processed_bytes = int(await r_client.hget("dataset_bytes", name) or 0)

    target_size = config.get("target_size_gb", float("inf"))
    target_size_bytes = int(target_size * 1024**3) if target_size != float("inf") else float("inf")

    if processed_bytes >= target_size_bytes:
        await r_client.hset("dataset_finished", name, "1")
        logger.info(
            f"[{name}] Already finished. Unique: {added_count}, Duplicates: {skipped_count}"
        )
        return name, True

    shard_files = {}
    if args.role == "main":
        shard_files = {
            i: open(
                os.path.join(args.output_dir, f"{name}_temp_{i:03d}.jsonl"), "a", encoding="utf-8"
            )
            for i in range(args.local_shards)
        }

    q = queue.Queue(maxsize=1000)
    load_args = config["kwargs"].copy()
    if args.hf_token:
        load_args["token"] = args.hf_token
    load_args["trust_remote_code"] = True

    try:
        dataset = load_dataset(**load_args, streaming=True)
    except Exception as e:
        logger.error(f"[{name}] Initialization error: {e}")
        return name, False

    def fetcher_thread():
        try:
            for item in dataset:
                q.put(item)
        except Exception as e:
            logger.error(f"[{name}] Fetcher error: {e}")
        finally:
            q.put(None)

    loop = asyncio.get_running_loop()
    fetch_task = loop.run_in_executor(None, fetcher_thread)

    docs_buffer = []
    target_reached = False

    async def update_progress_periodically():
        while not target_reached and not fetch_task.done():
            await r_client.hset("dataset_added", name, added_count)
            await r_client.hset("dataset_skipped", name, skipped_count)
            await r_client.hset("dataset_bytes", name, processed_bytes)
            await r_client.hset("dataset_progress", name, added_count + skipped_count)
            await asyncio.sleep(2)

    progress_task = asyncio.create_task(update_progress_periodically())

    async def flush_buffer():
        nonlocal processed_bytes, added_count, skipped_count, target_reached
        if not docs_buffer:
            return False

        ids_to_check = [d[0] for d in docs_buffer]
        sadd_results = await r_client.execute_command("BF.MADD", exact_hash_key, *ids_to_check)

        batch_to_send = []

        for i, (d_id, txt, b_len, meta) in enumerate(docs_buffer):
            if sadd_results[i] == 0:
                skipped_count += 1
                continue

            processed_bytes += b_len
            doc = {"content": txt, "meta": meta}

            if args.role == "remote":
                batch_to_send.append(doc)
            else:
                json_bytes = orjson.dumps(doc, default=str)
                shard_idx_file = random.randint(0, args.local_shards - 1)
                shard_files[shard_idx_file].write(json_bytes.decode("utf-8") + "\n")

            added_count += 1
            if processed_bytes >= target_size_bytes:
                target_reached = True
                break

        if args.role == "remote" and batch_to_send:
            while await r_client.llen(REDIS_QUEUE_NAME) > 1000:
                await asyncio.sleep(2)

            def compress_batch(data):
                return zlib.compress(orjson.dumps(data, default=str))

            compressed_batch = await loop.run_in_executor(None, compress_batch, batch_to_send)
            await r_client.rpush(REDIS_QUEUE_NAME, compressed_batch)

        docs_buffer.clear()
        return target_reached

    text_cols = config.get("text_cols")
    text_col = config.get("text_col")
    success = False

    try:
        while True:
            item = await loop.run_in_executor(None, q.get)
            if item is None:
                success = True
                break

            if text_cols:
                parts = []
                for col in text_cols:
                    val = item.get(col)
                    if val:
                        if col in ["old_contents", "new_contents", "code"] or (
                            col == "content" and name == "the-stack"
                        ):
                            parts.append(f"<{col}>\n```python\n{val}\n```")
                        else:
                            parts.append(f"<{col}>\n{val}")
                text = "\n\n".join(parts)
            else:
                text = item.get(text_col)
                if text and (
                    text_col in ["code", "old_contents", "new_contents"]
                    or (text_col == "content" and name == "the-stack")
                ):
                    text = f"```python\n{text}\n```"

            if not text:
                continue

            b_text = text.encode("utf-8")
            doc_id = xxhash.xxh128(b_text).hexdigest()
            meta = {k: v for k, v in item.items() if k not in (text_cols or [text_col])}
            meta["source"] = name
            docs_buffer.append((doc_id, text, len(b_text), meta))

            if len(docs_buffer) >= 500:
                tr = await flush_buffer()
                if tr:
                    success = True
                    break

        if docs_buffer and not target_reached:
            await flush_buffer()
            if processed_bytes >= target_size_bytes:
                success = True

    except Exception as e:
        logger.error(f"[{name}] Process execution error: {e}")
    finally:
        target_reached = True
        progress_task.cancel()

        if args.role == "main":
            for f in shard_files.values():
                f.close()

        await r_client.hset("dataset_added", name, added_count)
        await r_client.hset("dataset_skipped", name, skipped_count)
        await r_client.hset("dataset_bytes", name, processed_bytes)
        await r_client.hset("dataset_progress", name, added_count + skipped_count)

        if success:
            await r_client.hset("dataset_finished", name, "1")
            logger.info(f"[{name}] Finished. Unique: {added_count}, Duplicates: {skipped_count}")
        else:
            logger.warning(
                f"[{name}] Stopped prematurely. Unique: {added_count}, Duplicates: {skipped_count}"
            )

    return name, True


async def redis_queue_consumer(r_client, args):
    logger.info(f"[Consumer] Listening on queue '{REDIS_QUEUE_NAME}' for remote data...")

    shard_files = {
        i: open(os.path.join(args.output_dir, f"remote_temp_{i:03d}.jsonl"), "a", encoding="utf-8")
        for i in range(args.local_shards)
    }

    try:
        while True:
            result = await r_client.blpop(REDIS_QUEUE_NAME, timeout=5)
            if result:
                _, compressed_data = result
                try:
                    batch_json = zlib.decompress(compressed_data)
                    batch = orjson.loads(batch_json)
                    for doc in batch:
                        json_bytes = orjson.dumps(doc)
                        shard_idx_file = random.randint(0, args.local_shards - 1)
                        shard_files[shard_idx_file].write(json_bytes.decode("utf-8") + "\n")
                except Exception as e:
                    logger.error(f"[Consumer] Error processing Redis batch: {e}")
            else:
                await asyncio.sleep(0.1)
    except asyncio.CancelledError:
        pass
    finally:
        for f in shard_files.values():
            f.close()


async def global_progress_monitor(r_client, all_datasets):
    console = Console()
    with Live(console=console, refresh_per_second=2) as live:
        try:
            while True:
                table = Table(title="Dataset Processing Status (Global)")
                table.add_column("Dataset", style="cyan")
                table.add_column("Total Processed", justify="right")
                table.add_column("Added", style="green", justify="right")
                table.add_column("Skipped (Dupes)", style="red", justify="right")
                table.add_column("Size (GB)", justify="right")
                table.add_column("Status", justify="center")

                all_progress = await r_client.hgetall("dataset_progress")
                all_added = await r_client.hgetall("dataset_added")
                all_skipped = await r_client.hgetall("dataset_skipped")
                all_bytes = await r_client.hgetall("dataset_bytes")
                all_finished = await r_client.hgetall("dataset_finished")

                all_done = True
                for ds in all_datasets:
                    name = ds["name"]
                    prog = int(all_progress.get(name, 0))
                    add = int(all_added.get(name, 0))
                    skip = int(all_skipped.get(name, 0))
                    b_bytes = int(all_bytes.get(name, 0))
                    is_fin = all_finished.get(name, "0") == "1"

                    gb = b_bytes / (1024**3)
                    status = "[green]Complete[/green]" if is_fin else "[yellow]In Progress[/yellow]"
                    if not is_fin:
                        all_done = False

                    table.add_row(name, str(prog), str(add), str(skip), f"{gb:.2f}", status)

                live.update(table)
                if all_done:
                    break
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass


async def main():
    args = parse_arguments()
    config_data = load_config(args.config)
    all_datasets = config_data.get("datasets", [])

    datasets_to_process = [ds for ds in all_datasets if ds.get("role") == args.role]

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.final_dir, exist_ok=True)

    try:
        r_client = aioredis.Redis(host=args.redis_host, port=args.redis_port, decode_responses=True)
        r_client_raw = aioredis.Redis(host=args.redis_host, port=args.redis_port)

        if args.reset:
            logger.warning("Resetting progress in Redis...")
            await r_client.delete(
                "dataset_added",
                "dataset_skipped",
                "dataset_bytes",
                "dataset_progress",
                "dataset_finished",
                "global_exact_hashes",
                REDIS_QUEUE_NAME,
            )
            logger.info("Progress reset successfully.")

        exists = await r_client.exists("global_exact_hashes")
        if not exists:
            await r_client.execute_command(
                "BF.RESERVE", "global_exact_hashes", "0.001", "1000000000"
            )
            logger.info("Bloom Filter initialized.")
    except Exception as e:
        logger.warning(
            f"Bloom Filter check/initialization warning (Ensure RedisStack/RedisBloom is installed): {e}"
        )

    logger.info(
        f"Starting execution ({len(datasets_to_process)} tasks) in ROLE={args.role} mode..."
    )

    tasks = []
    if args.role == "main":
        consumer_task = asyncio.create_task(redis_queue_consumer(r_client_raw, args))
        tasks.append(consumer_task)
        monitor_task = asyncio.create_task(global_progress_monitor(r_client, all_datasets))
        tasks.append(monitor_task)

    process_tasks = []
    for ds in datasets_to_process:
        process_tasks.append(asyncio.create_task(process_single_dataset(r_client_raw, ds, args)))

    await asyncio.gather(*process_tasks)

    if args.role == "main":
        logger.info("Local datasets on the main machine are processed.")
        logger.info("The consumer is still listening to Redis for data from remote machines.")
        logger.info("--> Press Ctrl+C when remote nodes are finished to execute the final merge.")

        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        except KeyboardInterrupt:
            logger.info("Stopping data intake from remote machines...")
            for t in tasks:
                t.cancel()

        merge_and_shuffle(args)
    else:
        logger.info("Remote processing complete! All documents dispatched to Redis.")
        logger.info("Don't forget to stop the main machine (Ctrl+C) to trigger the final merge.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Program interrupted by user.")
