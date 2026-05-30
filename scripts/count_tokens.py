import argparse
import json
from transformers import AutoTokenizer
from tqdm import tqdm
import multiprocessing as mp

# Global tokenizer instance for each worker process
tokenizer = None

def init_worker(tokenizer_path):
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

def process_line(args):
    line, text_column = args
    try:
        data = json.loads(line)
        text = data.get(text_column, "")
        if text:
            # Return length of encoded tokens
            return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        pass
    return 0

def main():
    parser = argparse.ArgumentParser(description="Count tokens in a JSONL dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset (.jsonl)")
    parser.add_argument("--tokenizer", type=str, default="./tokenizer", help="Path to the tokenizer directory or HF model name")
    parser.add_argument("--text_column", type=str, default="content", help="JSON key containing the text")
    parser.add_argument("--num_workers", type=int, default=max(1, mp.cpu_count() - 1), help="Number of processes to use")
    parser.add_argument("--chunk_size", type=int, default=2000, help="Chunk size for multiprocessing")
    args = parser.parse_args()

    print(f"Initializing {args.num_workers} workers with tokenizer '{args.tokenizer}'...")
    
    pool = mp.Pool(processes=args.num_workers, initializer=init_worker, initargs=(args.tokenizer,))
    
    def line_generator():
        with open(args.dataset, "r", encoding="utf-8") as f:
            for line in f:
                yield (line, args.text_column)
                
    print(f"Counting tokens in {args.dataset}...")
    
    total_tokens = 0
    generator = line_generator()
    
    try:
        for count in tqdm(pool.imap_unordered(process_line, generator, chunksize=args.chunk_size)):
            total_tokens += count
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    finally:
        pool.close()
        pool.join()

    print(f"\n======================================")
    print(f"Total tokens in dataset: {total_tokens:,}")
    print(f"======================================")

if __name__ == "__main__":
    main()
