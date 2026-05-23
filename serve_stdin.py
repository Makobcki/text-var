from __future__ import annotations

import argparse
import json
import logging
import re
import signal
import sys
from dataclasses import dataclass

from full_pipeline import run_full_cycle

LOGGER = logging.getLogger(__name__)
_STOP = False


@dataclass(frozen=True)
class ServerConfig:
    device: str
    tokenizer: str
    max_bpe_len: int


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    fmt = "[%(levelname)s] - %(message)s - [%(filename)s:%(lineno)d]" if verbose else "%(message)s"
    logging.basicConfig(level=level, format=fmt)


def extract_code_from_text(text: str) -> str:
    matches = re.findall(r"```(?:python)?\\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[0].strip()
    return text.strip()


def _handle_signal(_: int, __: object) -> None:
    global _STOP
    _STOP = True


def serve_jsonl(cfg: ServerConfig) -> None:
    for line in sys.stdin:
        if _STOP:
            break
        payload = json.loads(line)
        prompt = str(payload.get("prompt", ""))
        artifacts = run_full_cycle(prompt, tokenizer_name=cfg.tokenizer, max_bpe_len=cfg.max_bpe_len, device=cfg.device)
        text = artifacts.decoded_text[0] if artifacts.decoded_text else ""
        sys.stdout.write(json.dumps({"code": extract_code_from_text(text)}) + "\n")
        sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Persistent stdin->code adapter for benchmark_eval")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--max-bpe-len", type=int, default=128)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    serve_jsonl(ServerConfig(device=args.device, tokenizer=args.tokenizer, max_bpe_len=args.max_bpe_len))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
