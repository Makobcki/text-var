"""Benchmark evaluator for HumanEval/MBPP style datasets."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchmarkProblem:
    """Represents a single coding benchmark task.

    Args:
        task_id: Unique task identifier.
        prompt: Function signature and docstring prompt for generation.
        test_code: Python test code asserting correctness.
        entry_point: Function name to call in tests.
    """

    task_id: str
    prompt: str
    test_code: str
    entry_point: str


def extract_code_from_text(text: str) -> str:
    """Extract pure Python code from raw model text.

    Args:
        text: Raw model output that may include Markdown fences.

    Returns:
        Cleaned Python code.
    """

    matches = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[0].strip()
    return text.strip()


def configure_logging(verbose: bool) -> None:
    """Configure application logging.

    Args:
        verbose: Enables debug logs when True.
    """

    level = logging.DEBUG if verbose else logging.WARNING
    fmt = "[%(levelname)s] - %(message)s - [%(filename)s:%(lineno)d]" if verbose else "%(message)s"
    logging.basicConfig(level=level, format=fmt)


def load_jsonl_problems(dataset_path: Path) -> list[BenchmarkProblem]:
    """Load HumanEval/MBPP-style tasks from JSONL.

    Args:
        dataset_path: Path to dataset file.

    Returns:
        Parsed benchmark problems.
    """

    problems: list[BenchmarkProblem] = []
    with dataset_path.open("r", encoding="utf-8") as file:
        for line in file:
            raw = json.loads(line)
            problems.append(BenchmarkProblem(str(raw["task_id"]), str(raw["prompt"]), str(raw["test"]), str(raw["entry_point"])))
    return problems


def estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """Estimate unbiased pass@k metric."""

    if num_samples <= 0 or k <= 0:
        return 0.0
    if num_samples - num_correct < k:
        return 1.0
    return 1.0 - math.comb(num_samples - num_correct, k) / math.comb(num_samples, k)


def run_candidate_in_sandbox(candidate_code: str, test_code: str, timeout_seconds: float = 3.0) -> bool:
    """Validate a candidate in a subprocess sandbox."""

    script = "\n".join([candidate_code, test_code])
    with tempfile.TemporaryDirectory(prefix="eval_sandbox_") as tmp_dir:
        path = Path(tmp_dir) / "runner.py"
        path.write_text(script, encoding="utf-8")
        run = subprocess.run([os.sys.executable, "-I", str(path)], capture_output=True, text=True, timeout=timeout_seconds, cwd=tmp_dir, check=False)
        return run.returncode == 0


def _oneshot_sampler(command: str) -> Callable[[BenchmarkProblem, int], Iterable[str]]:
    """Build per-call command sampler."""

    def sampler(problem: BenchmarkProblem, n_samples: int) -> Iterable[str]:
        for _ in range(n_samples):
            run = subprocess.run(command, input=problem.prompt, text=True, shell=True, capture_output=True, env=os.environ.copy(), check=False)
            yield extract_code_from_text(run.stdout)

    return sampler


def _jsonl_persistent_sampler(command: str) -> Callable[[BenchmarkProblem, int], Iterable[str]]:
    """Build persistent JSONL sampler.

    Args:
        command: Command that runs a long-lived JSONL server.

    Returns:
        Sampler callable.
    """

    proc = subprocess.Popen(command, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def sampler(problem: BenchmarkProblem, n_samples: int) -> Iterable[str]:
        assert proc.stdin and proc.stdout
        for _ in range(n_samples):
            proc.stdin.write(json.dumps({"prompt": problem.prompt}) + "\n")
            proc.stdin.flush()
            line = proc.stdout.readline()
            if not line:
                raise RuntimeError("Persistent model server returned empty response.")
            payload = json.loads(line)
            yield extract_code_from_text(str(payload.get("code", "")))

    return sampler


def evaluate_pass_at_k(problems: Sequence[BenchmarkProblem], sampler: Callable[[BenchmarkProblem, int], Iterable[str]], n_samples: int, k_values: Sequence[int], timeout_seconds: float = 3.0) -> dict[int, float]:
    """Evaluate benchmark and aggregate pass@k metrics."""

    totals: dict[int, float] = {k: 0.0 for k in k_values}
    for problem in problems:
        correct = 0
        for code in sampler(problem, n_samples):
            correct += int(run_candidate_in_sandbox(code, problem.test_code, timeout_seconds=timeout_seconds))
        for k in k_values:
            totals[k] += estimate_pass_at_k(n_samples, correct, k)
    count = len(problems) if problems else 1
    return {k: v / count for k, v in totals.items()}


def main() -> None:
    """CLI entrypoint for pass@k benchmark evaluation."""

    parser = argparse.ArgumentParser(description="HumanEval/MBPP pass@k evaluator")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-cmd", type=str, required=True)
    parser.add_argument("--model-protocol", choices=["oneshot", "jsonl"], default="oneshot")
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 10])
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    configure_logging(args.verbose)
    problems = load_jsonl_problems(args.dataset)
    sampler = _oneshot_sampler(args.model_cmd) if args.model_protocol == "oneshot" else _jsonl_persistent_sampler(args.model_cmd)
    print(json.dumps(evaluate_pass_at_k(problems, sampler, args.n_samples, args.k, args.timeout), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
