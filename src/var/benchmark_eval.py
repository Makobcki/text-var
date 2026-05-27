"""Benchmark evaluator for HumanEval/MBPP style datasets."""

from __future__ import annotations

import argparse
import atexit
import json
import logging
import math
import os
import re
import subprocess
import tempfile
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

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


def _get_required_field(
    record: dict[str, object], field_names: Sequence[str], line_number: int
) -> str:
    """Get a required string field from a dataset record.

    Args:
        record: Parsed JSON object for one dataset line.
        field_names: Ordered candidate keys to resolve.
        line_number: One-based source line number for error context.

    Returns:
        Resolved field as string.

    Raises:
        ValueError: If none of the candidate keys are present.
    """
    for field_name in field_names:
        if field_name in record:
            return str(record[field_name])
    raise ValueError(f"Missing required field(s) {field_names} at dataset line {line_number}")


def _normalize_test_code(test_payload: object) -> str:
    """Normalize test payload into executable Python test code.

    Args:
        test_payload: Raw test payload from dataset record.

    Returns:
        Python source code containing tests.

    Raises:
        ValueError: If no test code could be extracted.
    """
    if isinstance(test_payload, str):
        return test_payload
    if isinstance(test_payload, list):
        normalized_lines = [str(line) for line in test_payload if str(line).strip()]
        if not normalized_lines:
            raise ValueError("Test payload list is empty")
        return "\n".join(normalized_lines)
    raise ValueError(f"Unsupported test payload type: {type(test_payload).__name__}")


def _extract_entry_point_from_prompt(prompt: str) -> str | None:
    """Extract function entry point from prompt text.

    Args:
        prompt: Prompt source that may contain a Python function definition.

    Returns:
        Function name when found, otherwise None.
    """
    match = re.search(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", prompt)
    if not match:
        return None
    return match.group(1)


def _build_problem(record: dict[str, object], line_number: int) -> BenchmarkProblem:
    """Build a benchmark problem from a raw dataset record.

    Args:
        record: Parsed JSON object for one dataset line.
        line_number: One-based source line number for error context.

    Returns:
        Normalized benchmark problem.

    Raises:
        ValueError: If required fields are missing or malformed.
    """
    prompt = _get_required_field(record, ["prompt", "text", "question"], line_number)
    task_id: str
    if any(key in record for key in ("task_id", "id", "problem_id", "name")):
        task_id = _get_required_field(record, ["task_id", "id", "problem_id", "name"], line_number)
    else:
        task_id = f"line_{line_number}"

    test_payload = next(
        (record[key] for key in ("test", "test_code", "test_list", "tests") if key in record), None
    )
    if test_payload is None:
        raise ValueError(f"Missing required test field at dataset line {line_number}")
    test_code = _normalize_test_code(test_payload)

    if any(key in record for key in ("entry_point", "function_name", "fn_name")):
        entry_point = _get_required_field(
            record, ["entry_point", "function_name", "fn_name"], line_number
        )
    else:
        inferred_entry_point = _extract_entry_point_from_prompt(prompt)
        if inferred_entry_point is None:
            raise ValueError(
                f"Missing entry point and could not infer from prompt at dataset line {line_number}"
            )
        entry_point = inferred_entry_point

    return BenchmarkProblem(
        task_id=task_id, prompt=prompt, test_code=test_code, entry_point=entry_point
    )


def load_jsonl_problems(dataset_path: Path) -> list[BenchmarkProblem]:
    """Load HumanEval/MBPP-style tasks from JSONL.

    Args:
        dataset_path: Path to dataset file.

    Returns:
        Parsed benchmark problems.

    Raises:
        ValueError: If a record misses required schema fields.
    """
    problems: list[BenchmarkProblem] = []
    with dataset_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            raw = json.loads(line)
            problems.append(_build_problem(raw, line_number))
    return problems


def estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """Estimate unbiased pass@k metric.

    Args:
        num_samples: Total generated samples for a task.
        num_correct: Number of passing samples.
        k: Number of attempts.

    Returns:
        pass@k estimate in [0, 1].
    """
    if num_samples <= 0 or k <= 0:
        return 0.0
    if num_samples - num_correct < k:
        return 1.0
    return 1.0 - math.comb(num_samples - num_correct, k) / math.comb(num_samples, k)


def run_candidate_in_sandbox(
    candidate_code: str,
    test_code: str,
    timeout_seconds: float = 3.0,
    sandbox_backend: str = "docker",
) -> bool:
    """Validate a candidate in a subprocess sandbox.

    Args:
        candidate_code: Generated source code.
        test_code: Unit tests for the task.
        timeout_seconds: Maximum execution time.
        sandbox_backend: Execution backend ("docker" or "host").

    Returns:
        True if execution succeeded with code 0, False otherwise.
    """
    script = "\n".join([candidate_code, test_code])
    with tempfile.TemporaryDirectory(prefix="eval_sandbox_") as tmp_dir:
        path = Path(tmp_dir) / "runner.py"
        path.write_text(script, encoding="utf-8")
        if sandbox_backend == "docker":
            run = subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--network",
                    "none",
                    "--cpus",
                    "1",
                    "--memory",
                    "256m",
                    "--pids-limit",
                    "64",
                    "--read-only",
                    "--tmpfs",
                    "/tmp:rw,noexec,nosuid,size=64m",
                    "-v",
                    f"{tmp_dir}:/work:ro",
                    "-w",
                    "/work",
                    "python:3.12-alpine",
                    "python",
                    "-I",
                    "runner.py",
                ],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        elif sandbox_backend == "host":
            run = subprocess.run(
                [os.sys.executable, "-I", str(path)],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                cwd=tmp_dir,
                check=False,
            )
        else:
            raise ValueError(f"Unsupported sandbox backend: {sandbox_backend}")
        return run.returncode == 0


def _oneshot_sampler(command: str) -> Callable[[BenchmarkProblem, int], Iterable[str]]:
    """Build per-call command sampler.

    The command receives prompt on stdin and must print generated code to stdout.

    Args:
        command: Shell command for model invocation.

    Returns:
        Sampler callable.
    """

    def sampler(problem: BenchmarkProblem, n_samples: int) -> Iterable[str]:
        for _ in range(n_samples):
            run = subprocess.run(
                command,
                input=problem.prompt,
                text=True,
                shell=True,
                capture_output=True,
                env=os.environ.copy(),
                check=False,
            )
            yield extract_code_from_text(run.stdout)

    return sampler


def _jsonl_persistent_sampler(command: str) -> Callable[[BenchmarkProblem, int], Iterable[str]]:
    """Build persistent JSONL sampler.

    Args:
        command: Command that runs a long-lived JSONL server.

    Returns:
        Sampler callable.
    """
    proc = subprocess.Popen(
        command,
        shell=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    def _cleanup_proc() -> None:
        if proc.poll() is None:
            proc.terminate()

    atexit.register(_cleanup_proc)

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

    def close() -> None:
        """Stop the persistent model process."""
        if proc.poll() is None:
            proc.terminate()

    sampler.close = close
    return sampler


def _openai_api_sampler(
    base_url: str, model_name: str
) -> Callable[[BenchmarkProblem, int], Iterable[str]]:
    """Build a sampler that makes HTTP POST requests to an OpenAI-compatible API.

    Args:
        base_url: The server address (e.g., "http://localhost:8000")
        model_name: The model ID to request.

    Returns:
        Sampler callable.
    """

    def sampler(problem: BenchmarkProblem, n_samples: int) -> Iterable[str]:
        for _ in range(n_samples):
            req_data = json.dumps(
                {
                    "model": model_name,
                    "prompt": problem.prompt,
                    "max_tokens": 1024,
                    "temperature": 0.2,
                }
            ).encode("utf-8")

            req = urllib.request.Request(
                f"{base_url.rstrip('/')}/v1/completions",
                data=req_data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            try:
                with urllib.request.urlopen(req, timeout=120) as response:
                    resp_data = json.loads(response.read().decode("utf-8"))
                    text = resp_data["choices"][0]["text"]
                    yield extract_code_from_text(text)
            except Exception as e:
                LOGGER.error("API Error: %s", e)
                yield ""

    return sampler


def evaluate_pass_at_k(
    problems: Sequence[BenchmarkProblem],
    sampler: Callable[[BenchmarkProblem, int], Iterable[str]],
    n_samples: int,
    k_values: Sequence[int],
    timeout_seconds: float = 3.0,
    sandbox_backend: str = "docker",
) -> dict[int, float]:
    """Evaluate benchmark and aggregate pass@k metrics.

    Args:
        problems: Task list to evaluate.
        sampler: Function generating code candidates for each task.
        n_samples: Number of candidates per task.
        k_values: Sequence of k values for pass@k.
        timeout_seconds: Sandbox timeout per candidate.
        sandbox_backend: Execution backend ("docker" or "host").

    Returns:
        Dictionary of k -> mean pass@k across tasks.
    """
    totals: dict[int, float] = {k: 0.0 for k in k_values}
    for problem in problems:
        correct = 0
        for code in sampler(problem, n_samples):
            correct += int(
                run_candidate_in_sandbox(
                    code,
                    problem.test_code,
                    timeout_seconds=timeout_seconds,
                    sandbox_backend=sandbox_backend,
                )
            )
        for k in k_values:
            totals[k] += estimate_pass_at_k(n_samples, correct, k)
    count = len(problems) if problems else 1
    return {k: v / count for k, v in totals.items()}


def main() -> None:
    """CLI entrypoint for pass@k benchmark evaluation."""
    parser = argparse.ArgumentParser(description="HumanEval/MBPP pass@k evaluator")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to JSONL dataset")
    parser.add_argument("--model-cmd", type=str, required=True, help="Command or URL for the model")
    parser.add_argument(
        "--model-protocol", choices=["oneshot", "jsonl", "openai"], default="openai"
    )
    parser.add_argument(
        "--model-name", type=str, default="text-var", help="Model ID for OpenAI API"
    )
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 10])
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--sandbox-backend", choices=["docker", "host"], default="docker")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    configure_logging(args.verbose)
    problems = load_jsonl_problems(args.dataset)

    if args.model_protocol == "oneshot":
        sampler = _oneshot_sampler(args.model_cmd)
    elif args.model_protocol == "jsonl":
        sampler = _jsonl_persistent_sampler(args.model_cmd)
    elif args.model_protocol == "openai":
        sampler = _openai_api_sampler(args.model_cmd, args.model_name)

    print(
        json.dumps(
            evaluate_pass_at_k(
                problems,
                sampler,
                args.n_samples,
                args.k,
                args.timeout,
                sandbox_backend=args.sandbox_backend,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
