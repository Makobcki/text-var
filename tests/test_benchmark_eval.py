from __future__ import annotations

import pytest
from pathlib import Path

from src.var.benchmark_eval import extract_code_from_text, estimate_pass_at_k, evaluate_pass_at_k, BenchmarkProblem, _jsonl_persistent_sampler, load_jsonl_problems


def test_extract_code_from_markdown() -> None:
    text = """Here:\n```python\ndef add(a,b):\n    return a+b\n```"""
    assert extract_code_from_text(text) == "def add(a,b):\n    return a+b"


def test_estimate_pass_at_k_known_values() -> None:
    assert estimate_pass_at_k(10, 3, 1) == pytest.approx(0.3)


def test_evaluate_pass_at_k_end_to_end() -> None:
    problem = BenchmarkProblem("mbpp_1", "def mul(a,b):", "assert mul(3, 4) == 12", "mul")

    def sampler(_: BenchmarkProblem, __: int) -> list[str]:
        return ["def mul(a,b):\n    return a*b"]

    metrics = evaluate_pass_at_k([problem], sampler=sampler, n_samples=1, k_values=[1], sandbox_backend="host")
    assert metrics[1] == 1.0


@pytest.mark.skip(reason="Persistent process lifecycle is environment-sensitive in CI")
def test_jsonl_persistent_sampler() -> None:
    command = "python -c \"import sys,json; [sys.stdout.write(json.dumps({'code':'def f():\\n    return 1'})+'\\n') or sys.stdout.flush() for _ in sys.stdin]\""
    sampler = _jsonl_persistent_sampler(command)
    problem = BenchmarkProblem("t", "def f():", "assert f()==1", "f")
    out = list(sampler(problem, 2))
    assert len(out) == 2
    assert out[0].startswith("def f")
    close_fn = getattr(sampler, "close", None)
    if callable(close_fn):
        close_fn()


def test_load_jsonl_problems_supports_mbpp_alias_fields(tmp_path: Path) -> None:
    dataset = tmp_path / "mbpp.jsonl"
    dataset.write_text(
        '{"id": 1, "prompt": "def add(a,b):", "test": "assert add(1,2)==3", "entry_point": "add"}\n',
        encoding="utf-8",
    )

    problems = load_jsonl_problems(dataset)

    assert len(problems) == 1
    assert problems[0].task_id == "1"


def test_load_jsonl_problems_missing_required_field_raises_value_error(tmp_path: Path) -> None:
    dataset = tmp_path / "broken.jsonl"
    dataset.write_text('{"prompt": "def f():", "entry_point": "f"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Missing required test field"):
        load_jsonl_problems(dataset)


def test_load_jsonl_problems_infers_entry_point_and_task_id(tmp_path: Path) -> None:
    dataset = tmp_path / "mbpp_no_entry.jsonl"
    dataset.write_text(
        '{"prompt": "def solve(x):", "test_list": ["assert solve(2)==2"]}\n',
        encoding="utf-8",
    )

    problems = load_jsonl_problems(dataset)

    assert problems[0].task_id == "line_1"
    assert problems[0].entry_point == "solve"
    assert problems[0].test_code == "assert solve(2)==2"
