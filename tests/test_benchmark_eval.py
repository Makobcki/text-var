from __future__ import annotations

from src.var.benchmark_eval import extract_code_from_text, estimate_pass_at_k, evaluate_pass_at_k, BenchmarkProblem, _jsonl_persistent_sampler


def test_extract_code_from_markdown() -> None:
    text = """Here:\n```python\ndef add(a,b):\n    return a+b\n```"""
    assert extract_code_from_text(text) == "def add(a,b):\n    return a+b"


def test_estimate_pass_at_k_known_values() -> None:
    assert estimate_pass_at_k(10, 3, 1) == 0.3


def test_evaluate_pass_at_k_end_to_end() -> None:
    problem = BenchmarkProblem("mbpp_1", "def mul(a,b):", "assert mul(3, 4) == 12", "mul")

    def sampler(_: BenchmarkProblem, __: int) -> list[str]:
        return ["def mul(a,b):\n    return a*b"]

    metrics = evaluate_pass_at_k([problem], sampler=sampler, n_samples=1, k_values=[1])
    assert metrics[1] == 1.0


def test_jsonl_persistent_sampler() -> None:
    command = "python -c \"import sys,json; [sys.stdout.write(json.dumps({'code':'def f():\\n    return 1'})+'\\n') or sys.stdout.flush() for _ in sys.stdin]\""
    sampler = _jsonl_persistent_sampler(command)
    problem = BenchmarkProblem("t", "def f():", "assert f()==1", "f")
    out = list(sampler(problem, 2))
    assert len(out) == 2
    assert out[0].startswith("def f")
