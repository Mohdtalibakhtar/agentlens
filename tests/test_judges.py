"""Tests for judge backends."""

from __future__ import annotations

from agentlens.judges.fake import FakeJudge


def test_fake_judge_returns_static_response() -> None:
    judge = FakeJudge(response="hello")
    assert judge.query("sys", "user") == "hello"


def test_fake_judge_records_calls() -> None:
    judge = FakeJudge(response="ok")
    judge.query("system A", "user 1")
    judge.query("system A", "user 2")
    assert judge.calls == [("system A", "user 1"), ("system A", "user 2")]


def test_fake_judge_supports_callable_response() -> None:
    judge = FakeJudge(response=lambda system, user: f"echo:{user}")
    assert judge.query("sys", "hi") == "echo:hi"


def test_fake_judge_default_response_is_valid_json() -> None:
    import json

    judge = FakeJudge()
    parsed = json.loads(judge.query("sys", "user"))
    assert parsed["on_topic"] is True
