"""Unit tests for the verifiers v1 loader."""

import json
from pathlib import Path
from typing import Any

import pytest

from docent.data_models.chat import AssistantMessage, ContentReasoning, ToolMessage
from docent.loaders.load_verifiers import load_verifiers_episode, load_verifiers_run


def _trace(**over: Any) -> dict[str, Any]:
    trace: dict[str, Any] = {
        "version": 1,
        "id": "t1",
        "verifiers": {"version": "0.4.0"},
        "agent": {
            "name": "agent",
            "config": {"model": "qwen3-8b", "harness": {"id": "bash", "version": "1"}},
            "runtime": {"type": "docker", "borrowed": False},
            "trainable": True,
        },
        # A branched graph: node 3 is an abandoned reply to node 1, and the last node
        # (4) continues from node 2, so the last branch is 0, 1, 2, 4.
        "nodes": [
            {"message": {"role": "system", "content": "You add numbers."}},
            {"parent": 0, "message": {"role": "user", "content": "What is 2+2?"}},
            {
                "parent": 1,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "Use the calculator.",
                    "tool_calls": [
                        {"id": "c1", "type": "function", "name": "calc", "arguments": "{\"x\": 2"}
                    ],
                },
            },
            {"parent": 1, "message": {"role": "assistant", "content": "abandoned"}},
            {
                "parent": 2,
                "message": {
                    "role": "tool",
                    "tool_call_id": "c1",
                    "name": "calc",
                    "content": [{"type": "text", "text": "4"}, {"type": "image_url", "image_url": {"url": "data:"}}],
                },
            },
        ],
        "calls": [
            {"usage": {"prompt_tokens": 10, "completion_tokens": 5, "cached_input_tokens": 3, "cost": 0.01}},
            {"usage": {"prompt_tokens": 12, "completion_tokens": 2, "cost": 0.02}},
        ],
        "rewards": {"exact": {"score": 1.0, "weight": 0.5}, "format": {"score": 0.5, "weight": 1.0}},
        "metrics": {"turns": 2.0},
        "ok": True,
        "is_timeout": False,
        "stop_condition": "agent_completed",
        "errors": [],
        "timing": {
            "start": 100.0,
            "boot": {"start": 100.0, "end": 101.0},
            "agent": {"start": 101.0, "end": 111.0, "model": {"duration": 8.0}, "harness": {"duration": 2.0}},
            "scoring": {"start": 111.0, "end": 111.5},
        },
        "num_input_tokens": 25,
        "num_output_tokens": 7,
        "num_total_tokens": 32,
    }
    trace.update(over)
    return trace


def _episode(**over: Any) -> dict[str, Any]:
    episode: dict[str, Any] = {
        "id": "e1",
        "env": {"id": "gsm8k"},
        "task": {"type": "task", "data": {"idx": 3, "name": "add-2"}, "key": "k3", "hash": "h3"},
        "run": {"type": "eval", "id": "r1", "name": "gsm8k--qwen3--bash"},
        "ok": True,
        "is_timeout": False,
        "errors": [],
        "traces": [_trace()],
    }
    episode.update(over)
    return episode


@pytest.mark.unit
def test_a_trace_carries_what_verifiers_records():
    [run] = load_verifiers_episode(_episode())
    m = run.metadata
    assert run.name == "add-2 · bash · qwen3-8b"
    assert m["env"] == "gsm8k"
    assert m["task"] == {"key": "k3", "idx": 3, "name": "add-2"}
    assert (m["harness"], m["model"], m["runtime"]) == ("bash", "qwen3-8b", "docker")
    # verifiers does not store the reward; it is sum(score * weight).
    assert m["scores"] == {"reward": 1.0, "exact": 1.0, "format": 0.5}
    assert m["rewards"]["exact"] == {"score": 1.0, "weight": 0.5}
    assert m["stop_condition"] == "agent_completed"
    assert m["tokens"] == {"input": 25, "output": 7, "total": 32, "cached": 3}
    assert m["cost_usd"] == pytest.approx(0.03)
    assert m["timing_s"] == {"boot": 1.0, "agent": 10.0, "scoring": 0.5, "agent_model": 8.0, "agent_harness": 2.0}
    assert m["run"] == {"type": "eval", "id": "r1", "name": "gsm8k--qwen3--bash"}
    assert "error_type" not in m


@pytest.mark.unit
def test_the_transcript_is_the_last_branch_with_reasoning_and_tool_calls():
    [run] = load_verifiers_episode(_episode())
    messages = run.transcripts[0].messages
    assert [m.role for m in messages] == ["system", "user", "assistant", "tool"]
    assistant = messages[2]
    assert isinstance(assistant, AssistantMessage)
    assert isinstance(assistant.content, list) and isinstance(assistant.content[0], ContentReasoning)
    [call] = assistant.tool_calls or []
    # Arguments that are not JSON stay readable, with the parse error kept.
    assert call.function == "calc" and call.arguments == {"raw": "{\"x\": 2"}
    assert call.parse_error
    tool = messages[3]
    assert isinstance(tool, ToolMessage) and tool.tool_call_id == "c1" and tool.function == "calc"
    assert [c.text for c in tool.content] == ["4", "[image]"]  # type: ignore[union-attr]


@pytest.mark.unit
def test_an_errored_trace_says_how_it_ended():
    trace = _trace(
        ok=False,
        stop_condition="error",
        rewards={},
        errors=[{"type": "SandboxError", "message": "boot failed"}],
    )
    [run] = load_verifiers_episode(_episode(ok=False, traces=[trace]))
    m = run.metadata
    assert (m["ok"], m["error_type"], m["error_message"]) == (False, "SandboxError", "boot failed")
    assert m["scores"] == {}


@pytest.mark.unit
def test_an_episode_without_traces_still_shows_its_error():
    [run] = load_verifiers_episode(
        _episode(ok=False, traces=[], errors=[{"type": "Timeout", "message": "never started"}])
    )
    assert run.transcripts[0].messages == []
    assert run.metadata["error_type"] == "Timeout"


@pytest.mark.unit
def test_a_linear_graph_without_parents_is_read_in_order():
    nodes = [
        {"message": {"role": "user", "content": "hi"}},
        {"message": {"role": "assistant", "content": "hello"}},
    ]
    [run] = load_verifiers_episode(_episode(traces=[_trace(nodes=nodes)]))
    assert [m.text for m in run.transcripts[0].messages] == ["hi", "hello"]


@pytest.mark.unit
def test_a_run_directory_skips_a_torn_last_line_and_takes_its_name(tmp_path: Path):
    (tmp_path / "configs" / "resolved").mkdir(parents=True)
    (tmp_path / "configs" / "resolved" / "eval.json").write_text(json.dumps({"run": {"name": "my-run"}}))
    lines = [json.dumps(_episode()), json.dumps(_episode(id="e2")), "{\"id\": \"e3\", \"tra"]
    (tmp_path / "traces.jsonl").write_text("\n".join(lines))
    runs = list(load_verifiers_run(tmp_path))
    assert [r.metadata["episode_id"] for r in runs] == ["e1", "e2"]
    assert runs[0].metadata["run"]["name"] == "my-run"
    assert runs[0].metadata["run"]["dir"] == tmp_path.name
