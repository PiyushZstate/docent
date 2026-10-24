"""Load verifiers v1 eval runs (`vf-eval` output) as Docent agent runs.

A run directory holds `traces.jsonl`, one episode per line. An episode has one trace
per agent, and each trace becomes one `AgentRun`: its transcript is the message
graph's last branch, and its metadata carries what verifiers records about the
rollout, so Docent can filter and chart by it:

- `scores`: the reward (`sum(score * weight)`, which verifiers does not store) and
  each named reward's raw score.
- `task`, `env`, `harness`, `model`, `runtime`, `agent`: where it ran.
- `ok`, `is_timeout`, `stop_condition`, `error_type`, `error_message`: how it ended.
- `rewards` (score and weight), `metrics`, `tokens`, `cost_usd`, `timing_s`.
- `run`, `episode_id`, `trace_id`, `verifiers_version`: where it came from.

Field meanings follow verifiers' `episode.py`, `trace.py` and `graph.py`. The file
is written with `exclude_none`, so an absent key means unset, and its records are
open: unknown fields are ignored, never an error.
"""

import json
from pathlib import Path
from typing import Any, Iterator, cast

from docent._log_util.logger import get_logger
from docent.data_models import AgentRun, Transcript
from docent.data_models.chat import (
    ChatMessage,
    Content,
    ContentReasoning,
    ContentText,
    ToolCall,
    parse_chat_message,
)

logger = get_logger(__name__)

TRACES_FILE = "traces.jsonl"


def _obj(value: Any) -> dict[str, Any]:
    """A JSON object, or an empty one when the value is missing or not an object."""
    return cast(dict[str, Any], value) if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    """A JSON array, or an empty one when the value is missing or not an array."""
    return cast(list[Any], value) if isinstance(value, list) else []


def load_verifiers_run(run_dir: str | Path) -> Iterator[AgentRun]:
    """Every trace of a `vf-eval` run directory, as agent runs.

    Reads `traces.jsonl` line by line, since lines carry token streams and a run can
    be gigabytes. A line that does not parse (a torn last line while a run is still
    writing, or a malformed row) is skipped with a warning, as verifiers' own resume
    does.
    """
    run_dir = Path(run_dir)
    run = _run_facts(run_dir)
    with open(run_dir / TRACES_FILE, encoding="utf-8") as f:
        for number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                episode: Any = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"{run_dir / TRACES_FILE}:{number}: unreadable line, skipped")
                continue
            if isinstance(episode, dict):
                yield from load_verifiers_episode(_obj(episode), run)


def load_verifiers_episode(
    episode: dict[str, Any], run: dict[str, Any] | None = None
) -> list[AgentRun]:
    """One episode line's traces as agent runs. An episode with no traces (the task
    was dispatched and produced none) becomes one run with an empty transcript, so
    its error still shows."""
    run = {**_obj(episode.get("run")), **(run or {})}
    traces = [_obj(t) for t in _list(episode.get("traces"))]
    if not traces:
        metadata = _episode_metadata(episode, run)
        return [
            AgentRun(
                name=_name(metadata), transcripts=[Transcript(messages=[])], metadata=metadata
            )
        ]
    return [_trace_run(episode, trace, run) for trace in traces]


def _last_error(record: dict[str, Any]) -> dict[str, Any]:
    errors = _list(record.get("errors"))
    return _obj(errors[-1]) if errors else {}


def _trace_run(episode: dict[str, Any], trace: dict[str, Any], run: dict[str, Any]) -> AgentRun:
    metadata = _episode_metadata(episode, run)
    agent = _obj(trace.get("agent"))
    config = _obj(agent.get("config"))
    harness = _obj(config.get("harness"))

    rewards = {
        str(name): _obj(r)
        for name, r in _obj(trace.get("rewards")).items()
        if isinstance(r, dict)
    }
    scores: dict[str, float] = {}
    if rewards:
        scores["reward"] = sum(
            float(r.get("score") or 0.0) * float(r.get("weight", 1.0)) for r in rewards.values()
        )
        for name, r in rewards.items():
            scores[name] = float(r.get("score") or 0.0)

    error = _last_error(trace)
    metadata.update(
        {
            "trace_id": trace.get("id"),
            "verifiers_version": _obj(trace.get("verifiers")).get("version"),
            "agent": agent.get("name"),
            "harness": harness.get("id"),
            "harness_version": harness.get("version"),
            "model": config.get("model"),
            "runtime": _obj(agent.get("runtime")).get("type"),
            "ok": trace.get("ok", metadata.get("ok")),
            "is_timeout": trace.get("is_timeout", metadata.get("is_timeout")),
            "stop_condition": trace.get("stop_condition"),
            "error_type": error.get("type", metadata.get("error_type")),
            "error_message": error.get("message", metadata.get("error_message")),
            "scores": scores,
            "rewards": rewards,
            "metrics": _obj(trace.get("metrics")),
            "tokens": _tokens(trace),
            "cost_usd": _cost(trace),
            "timing_s": _timing(_obj(trace.get("timing"))),
        }
    )
    metadata = {k: v for k, v in metadata.items() if v is not None}
    return AgentRun(
        name=_name(metadata),
        transcripts=[Transcript(messages=_branch_messages(trace))],
        metadata=metadata,
    )


def _episode_metadata(episode: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    task = _obj(episode.get("task"))
    data = _obj(task.get("data"))
    error = _last_error(episode)
    identity = {
        "key": task.get("key") or task.get("hash"),
        "idx": data.get("idx"),
        "id": data.get("id"),
        "name": data.get("name"),
    }
    metadata: dict[str, Any] = {
        "episode_id": episode.get("id"),
        "env": _obj(episode.get("env")).get("id"),
        "task": {k: v for k, v in identity.items() if v is not None},
        "run": {k: v for k, v in run.items() if k in ("id", "name", "dir", "type")},
        "ok": episode.get("ok"),
        "is_timeout": episode.get("is_timeout"),
        "error_type": error.get("type"),
        "error_message": error.get("message"),
    }
    return {k: v for k, v in metadata.items() if v is not None}


def _name(metadata: dict[str, Any]) -> str:
    task = _obj(metadata.get("task"))
    where = " · ".join(str(x) for x in (metadata.get("harness"), metadata.get("model")) if x)
    label = str(task.get("name") or task.get("key") or metadata.get("episode_id") or "rollout")
    return f"{label} · {where}" if where else label


def _run_facts(run_dir: Path) -> dict[str, Any]:
    """The run's name from its resolved config, when it has one; the directory
    otherwise."""
    facts: dict[str, Any] = {"dir": run_dir.name}
    resolved = run_dir / "configs" / "resolved"
    for path in sorted(resolved.glob("*.json")) if resolved.is_dir() else []:
        try:
            config: Any = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        name = _obj(_obj(config).get("run")).get("name")
        if name:
            facts["name"] = name
            break
    return facts


def _usage(trace: dict[str, Any], key: str) -> list[Any]:
    """One usage field over the trace's model calls, `None` where a call lacks it."""
    return [_obj(_obj(c).get("usage")).get(key) for c in _list(trace.get("calls"))]


def _tokens(trace: dict[str, Any]) -> dict[str, int]:
    tokens: dict[str, Any] = {
        "input": trace.get("num_input_tokens"),
        "output": trace.get("num_output_tokens"),
        "total": trace.get("num_total_tokens"),
    }
    for name, key in (("cached", "cached_input_tokens"), ("reasoning", "reasoning_tokens")):
        values = _usage(trace, key)
        if any(v is not None for v in values):
            tokens[name] = sum(int(v or 0) for v in values)
    return {k: int(v) for k, v in tokens.items() if v is not None}


def _cost(trace: dict[str, Any]) -> float | None:
    known = [float(c) for c in _usage(trace, "cost") if c is not None]
    return sum(known) if known else None


def _timing(timing: dict[str, Any]) -> dict[str, float]:
    """Each phase's duration in seconds, for the phases that were timed."""
    out: dict[str, float] = {}
    for phase in ("boot", "setup", "agent", "finalize", "scoring"):
        span = _obj(timing.get(phase))
        start, end = span.get("start"), span.get("end")
        if start and end:
            out[phase] = round(float(end) - float(start), 3)
    agent = _obj(timing.get("agent"))
    for part in ("model", "harness"):
        duration = _obj(agent.get(part)).get("duration")
        if duration is not None:
            out[f"agent_{part}"] = round(float(duration), 3)
    return out


def _branch(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The graph's last branch, root first: from the last node up its parents. A
    graph where no node names a parent is one straight line. A parent that points
    at itself or forward (a malformed graph) ends the walk rather than looping."""
    if not nodes:
        return []
    if not any("parent" in n for n in nodes):
        return nodes
    branch: list[dict[str, Any]] = []
    at: int | None = len(nodes) - 1
    while at is not None and 0 <= at < len(nodes):
        branch.append(nodes[at])
        parent = nodes[at].get("parent")
        at = parent if isinstance(parent, int) and parent < at else None
    branch.reverse()
    return branch


def _branch_messages(trace: dict[str, Any]) -> list[ChatMessage]:
    messages: list[ChatMessage] = []
    for node in _branch([_obj(n) for n in _list(trace.get("nodes"))]):
        message = node.get("message")
        if not isinstance(message, dict):
            continue
        try:
            messages.append(_message(_obj(message)))
        except ValueError as e:
            logger.warning(f"trace {trace.get('id')}: message skipped: {e}")
    return messages


def _content(content: Any) -> str | list[Content]:
    if isinstance(content, str):
        return content
    parts: list[Content] = []
    for part in (_obj(p) for p in _list(content)):
        if part.get("type") == "text":
            parts.append(ContentText(text=str(part.get("text", ""))))
        elif part.get("type") == "image_url":
            # Images are not carried over: a data URL can be megabytes per turn.
            parts.append(ContentText(text="[image]"))
    return parts


def _arguments(raw: Any) -> tuple[dict[str, Any], str | None]:
    """A tool call's arguments as an object, from the raw string the model emitted.
    Arguments that are not JSON stay readable under `raw`, with the parse error."""
    if isinstance(raw, dict):
        return _obj(raw), None
    if not isinstance(raw, str) or not raw.strip():
        return {}, None
    try:
        parsed: Any = json.loads(raw)
    except json.JSONDecodeError as e:
        return {"raw": raw}, str(e)
    return (_obj(parsed), None) if isinstance(parsed, dict) else ({"value": parsed}, None)


def _message(message: dict[str, Any]) -> ChatMessage:
    role = message.get("role")
    content = _content(message.get("content"))
    if role == "assistant":
        reasoning = message.get("reasoning_content")
        if reasoning:
            rest: list[Content] = (
                [ContentText(text=content)] if isinstance(content, str) and content
                else [] if isinstance(content, str)
                else content
            )
            content = [ContentReasoning(reasoning=str(reasoning)), *rest]
        calls: list[ToolCall] = []
        for call in (_obj(c) for c in _list(message.get("tool_calls"))):
            arguments, parse_error = _arguments(call.get("arguments"))
            calls.append(
                ToolCall(
                    id=str(call.get("id", "")),
                    function=str(call.get("name", "")),
                    arguments=arguments,
                    type="function",
                    parse_error=parse_error,
                )
            )
        return parse_chat_message(
            {"role": "assistant", "content": content, "tool_calls": calls or None}
        )
    if role == "tool":
        return parse_chat_message(
            {
                "role": "tool",
                "content": content,
                "tool_call_id": message.get("tool_call_id"),
                "function": message.get("name"),
            }
        )
    if role in ("system", "user"):
        return parse_chat_message({"role": role, "content": content})
    raise ValueError(f"unknown role {role!r}")
