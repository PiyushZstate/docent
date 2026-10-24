"""Unit tests for the verifiers run watcher, against an in-memory Docent."""

import json
import os
from pathlib import Path
from typing import Any

import pytest

from docent.data_models import AgentRun
from docent.loaders.watch_verifiers import Watcher, collection_id


class FakeDocent:
    def __init__(self) -> None:
        self.collections: dict[str, dict[str, Any]] = {}
        self.runs: dict[str, list[AgentRun]] = {}

    def list_collections(self) -> list[dict[str, Any]]:
        return [{"id": cid, **c} for cid, c in self.collections.items()]

    def create_collection(
        self,
        collection_id: str | None = None,
        name: str | None = None,
        description: str | None = None,
    ) -> str:
        assert collection_id is not None and collection_id not in self.collections
        self.collections[collection_id] = {"name": name, "description": description}
        self.runs[collection_id] = []
        return collection_id

    def list_agent_run_ids(self, collection_id: str) -> list[str]:
        return [r.id for r in self.runs[collection_id]]

    def add_agent_runs(self, collection_id: str, agent_runs: list[AgentRun]) -> dict[str, Any]:
        ids = {r.id for r in self.runs[collection_id]}
        # Docent inserts plainly: a repeated id would fail there, so fail here too.
        assert not ids & {r.id for r in agent_runs}, "an agent run was added twice"
        self.runs[collection_id].extend(agent_runs)
        return {"status": "success"}


def _episode(i: int) -> str:
    return json.dumps(
        {
            "id": f"e{i}",
            "env": {"id": "gsm8k"},
            "task": {"key": f"t{i}", "data": {"idx": i}},
            "ok": True,
            "traces": [
                {
                    "id": f"tr{i}",
                    "agent": {"name": "agent", "config": {"model": "m", "harness": {"id": "bash"}}},
                    "nodes": [{"message": {"role": "user", "content": f"task {i}"}}],
                    "rewards": {"exact": {"score": 1.0, "weight": 1.0}},
                }
            ],
        }
    )


def _run(root: Path, name: str = "gsm8k--m--bash") -> Path:
    run = root / "outputs" / name
    run.mkdir(parents=True)
    (run / "traces.jsonl").write_text("")
    return run


def _append(run: Path, text: str) -> None:
    with open(run / "traces.jsonl", "a") as f:
        f.write(text)


@pytest.mark.unit
def test_new_lines_reach_one_collection_per_run(tmp_path: Path):
    docent = FakeDocent()
    run = _run(tmp_path)
    watcher = Watcher(docent, [tmp_path])
    _append(run, _episode(1) + "\n")
    assert watcher.poll() == 1
    _append(run, _episode(2) + "\n")
    assert watcher.poll() == 1
    assert watcher.poll() == 0
    [cid] = docent.collections
    assert cid == collection_id(run)
    assert docent.collections[cid]["name"] == "gsm8k--m--bash"
    assert [r.metadata["episode_id"] for r in docent.runs[cid]] == ["e1", "e2"]


@pytest.mark.unit
def test_a_line_still_being_written_waits_for_its_newline(tmp_path: Path):
    docent = FakeDocent()
    run = _run(tmp_path)
    watcher = Watcher(docent, [tmp_path])
    line = _episode(1)
    _append(run, line[:20])
    assert watcher.poll() == 0
    _append(run, line[20:] + "\n")
    assert watcher.poll() == 1


@pytest.mark.unit
def test_a_resume_rewrite_adds_only_the_new_episodes(tmp_path: Path):
    docent = FakeDocent()
    run = _run(tmp_path)
    watcher = Watcher(docent, [tmp_path])
    _append(run, _episode(1) + "\n" + _episode(2) + "\n")
    assert watcher.poll() == 2
    # `vf-eval --resume` keeps finished episodes, drops errored ones and renames a
    # rewritten file over the old one, here with episode 3 added.
    tmp = run / "traces.jsonl.tmp"
    tmp.write_text(_episode(1) + "\n" + _episode(3) + "\n")
    os.replace(tmp, run / "traces.jsonl")
    assert watcher.poll() == 1
    [cid] = docent.collections
    assert [r.metadata["episode_id"] for r in docent.runs[cid]] == ["e1", "e2", "e3"]


@pytest.mark.unit
def test_a_restarted_watcher_skips_what_docent_already_holds(tmp_path: Path):
    docent = FakeDocent()
    run = _run(tmp_path)
    _append(run, _episode(1) + "\n")
    assert Watcher(docent, [tmp_path]).poll() == 1
    _append(run, _episode(2) + "\n")
    assert Watcher(docent, [tmp_path]).poll() == 1
    [cid] = docent.collections
    assert len(docent.runs[cid]) == 2


@pytest.mark.unit
def test_a_root_can_be_a_run_directory_itself(tmp_path: Path):
    docent = FakeDocent()
    run = _run(tmp_path)
    _append(run, _episode(1) + "\n")
    assert Watcher(docent, [run]).poll() == 1


@pytest.mark.unit
def test_the_roots_file_is_reread_on_every_pass(tmp_path: Path):
    docent = FakeDocent()
    roots_file = tmp_path / "roots"
    watcher = Watcher(docent, [], roots_file)
    # Missing to begin with: the engine may start the watcher before writing it.
    assert watcher.poll() == 0
    first, second = tmp_path / "a", tmp_path / "b"
    _append(_run(first), _episode(1) + "\n")
    _append(_run(second), _episode(2) + "\n")
    roots_file.write_text(f"# worktrees\n{first}\n\n")
    assert watcher.poll() == 1
    roots_file.write_text(f"{first}\n{second}\n{second}\n")
    assert watcher.poll() == 1
    assert len(docent.collections) == 2


class FlakyDocent(FakeDocent):
    """Refuses adds while `down`, as Docent does while it is still starting."""

    def __init__(self) -> None:
        super().__init__()
        self.down = True

    def add_agent_runs(self, collection_id: str, agent_runs: list[AgentRun]) -> dict[str, Any]:
        if self.down:
            raise ConnectionError("Docent is not up yet")
        return super().add_agent_runs(collection_id, agent_runs)


@pytest.mark.unit
def test_lines_a_failed_add_refused_are_sent_on_the_next_pass(tmp_path: Path):
    docent = FlakyDocent()
    run = _run(tmp_path)
    watcher = Watcher(docent, [tmp_path])
    _append(run, _episode(1) + "\n")
    assert watcher.poll() == 0
    docent.down = False
    assert watcher.poll() == 1
    assert watcher.poll() == 0
    [cid] = docent.collections
    assert [r.metadata["episode_id"] for r in docent.runs[cid]] == ["e1"]
