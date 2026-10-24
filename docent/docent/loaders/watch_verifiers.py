"""Keep Docent up to date with the verifiers runs under some directories.

Each run directory (`<root>/outputs/<run>/traces.jsonl`, as `vf-eval` writes it)
gets one collection, and each new episode line its agent runs, as the line is
written. Usage:

    python -m docent.loaders.watch_verifiers ROOT [ROOT ...]
        [--server-url URL] [--web-url URL] [--interval SECONDS] [--once]

with the API key in `DOCENT_API_KEY`.

Ids are derived, not random: a run directory's collection id from its path, and an
agent run's id from its collection, episode and trace. So a restart, or
`vf-eval --resume` rewriting `traces.jsonl`, adds nothing twice: on first sight of
a collection the watcher asks Docent which runs it already holds and skips those.

Files are polled rather than watched, which needs no dependency and costs one
`stat` per run per pass. Only complete lines are read: a line still being written
is picked up on a later pass.
"""

import argparse
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

from docent._log_util.logger import get_logger
from docent.data_models import AgentRun
from docent.loaders.load_verifiers import TRACES_FILE, load_verifiers_episode, run_facts

logger = get_logger(__name__)

# why: a fixed namespace, so the same path or trace always maps to the same id.
NAMESPACE = uuid.UUID("5e3a1e55-0d0c-4e57-9a1b-7e4a11ab0c01")


class DocentClient(Protocol):
    """The calls the watcher makes; `docent.Docent` provides them."""

    def list_collections(self) -> list[dict[str, Any]]: ...

    def create_collection(
        self,
        collection_id: str | None = None,
        name: str | None = None,
        description: str | None = None,
    ) -> str: ...

    def list_agent_run_ids(self, collection_id: str) -> list[str]: ...

    def add_agent_runs(self, collection_id: str, agent_runs: list[AgentRun]) -> dict[str, Any]: ...


@dataclass
class _Tail:
    """How far one `traces.jsonl` has been read."""

    inode: int = -1
    offset: int = 0
    size: int = -1
    mtime_ns: int = -1


@dataclass
class _Collection:
    id: str
    known: set[str] = field(default_factory=set[str])


def collection_id(run_dir: Path) -> str:
    return str(uuid.uuid5(NAMESPACE, str(run_dir.resolve())))


def agent_run_id(collection: str, run: AgentRun) -> str:
    episode = run.metadata.get("episode_id", "")
    trace = run.metadata.get("trace_id", "")
    return str(uuid.uuid5(NAMESPACE, f"{collection}/{episode}/{trace}"))


class Watcher:
    def __init__(self, client: DocentClient, roots: list[Path]):
        self.client = client
        self.roots = roots
        self._tails: dict[Path, _Tail] = {}
        self._collections: dict[Path, _Collection] = {}
        self._existing: set[str] | None = None

    def run_dirs(self) -> list[Path]:
        """Run directories under the roots: a root that is itself a run, or the
        runs in its `outputs/`."""
        found: list[Path] = []
        for root in self.roots:
            if (root / TRACES_FILE).is_file():
                found.append(root)
            found.extend(sorted(p.parent for p in root.glob(f"outputs/*/{TRACES_FILE}")))
        return found

    def poll(self) -> int:
        """One pass over every run. Returns how many agent runs were added."""
        added = 0
        for run_dir in self.run_dirs():
            try:
                added += self._poll_run(run_dir)
            except OSError as e:
                logger.warning(f"{run_dir}: {e}")
        return added

    def _poll_run(self, run_dir: Path) -> int:
        path = run_dir / TRACES_FILE
        stat = path.stat()
        tail = self._tails.setdefault(path, _Tail())
        if (stat.st_ino, stat.st_size, stat.st_mtime_ns) == (tail.inode, tail.size, tail.mtime_ns):
            return 0
        if stat.st_ino != tail.inode or stat.st_size < tail.offset:
            # A new file, or one rewritten in place (`--resume` writes a temporary
            # file and renames it over): read it again from the start. Ids already
            # in Docent are skipped below.
            tail.offset = 0
        with open(path, "rb") as f:
            f.seek(tail.offset)
            chunk = f.read()
        end = chunk.rfind(b"\n") + 1
        lines = chunk[:end].decode("utf-8", errors="replace").splitlines()
        tail.inode, tail.size, tail.mtime_ns = stat.st_ino, stat.st_size, stat.st_mtime_ns
        tail.offset += end
        if not lines:
            return 0

        collection = self._collection(run_dir)
        facts = run_facts(run_dir)
        fresh: list[AgentRun] = []
        for line in lines:
            if not line.strip():
                continue
            try:
                episode: Any = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"{path}: unreadable line, skipped")
                continue
            if not isinstance(episode, dict):
                continue
            for run in load_verifiers_episode(cast(dict[str, Any], episode), facts):
                run.id = agent_run_id(collection.id, run)
                if run.id not in collection.known:
                    collection.known.add(run.id)
                    fresh.append(run)
        if fresh:
            self.client.add_agent_runs(collection.id, fresh)
            logger.info(f"{run_dir.name}: added {len(fresh)} agent runs")
        return len(fresh)

    def _collection(self, run_dir: Path) -> _Collection:
        if run_dir in self._collections:
            return self._collections[run_dir]
        cid = collection_id(run_dir)
        if self._existing is None:
            self._existing = {str(c.get("id")) for c in self.client.list_collections()}
        if cid in self._existing:
            known: set[str] = set(self.client.list_agent_run_ids(cid))
        else:
            self.client.create_collection(
                collection_id=cid, name=run_dir.name, description=str(run_dir.resolve())
            )
            self._existing.add(cid)
            known = set[str]()
        collection = _Collection(cid, known)
        self._collections[run_dir] = collection
        return collection


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Keep Docent up to date with the verifiers runs under some directories."
    )
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--server-url", default="http://localhost:8889")
    parser.add_argument("--web-url", default="http://localhost:3001")
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--once", action="store_true", help="one pass, then exit")
    args = parser.parse_args()

    from docent import Docent

    client = Docent(
        api_key=os.environ.get("DOCENT_API_KEY"), server_url=args.server_url, web_url=args.web_url
    )
    watcher = Watcher(client, args.roots)
    while True:
        watcher.poll()
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
