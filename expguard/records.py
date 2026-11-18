from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Iterator


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{number} must contain a JSON object")
                yield row


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, encoding="utf-8", delete=False
    ) as target:
        try:
            for row in rows:
                target.write(json_text(row) + "\n")
            target.close()
            os.replace(target.name, path)
        finally:
            Path(target.name).unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class TextFields:
    title: str
    description: str
    content: str

    def __post_init__(self) -> None:
        if any(type(value) is not str or not value.strip() for value in (self.title, self.description, self.content)):
            raise ValueError("Title, description and content must be nonempty strings")

    def semantic_text(self) -> str:
        return "\n".join((self.title, self.description, self.content))


@dataclass(frozen=True, slots=True)
class Provenance:
    source_trajectory: str
    supporting_actions: tuple[str, ...]
    supporting_observations: tuple[str, ...]
    extraction_record: str
    lineage: str

    @classmethod
    def from_dict(cls, row: dict) -> Provenance:
        return cls(
            row["source_trajectory"],
            tuple(row["supporting_actions"]),
            tuple(row["supporting_observations"]),
            row["extraction_record"],
            row.get("lineage", row["source_trajectory"]),
        )


@dataclass(frozen=True, slots=True)
class VersionMetadata:
    number: int
    parent: int | None
    timestamp: int
    operator: str
    boundary: str | None = None


@dataclass(frozen=True, slots=True)
class Experience(TextFields):
    id: str
    provenance: Provenance
    version_metadata: VersionMetadata

    @property
    def version(self) -> int:
        return self.version_metadata.number

    @property
    def key(self) -> str:
        return json_text((self.id, self.version))

    @classmethod
    def from_dict(cls, row: dict) -> Experience:
        return cls(
            row["title"], row["description"], row["content"], row["id"],
            Provenance.from_dict(row["provenance"]), VersionMetadata(**row["version_metadata"]),
        )

    def revised(self, text: TextFields, operator: str, timestamp: int, boundary: str) -> Experience:
        if timestamp < self.version_metadata.timestamp:
            raise ValueError("Revision predates its parent")
        return replace(
            self, **asdict(text),
            version_metadata=VersionMetadata(self.version + 1, self.version, timestamp, operator, boundary),
        )

    def fields(self) -> TextFields:
        return TextFields(self.title, self.description, self.content)


@dataclass(frozen=True, slots=True)
class Task:
    id: str
    objective: str
    constraints: tuple[str, ...] = ()
    evaluation_spec: str = ""

    @classmethod
    def from_dict(cls, row: dict) -> Task:
        return cls(row["id"], row["objective"], tuple(row.get("constraints", ())), row.get("evaluation_spec", ""))

    def semantic_text(self) -> str:
        return "\n".join((self.objective, *self.constraints, self.evaluation_spec))


@dataclass(frozen=True, slots=True)
class Artifact:
    id: str
    version: int
    timestamp: int
    text: str
    source: str

    @property
    def key(self) -> str:
        return json_text((self.id, self.version))


@dataclass(frozen=True, slots=True)
class Observation:
    id: str
    timestamp: int
    text: str


@dataclass(frozen=True, slots=True)
class Action:
    id: str
    timestamp: int
    text: str
    observations: tuple[Observation, ...] = ()
    reads: tuple[Artifact, ...] = ()
    writes: tuple[Artifact, ...] = ()
    applications: tuple[tuple[str, int], ...] = ()
    reasoning: str = ""

    @classmethod
    def from_dict(cls, row: dict) -> Action:
        return cls(
            row["id"], row["timestamp"], row["text"],
            tuple(Observation(**item) for item in row.get("observations", ())),
            tuple(Artifact(**item) for item in row.get("reads", ())),
            tuple(Artifact(**item) for item in row.get("writes", ())),
            tuple((identity, version) for identity, version in row.get("applications", ())),
            row.get("reasoning", ""),
        )


@dataclass(frozen=True, slots=True)
class Retrieval:
    experience_id: str
    version: int
    task: Task
    timestamp: int
    boundary_id: str

    @classmethod
    def from_dict(cls, row: dict) -> Retrieval:
        return cls(row["experience_id"], row["version"], Task.from_dict(row["task"]), row["timestamp"], row["boundary_id"])


@dataclass(frozen=True, slots=True)
class Trajectory:
    id: str
    lineage: str
    task: Task
    actions: tuple[Action, ...]
    completed_at: int
    success: bool | None = None
    evaluator_evidence: str = ""
    checkpoint_id: str = ""

    @classmethod
    def from_dict(cls, row: dict) -> Trajectory:
        return cls(
            row["id"], row.get("lineage", row["id"]), Task.from_dict(row["task"]),
            tuple(Action.from_dict(item) for item in row["actions"]), row["completed_at"],
            row.get("success"), row.get("evaluator_evidence", ""), row.get("checkpoint_id", ""),
        )


@dataclass(frozen=True, slots=True)
class Boundary:
    id: str
    timestamp: int
    trajectory_id: str
    task: Task
    state_text: str
    prefix: tuple[Action, ...] = ()
    artifacts: tuple[Artifact, ...] = ()

    @classmethod
    def from_dict(cls, row: dict) -> Boundary:
        return cls(
            row["id"], row["timestamp"], row["trajectory_id"], Task.from_dict(row["task"]),
            row["state_text"], tuple(Action.from_dict(item) for item in row.get("prefix", ())),
            tuple(Artifact(**item) for item in row.get("artifacts", ())),
        )

    def query(self) -> str:
        return "\n".join((self.task.semantic_text(), self.state_text))
