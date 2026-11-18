from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass
from typing import Iterable

from .experience import ExperienceLibrary
from .graph import GraphBuilder, GraphSnapshot
from .operators import MinimalNeutralizer
from .records import Boundary, Experience, json_text
from .runtime import Checkpoint, ContinuationRunner, Rollout

UNKNOWN = "UNKNOWN"
Label = int | str


def observed(value: Label | None) -> bool:
    return type(value) is int and value in (0, 1)


@dataclass(frozen=True, slots=True)
class RiskLabels:
    attack: Label
    task: Label

    @property
    def mask(self) -> tuple[bool, bool]:
        return observed(self.attack), observed(self.task)

    @property
    def any_risk(self) -> Label:
        if 1 in (self.attack, self.task):
            return 1
        if self.attack == self.task == 0:
            return 0
        return UNKNOWN

    @classmethod
    def paired(cls, original: Outcomes, neutralized: Outcomes) -> RiskLabels:
        return cls(
            int(original.attack > neutralized.attack)
            if observed(original.attack) and observed(neutralized.attack) else UNKNOWN,
            int(neutralized.task > original.task)
            if observed(original.task) and observed(neutralized.task) else UNKNOWN,
        )


@dataclass(frozen=True, slots=True)
class Outcomes:
    attack: int | None
    task: int | None

    def __post_init__(self) -> None:
        if any(value is not None and not observed(value) for value in (self.attack, self.task)):
            raise ValueError("Outcomes must be 0, 1 or unknown")

    @classmethod
    def from_dict(cls, row: dict) -> Outcomes:
        def decode(value):
            if value == UNKNOWN:
                return None
            return int(value) if type(value) is bool else value

        return cls(decode(row.get("attack")), decode(row.get("task")))


@dataclass(frozen=True, slots=True)
class Pair:
    seed: int
    original: Outcomes
    neutralized: Outcomes

    @classmethod
    def from_dict(cls, row: dict) -> Pair:
        return cls(row["seed"], Outcomes.from_dict(row["original"]), Outcomes.from_dict(row["neutralized"]))

    def labels(self) -> RiskLabels:
        return RiskLabels.paired(self.original, self.neutralized)


def majority(values: Iterable[Label]) -> Label:
    votes = [value for value in values if observed(value)]
    zeros, ones = votes.count(0), votes.count(1)
    return int(ones > zeros) if ones != zeros else UNKNOWN


@dataclass(frozen=True, slots=True)
class ReplayRecord:
    original: Experience
    graph: GraphSnapshot
    graph_one_hop: GraphSnapshot
    pairs: tuple[Pair, ...]
    neutralized: Experience | None = None
    rollouts: tuple[tuple[Rollout, Rollout], ...] = ()
    checkpoint: Checkpoint | None = None

    def __post_init__(self) -> None:
        if not self.pairs or len({pair.seed for pair in self.pairs}) != len(self.pairs):
            raise ValueError("Matched pairs require distinct repetition seeds")
        if (
            self.graph.target_id != self.original.key or self.graph_one_hop.target_id != self.original.key
            or self.graph.boundary_id != self.graph_one_hop.boundary_id
            or self.graph.timestamp != self.graph_one_hop.timestamp
            or self.graph.task_id != self.graph_one_hop.task_id
        ):
            raise ValueError("Replay graph does not match the original experience and checkpoint")
        if self.neutralized is not None and (
            self.neutralized.id != self.original.id or self.neutralized.provenance != self.original.provenance
            or self.neutralized.version_metadata.parent != self.original.version
            or self.neutralized.version_metadata.operator != "neutralization"
        ):
            raise ValueError("Neutralized experience does not match the original intervention lineage")
        if self.checkpoint is not None and self.checkpoint.boundary.id != self.graph.boundary_id:
            raise ValueError("Replay checkpoint does not match the snapshot")
        if self.rollouts:
            recorded = tuple(
                Pair(left.seed, Outcomes(left.attack, left.task), Outcomes(right.attack, right.task))
                for left, right in self.rollouts if left.seed == right.seed
            )
            if len(self.rollouts) != len(self.pairs) or recorded != self.pairs:
                raise ValueError("Replay continuations do not match their pair records")

    @classmethod
    def from_dict(cls, row: dict, *, include_rollouts: bool = True) -> ReplayRecord:
        rollouts = row.get("rollouts", ()) if include_rollouts else ()
        checkpoint = row.get("checkpoint") if include_rollouts else None
        return cls(
            Experience.from_dict(row["original"]),
            GraphSnapshot.from_dict(row["graph"]), GraphSnapshot.from_dict(row["graph_one_hop"]),
            tuple(Pair.from_dict(item) for item in row["pairs"]),
            Experience.from_dict(row["neutralized"]) if row.get("neutralized") is not None else None,
            tuple((Rollout.from_dict(left), Rollout.from_dict(right)) for left, right in rollouts),
            Checkpoint.from_dict(checkpoint) if checkpoint is not None else None,
        )

    def labels(self, count: int = 3) -> RiskLabels:
        if count < 1 or len(self.pairs) < count:
            raise ValueError(f"Need {count} matched pairs, found {len(self.pairs)}")
        labels = [pair.labels() for pair in self.pairs[:count]]
        return RiskLabels(majority(label.attack for label in labels), majority(label.task for label in labels))

    def to_dict(self, count: int = 3) -> dict:
        aggregate = self.labels(count)
        return {
            **asdict(self),
            "pair_labels": [{"labels": asdict(pair.labels()), "mask": pair.labels().mask} for pair in self.pairs],
            "labels": asdict(aggregate),
            "mask": aggregate.mask,
            "repetitions": count,
        }


def replay_graphs(original: Experience, boundary: Boundary, builder: GraphBuilder, library: ExperienceLibrary):
    versions = library.versions()
    return builder.snapshot(original, boundary, versions), builder.snapshot(original, boundary, versions, hops=1)


class MatchedReplay:
    def __init__(self, runner: ContinuationRunner, neutralizer: MinimalNeutralizer):
        self.runner = runner
        self.neutralizer = neutralizer

    def run(
        self, checkpoint: Checkpoint, original: Experience, builder: GraphBuilder,
        library: ExperienceLibrary, *, repetitions: int = 3, seed: int = 0,
    ) -> ReplayRecord:
        if not 1 <= repetitions <= 5:
            raise ValueError("Replay repetitions must be in the range 1 to 5")
        boundary = checkpoint.boundary
        graph, graph_one_hop = replay_graphs(original, boundary, builder, library)
        neutralized = self.neutralizer.neutralize(original, boundary)
        identities = [item.id for item in checkpoint.active]
        if len(identities) != len(set(identities)):
            raise ValueError("Checkpoint contains duplicate experience identities")
        active = checkpoint.active
        if original.id not in identities:
            active = (*active, original)
        branch_original = tuple(original if item.id == original.id else item for item in active)
        branch_neutralized = tuple(neutralized if item.id == original.id else item for item in active)
        key = hashlib.sha256(json_text((seed, boundary.id, original.key)).encode("utf-8")).digest()
        rng = random.Random(int.from_bytes(key))
        pairs, rollouts = [], []
        for pair_seed in rng.sample(range(2**31), repetitions):
            left = self.runner.run(checkpoint, pair_seed, active=branch_original)
            right = self.runner.run(checkpoint, pair_seed, active=branch_neutralized)
            rollouts.append((left, right))
            pairs.append(Pair(pair_seed, Outcomes(left.attack, left.task), Outcomes(right.attack, right.task)))
        return ReplayRecord(original, graph, graph_one_hop, tuple(pairs), neutralized, tuple(rollouts), checkpoint)


def lineage_split(records: list[ReplayRecord], seed: int = 0) -> dict[str, list[ReplayRecord]]:
    links = {}

    def root(key):
        links.setdefault(key, key)
        while key != links[key]:
            links[key] = links[links[key]]
            key = links[key]
        return key

    for record in records:
        original = record.original
        identity = root(("experience", original.id))
        for namespace, value in (("lineage", original.provenance.lineage),
                                 ("source", original.provenance.source_trajectory)):
            links[root((namespace, value))] = identity
    components = {}
    for record in records:
        key = root(("experience", record.original.id))
        components.setdefault(key, []).append(record)

    groups = [components[key] for key in sorted(components)]
    if len(groups) < 3:
        raise ValueError("Train/validation/test splitting requires at least three independent lineage groups")
    random.Random(seed).shuffle(groups)
    groups.sort(key=len, reverse=True)
    partitions = {"train": [], "validation": [], "test": []}
    targets = dict(zip(partitions, (0.8, 0.1, 0.1), strict=True))
    for index, group in enumerate(groups):
        empty = [name for name, split in partitions.items() if not split]
        available = empty if len(groups) - index == len(empty) else partitions
        name = max(available, key=lambda part: len(records) * targets[part] - len(partitions[part]))
        partitions[name].extend(group)
    return partitions
