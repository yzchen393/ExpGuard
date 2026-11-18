from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter

from .defense import OnlineDefense, refresh_context
from .experience import ExperienceLibrary, ExperienceRetrieval
from .llm import LanguageModel
from .graph import GraphBuilder
from .metrics import online_metrics
from .model import BGEM3Encoder, RiskPredictor
from .operators import RepairOperator
from .records import Boundary, Experience, Retrieval, Trajectory, read_jsonl
from .runtime import Checkpoint, ContinuationRunner
from .supervision import MatchedReplay, Outcomes, Pair, ReplayRecord, replay_graphs


@dataclass(frozen=True, slots=True)
class Benchmark:
    name: str
    threshold: float | None = None

    def rows(self, path: Path, backbone: str | None = None):
        for row in read_jsonl(path):
            if row.get("benchmark", self.name) == self.name and (backbone is None or row.get("backbone") == backbone):
                yield row

    def replay_records(self, path: Path, backbone: str | None) -> list[ReplayRecord]:
        records = [ReplayRecord.from_dict(row, include_rollouts=False) for row in self.rows(path, backbone)]
        if not records:
            raise ValueError(f"No replay records for {self.name}")
        return records

    def cases(self, path: Path, backbone: str | None) -> list[BenchmarkCase]:
        cases = [BenchmarkCase.from_dict(row) for row in self.rows(path, backbone)]
        if not cases or len({case.id for case in cases}) != len(cases):
            raise ValueError(f"{self.name} requires nonempty, uniquely identified cases")
        return cases


BENCHMARKS = {
    benchmark.name: benchmark for benchmark in (
        Benchmark("MemForm"), Benchmark("CIK-Bench"), Benchmark("SMP-Bench"),
        Benchmark("STATE-Bench", 0.085), Benchmark("AppWorld", 0.093), Benchmark("MemoryArena", 0.076),
    )
}
ONLINE = tuple(name for name, dataset in BENCHMARKS.items() if dataset.threshold is not None)


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    id: str
    backbone: str
    condition: str
    trajectories: tuple[Trajectory, ...]
    experiences: tuple[Experience, ...]
    retrievals: tuple[Retrieval, ...]
    checkpoint: Checkpoint | None
    boundary: Boundary
    target: tuple[str, int] | None
    pairs: tuple[Pair, ...]

    @classmethod
    def from_dict(cls, row: dict) -> BenchmarkCase:
        checkpoint = Checkpoint.from_dict(row["checkpoint"]) if "checkpoint" in row else None
        boundary = checkpoint.boundary if checkpoint is not None else Boundary.from_dict(row["boundary"])
        target = row.get("target")
        pairs = tuple(Pair.from_dict(item) for item in row.get("pairs", ()))
        if pairs and target is None:
            raise ValueError("Recorded replay pairs require a target experience version")
        return cls(
            row["case_id"], row.get("backbone", ""), row.get("condition", ""),
            tuple(Trajectory.from_dict(item) for item in row.get("trajectories", ())),
            tuple(Experience.from_dict(item) for item in row.get("experiences", ())),
            tuple(Retrieval.from_dict(item) for item in row.get("retrievals", ())),
            checkpoint, boundary, (target["id"], target["version"]) if target is not None else None, pairs,
        )

    def library(self, shared: tuple[Experience, ...]) -> ExperienceLibrary:
        versions = {}
        for item in (*shared, *self.experiences):
            key = (item.id, item.version)
            if key in versions and versions[key] != item:
                raise ValueError(f"Conflicting experience version: {key}")
            versions[key] = item
        library = ExperienceLibrary()
        for key in sorted(versions):
            library.add(versions[key])
        return library

    def graph_builder(self) -> GraphBuilder:
        return GraphBuilder(self.trajectories, self.retrievals)

    def metadata(self, benchmark: Benchmark) -> dict:
        return {"benchmark": benchmark.name, "case_id": self.id, "backbone": self.backbone, "condition": self.condition}


def build_replay_records(
    dataset: Benchmark, cases: list[BenchmarkCase], shared: tuple[Experience, ...], *,
    repetitions: int, seed: int, encoder: BGEM3Encoder | None, llm: LanguageModel | None,
    replay: MatchedReplay | None,
):
    for case in cases:
        library = case.library(shared)
        builder = case.graph_builder()
        boundary = case.boundary
        if case.target is not None:
            candidates = (library.get(*case.target),)
        elif encoder is not None and llm is not None:
            retrieval = ExperienceRetrieval(library, encoder, llm)
            candidates = retrieval.applicable(retrieval.top_k(boundary), boundary)
        else:
            raise ValueError(f"Missing recorded candidate for {case.id}")
        for target in candidates:
            if case.pairs:
                graph, graph_one_hop = replay_graphs(target, boundary, builder, library)
                record = ReplayRecord(target, graph, graph_one_hop, case.pairs, checkpoint=case.checkpoint)
            elif case.checkpoint is not None and replay is not None:
                checkpoint = case.checkpoint
                if case.target is None:
                    checkpoint = replace(
                        checkpoint,
                        active=refresh_context(checkpoint.active, candidates, candidates),
                    )
                record = replay.run(checkpoint, target, builder, library, repetitions=repetitions, seed=seed)
            else:
                raise ValueError(f"Missing matched pairs or executable state checkpoint for {case.id}")
            yield {**case.metadata(dataset), **record.to_dict(repetitions)}


def summarize_online(rows: list[dict]) -> dict:
    grouped = defaultdict(list)
    measurements = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (row.get("backbone", ""), row.get("condition", ""), row.get("defense", "expguard"), row.get("gamma"))
        grouped[key].append(Outcomes.from_dict(row["outcomes"]))
        for name, value in row.get("overhead", {}).items():
            if value is not None:
                measurements[key][name].append(value)
    keys = sorted(grouped, key=lambda key: (*key[:3], -1.0 if key[3] is None else key[3]))
    summary = []
    for key in keys:
        backbone, condition, defense, gamma = key
        row = {"backbone": backbone, "condition": condition, "defense": defense, **online_metrics(grouped[key])}
        if gamma is not None:
            row["gamma"] = gamma
        summary.append(row)
    baselines = {
        (row["backbone"], row["condition"]): row for row in summary if row["defense"] == "no-defense"
    }
    for row in summary:
        key = (row["backbone"], row["condition"], row["defense"], row.get("gamma"))
        if measurements[key]:
            row["overhead"] = {
                name: {"mean": mean(values), "std": pstdev(values), "observed": len(values)}
                for name, values in measurements[key].items()
            }
        baseline = baselines.get((row["backbone"], row["condition"]))
        if baseline is not None and row["defense"] != "no-defense":
            for metric in ("asr", "tsr"):
                row["delta_" + metric + "_pp"] = (
                    100 * (row[metric] - baseline[metric])
                    if row[metric] is not None and baseline[metric] is not None else None
                )
    return {"groups": summary, "executions": len(rows)}


def evaluate_online(
    dataset: Benchmark, cases: list[BenchmarkCase], shared: tuple[Experience, ...], *,
    encoder: BGEM3Encoder, llm: LanguageModel, agent: LanguageModel, predictor: RiskPredictor,
    gamma: float, seed: int, device: str, include_baseline: bool = False,
) -> tuple[dict, list[dict]]:
    runner = ContinuationRunner(agent)
    repair = RepairOperator(llm)
    rows = []
    for index, case in enumerate(cases):
        if case.checkpoint is None:
            raise ValueError(f"Case {case.id} has no executable state checkpoint")
        library = case.library(shared)
        retrieval = ExperienceRetrieval(library, encoder, llm)
        gate = OnlineDefense(retrieval, case.graph_builder(), repair, predictor, encoder, gamma, device)
        elapsed = 0.0
        token_count = 0
        complete_usage = True

        def guarded_retrieval(boundary, active):
            nonlocal elapsed, token_count, complete_usage
            opened = perf_counter()
            tokens_before = llm.total_tokens
            missing_before = llm.unreported_requests
            refreshed = gate(boundary, active)
            elapsed += perf_counter() - opened
            token_count += llm.total_tokens - tokens_before
            complete_usage &= llm.unreported_requests == missing_before
            return refreshed

        def no_defense(boundary, active):
            candidates = retrieval.applicable(retrieval.top_k(boundary), boundary)
            return refresh_context(active, candidates, candidates)

        if case.checkpoint.active:
            raise ValueError("Online evaluation must start with an empty active experience context")
        if include_baseline:
            baseline = runner.run(case.checkpoint, seed + index, on_retrieval=no_defense)
            rows.append({
                **case.metadata(dataset), "defense": "no-defense",
                "outcomes": asdict(Outcomes(baseline.attack, baseline.task)), "rollout": asdict(baseline),
                "overhead": {"latency_seconds": 0.0, "defense_tokens": 0},
            })
        rollout = runner.run(case.checkpoint, seed + index, on_retrieval=guarded_retrieval)
        rows.append({
            **case.metadata(dataset), "defense": "expguard", "gamma": gamma,
            "outcomes": asdict(Outcomes(rollout.attack, rollout.task)), "rollout": asdict(rollout),
            "overhead": {"latency_seconds": elapsed, "defense_tokens": token_count if complete_usage else None},
            "retrievals": {identity: asdict(result) for identity, result in gate.results.items()},
        })
    return summarize_online(rows), rows
