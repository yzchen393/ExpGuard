from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from .model import ENCODERS, HEADS, BGEM3Encoder
from .supervision import ReplayRecord, lineage_split
from .training import ExperimentSetting, evaluate, load_predictor, prepare, train


def settings(suite: str) -> tuple[ExperimentSetting, ...]:
    full = ExperimentSetting()
    if suite == "ablations":
        return (
            full,
            replace(full, name="shuffled-edges", shuffle_edges=True),
            replace(full, name="one-hop", hops=1),
            replace(full, name="without-attack-risk", heads=("task",)),
            replace(full, name="without-task-failure-risk", heads=("attack",)),
        )
    if suite == "encoders":
        return tuple(replace(full, name=architecture, architecture=architecture) for architecture in ENCODERS)
    if suite == "replay-count":
        return tuple(replace(full, name=f"R-{count}", repetitions=count) for count in range(1, 6))
    if suite == "loss-weight":
        return tuple(
            replace(full, name=f"{head}-{value:g}", **{head + "_weight": value})
            for head in HEADS for value in (0.25, 0.5, 1.0, 2.0, 4.0)
        )
    raise ValueError(f"Unknown experiment suite: {suite}")


def fit(
    partitions: dict[str, list[ReplayRecord]], encoder: BGEM3Encoder, setting: ExperimentSetting,
    checkpoint: Path, *, device: str, seed: int, resume: bool,
) -> dict:
    training = prepare(partitions["train"], encoder, setting, seed)
    validation = prepare(partitions["validation"], encoder, setting, seed)
    model = train(training, validation, setting, checkpoint, device=device, seed=seed, resume=resume)
    validation_report = evaluate(model, validation, device)
    del training, validation
    test = prepare(partitions["test"], encoder, setting, seed)
    return {
        "validation": validation_report, "test": evaluate(model, test, device),
        "split_sizes": {name: len(partition) for name, partition in partitions.items()},
    }


def run_suite(
    records: list[ReplayRecord], encoder: BGEM3Encoder, suite: str, directory: Path, *,
    device: str, seed: int, resume: bool,
) -> dict:
    variants = settings(suite)
    required = max(setting.repetitions for setting in variants)
    if any(len(record.pairs) < required for record in records):
        raise ValueError(f"Every record in this suite must have at least {required} matched pairs")
    partitions = lineage_split(records, seed)
    results = {}
    for setting in variants:
        path = directory / (setting.name + ".pt")
        results[setting.name] = fit(
            partitions, encoder, setting, path, device=device, seed=seed, resume=resume and path.exists(),
        )
    return results


def evaluate_checkpoint(records: list[ReplayRecord], encoder: BGEM3Encoder, path: Path, device: str) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=True)
    setting = ExperimentSetting(**state["setting"])
    model = load_predictor(path, device)
    test = lineage_split(records, state["seed"])["test"]
    return evaluate(model, prepare(test, encoder, setting, state["seed"]), device)


def threshold_grid(gamma: float) -> tuple[float, ...]:
    return tuple(sorted({0.0, 0.01, gamma, 0.1, 0.5, 0.9, 0.99}))
