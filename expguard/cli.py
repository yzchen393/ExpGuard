from __future__ import annotations

import argparse
import os
from contextlib import ExitStack
from pathlib import Path

from .benchmarks import BENCHMARKS, ONLINE, build_replay_records, evaluate_online, summarize_online
from .experience import ExperienceLibrary, ReasoningBankExtractor
from .experiments import evaluate_checkpoint, fit, run_suite, threshold_grid
from .llm import LanguageModel
from .model import BGEM3Encoder
from .operators import MinimalNeutralizer
from .records import Trajectory, json_text, read_jsonl, write_jsonl
from .runtime import ContinuationRunner
from .supervision import MatchedReplay, lineage_split
from .training import ExperimentSetting, load_predictor

SUITES = ("ablations", "encoders", "replay-count", "loss-weight")


def model_options(parser, *, agent: bool = True):
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--operator-model", required=True)
    if agent:
        parser.add_argument("--agent-model")
    parser.add_argument("--api-key-env", default="EXPGUARD_API_KEY")


def device_options(parser, *, predictor: bool = True):
    if predictor:
        parser.add_argument("--device", default="cpu")
    parser.add_argument("--encoder-device", default="cpu")
    parser.add_argument("--bge-m3", default="BAAI/bge-m3")


def benchmark_options(parser, names=tuple(BENCHMARKS)):
    parser.add_argument("--benchmark", choices=names, required=True)
    parser.add_argument("--backbone")
    parser.add_argument("--input", type=Path, required=True)


def seed_option(parser):
    parser.add_argument("--seed", type=int, default=0)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="expguard")
    commands = parser.add_subparsers(dest="command", required=True)

    extract = commands.add_parser("extract")
    extract.add_argument("--input", type=Path, required=True)
    extract.add_argument("--library", type=Path, required=True)
    extract.add_argument("--reasoningbank", type=Path, required=True)
    model_options(extract, agent=False)

    recorded = commands.add_parser("import-replays")
    replay = commands.add_parser("replay")
    for command in (recorded, replay):
        benchmark_options(command)
        seed_option(command)
        command.add_argument("--library", type=Path)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--repetitions", type=int, choices=range(1, 6), default=3)
    model_options(replay)
    device_options(replay, predictor=False)

    train = commands.add_parser("train")
    detect = commands.add_parser("evaluate-detection")
    suite = commands.add_parser("experiments")
    for command in (train, detect, suite):
        benchmark_options(command)
        device_options(command)
    for command in (train, suite):
        seed_option(command)
        command.add_argument("--resume", action="store_true")
    for command in (train, detect):
        command.add_argument("--checkpoint", type=Path, required=True)
    suite.add_argument("--suite", choices=SUITES, required=True)
    suite.add_argument("--checkpoints", type=Path, required=True)

    online = commands.add_parser("evaluate-online")
    thresholds = commands.add_parser("sweep-threshold")
    for command in (online, thresholds):
        benchmark_options(command, ONLINE)
        seed_option(command)
        device_options(command)
        model_options(command)
        command.add_argument("--library", type=Path)
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    online.add_argument("--gamma", type=float)
    online.add_argument("--include-baseline", action="store_true")

    score = commands.add_parser("score-online-records")
    benchmark_options(score, ONLINE)
    return parser.parse_args(argv)


def language_models(args, stack):
    key = os.environ.get(args.api_key_env)
    operators = stack.enter_context(LanguageModel(args.endpoint, args.operator_model, key))
    agent_model = getattr(args, "agent_model", None)
    agent = (
        stack.enter_context(LanguageModel(args.endpoint, agent_model, key)) if agent_model else operators
    )
    return operators, agent


def make_encoder(args) -> BGEM3Encoder:
    return BGEM3Encoder(args.bge_m3, args.encoder_device)


def extract_experiences(args) -> dict:
    library = ExperienceLibrary(args.library)
    total = 0
    with ExitStack() as stack:
        llm, _ = language_models(args, stack)
        extractor = ReasoningBankExtractor(args.reasoningbank, llm)
        for row in read_jsonl(args.input):
            for item in extractor.extract(Trajectory.from_dict(row)):
                library.add(item)
                total += 1
    return {"extracted": total, "library": str(args.library)}


def replay_data(args, dataset) -> dict:
    shared = ExperienceLibrary(args.library).versions()
    cases = dataset.cases(args.input, args.backbone)
    records = 0

    def count_rows(rows):
        nonlocal records
        for row in rows:
            records += 1
            yield row

    with ExitStack() as stack:
        encoder = llm = replay = None
        if args.command == "replay":
            llm, agent = language_models(args, stack)
            if any(case.target is None for case in cases):
                encoder = make_encoder(args)
            replay = MatchedReplay(ContinuationRunner(agent), MinimalNeutralizer(llm))
        write_jsonl(args.output, count_rows(build_replay_records(
            dataset, cases, shared, repetitions=args.repetitions, seed=args.seed,
            encoder=encoder, llm=llm, replay=replay,
        )))
    return {"benchmark": dataset.name, "records": records, "output": str(args.output)}


def run_detection(args, dataset) -> dict:
    records = dataset.replay_records(args.input, args.backbone)
    encoder = make_encoder(args)
    if args.command == "train":
        partitions = lineage_split(records, args.seed)
        results = fit(
            partitions, encoder, ExperimentSetting(), args.checkpoint,
            device=args.device, seed=args.seed, resume=args.resume,
        )
    elif args.command == "evaluate-detection":
        results = evaluate_checkpoint(records, encoder, args.checkpoint, args.device)
    else:
        if args.suite == "replay-count" and dataset.name != "CIK-Bench":
            raise ValueError("The replay-count experiment uses CIK-Bench")
        results = run_suite(
            records, encoder, args.suite, args.checkpoints,
            device=args.device, seed=args.seed, resume=args.resume,
        )
    return {"benchmark": dataset.name, "results": results}


def run_online(args, dataset) -> dict:
    predictor = load_predictor(args.checkpoint, args.device)
    cases = dataset.cases(args.input, args.backbone)
    encoder = make_encoder(args)
    shared = ExperienceLibrary(args.library).versions()
    metrics = {}
    if args.command == "sweep-threshold":
        thresholds = threshold_grid(dataset.threshold)
    else:
        thresholds = (dataset.threshold if args.gamma is None else args.gamma,)
    with ExitStack() as stack:
        llm, agent = language_models(args, stack)

        def executions():
            for gamma in thresholds:
                summary, rows = evaluate_online(
                    dataset, cases, shared, encoder=encoder, llm=llm, agent=agent, predictor=predictor,
                    gamma=gamma, seed=args.seed, device=args.device,
                    include_baseline=getattr(args, "include_baseline", False),
                )
                metrics[str(gamma)] = summary
                yield from rows

        write_jsonl(args.output, executions())
    return {"benchmark": dataset.name, "thresholds": metrics}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output = getattr(args, "output", None)
    library = getattr(args, "library", None)
    checkpoint = getattr(args, "checkpoint", None)
    if output is not None and output.resolve() in {
        path.resolve() for path in (args.input, library, checkpoint) if path is not None
    }:
        raise ValueError("Input and output paths must differ")
    if args.command == "train" and checkpoint.resolve() == args.input.resolve():
        raise ValueError("Replay input and training checkpoint paths must differ")
    if args.command == "extract":
        if args.input.resolve() == args.library.resolve():
            raise ValueError("Trajectory input and experience library paths must differ")
        report = extract_experiences(args)
    else:
        dataset = BENCHMARKS[args.benchmark]
        if args.command in {"import-replays", "replay"}:
            report = replay_data(args, dataset)
        elif args.command in {"train", "evaluate-detection", "experiments"}:
            report = run_detection(args, dataset)
        elif args.command == "score-online-records":
            report = {"benchmark": dataset.name, **summarize_online(list(dataset.rows(args.input, args.backbone)))}
        else:
            report = run_online(args, dataset)
    print(json_text(report))


if __name__ == "__main__":
    main()
