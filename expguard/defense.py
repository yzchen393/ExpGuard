from __future__ import annotations

from dataclasses import dataclass

import torch
from torch_geometric.data import Batch

from .experience import ExperienceLibrary, ExperienceRetrieval
from .graph import GraphBuilder, GraphSnapshot
from .model import HEADS, BGEM3Encoder, RiskPredictor
from .operators import RepairOperator
from .records import Boundary, Experience, Retrieval


@dataclass(frozen=True, slots=True)
class Assessment:
    experience: Experience
    graph: GraphSnapshot
    attack_probability: float
    task_probability: float
    risk: float
    needs_repair: bool


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    candidates: tuple[Experience, ...]
    applicable: tuple[Experience, ...]
    admitted: tuple[Experience, ...]
    withheld: tuple[Experience, ...]
    assessments: tuple[Assessment, ...]
    active: tuple[Experience, ...]


def refresh_context(
    active: tuple[Experience, ...], applicable: tuple[Experience, ...], admitted: tuple[Experience, ...],
) -> tuple[Experience, ...]:
    refreshed = {item.id for item in applicable}
    return (*(item for item in active if item.id not in refreshed), *admitted)


class OnlineDefense:
    def __init__(
        self, retrieval: ExperienceRetrieval, graph: GraphBuilder, repair: RepairOperator,
        predictor: RiskPredictor, encoder: BGEM3Encoder, gamma: float, device: str,
    ):
        if not 0 <= gamma <= 1 or set(predictor.head_names) != set(HEADS):
            raise ValueError("The online gate requires both heads and a threshold in [0, 1]")
        self.retrieval = retrieval
        self.library: ExperienceLibrary = retrieval.library
        self.graph = graph
        self.repair = repair
        self.predictor = predictor.eval()
        self.encoder = encoder
        self.gamma = gamma
        self.device = device
        self.results: dict[str, RetrievalResult] = {}

    @torch.no_grad()
    def assess(self, item: Experience, boundary: Boundary) -> Assessment:
        snapshot = self.graph.snapshot(item, boundary, self.library.versions())
        graph = Batch.from_data_list([snapshot.to_heterodata(self.encoder)]).to(self.device)
        values = self.predictor.probabilities(graph)
        attack = float(values["attack"].item())
        task = float(values["task"].item())
        if not (0 <= attack <= 1 and 0 <= task <= 1):
            raise ValueError("The predictor returned invalid risk probabilities")
        risk = max(attack, task)
        return Assessment(item, snapshot, attack, task, risk, risk >= self.gamma)

    def retrieve(self, boundary: Boundary, active: tuple[Experience, ...]) -> RetrievalResult:
        if boundary.id in self.results:
            raise ValueError(f"Duplicate retrieval boundary {boundary.id}")
        candidates = self.retrieval.top_k(boundary)
        applicable = self.retrieval.applicable(candidates, boundary)
        admitted, withheld, assessments = [], [], []
        retrievals = [
            Retrieval(item.id, item.version, boundary.task, boundary.timestamp, boundary.id)
            for item in candidates
        ]
        for original in applicable:
            item = original
            assessment = self.assess(item, boundary)
            assessments.append(assessment)
            for _ in range(3):
                if not assessment.needs_repair:
                    break
                item = self.library.add(self.repair.repair(item, boundary, assessment.graph))
                retrievals.append(Retrieval(item.id, item.version, boundary.task, boundary.timestamp, boundary.id))
                assessment = self.assess(item, boundary)
                assessments.append(assessment)
            if assessment.risk < self.gamma:
                admitted.append(item)
            else:
                withheld.append(item)
        self.graph.retrievals = (*self.graph.retrievals, *retrievals)
        admitted = tuple(admitted)
        result = RetrievalResult(
            candidates, applicable, admitted, tuple(withheld), tuple(assessments),
            refresh_context(active, applicable, admitted),
        )
        self.results[boundary.id] = result
        return result

    def __call__(self, boundary: Boundary, active: tuple[Experience, ...]) -> tuple[Experience, ...]:
        return self.retrieve(boundary, active).active
