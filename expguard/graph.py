from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, replace

import torch
from torch_geometric.data import HeteroData

from .records import Action, Artifact, Boundary, Experience, Retrieval, Task, Trajectory, json_text

NODE_TYPES = ("experience", "task", "action", "observation", "artifact")
CORE_RELATIONS = (
    ("experience", "DERIVED_FROM", "action"),
    ("action", "PRODUCES", "observation"),
    ("action", "NEXT", "action"),
    ("action", "READS", "artifact"),
    ("action", "WRITES", "artifact"),
    ("experience", "RETRIEVED_FOR", "task"),
    ("experience", "APPLIES_TO", "action"),
)
EDGE_TYPES = CORE_RELATIONS + tuple((end, "REV_" + name, start) for start, name, end in CORE_RELATIONS)
DEPENDENCIES = {"PRODUCES", "NEXT", "READS", "WRITES"}
ATTRIBUTE_DIM = len(NODE_TYPES) + 4


@dataclass(frozen=True, slots=True)
class Node:
    kind: str
    id: str
    text: str
    timestamp: int
    source: str
    checkpoint: str = ""
    version: int = 0
    provenance: str = ""

    @property
    def key(self) -> tuple[str, str]:
        return self.kind, self.id

    def features(self, timestamp: int, trajectory: str) -> list[float]:
        return [
            *(float(self.kind == name) for name in NODE_TYPES),
            math.log1p(max(0, self.timestamp)),
            math.log1p(timestamp - self.timestamp),
            math.log1p(self.version),
            float(self.source == trajectory),
        ]


@dataclass(frozen=True, slots=True)
class Edge:
    relation: str
    left: tuple[str, str]
    right: tuple[str, str]
    timestamp: int
    source: str
    checkpoint: str = ""

    @property
    def kind(self) -> tuple[str, str, str]:
        return self.left[0], self.relation, self.right[0]

    def reverse(self) -> Edge:
        return replace(self, relation="REV_" + self.relation, left=self.right, right=self.left)


@dataclass(frozen=True, slots=True)
class GraphSnapshot:
    boundary_id: str
    timestamp: int
    trajectory_id: str
    task_id: str
    target_id: str
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]

    def __post_init__(self) -> None:
        by_key = {node.key: node for node in self.nodes}
        anchors = {("experience", self.target_id), ("task", self.task_id)}
        if len(by_key) != len(self.nodes) or not anchors <= by_key.keys():
            raise ValueError("A snapshot requires unique nodes and both readout anchors")
        if any(
            node.timestamp > self.timestamp
            or (node.timestamp == self.timestamp and node.key not in anchors)
            or node.kind not in NODE_TYPES
            or node.version < 0 for node in self.nodes
        ):
            raise ValueError("Snapshot node is outside the available evidence")
        for edge in self.edges:
            if (
                edge.kind not in CORE_RELATIONS or edge.left not in by_key or edge.right not in by_key
                or edge.timestamp > self.timestamp
                or (edge.timestamp == self.timestamp and edge.relation not in {"RETRIEVED_FOR", "DERIVED_FROM"})
            ):
                raise ValueError("Snapshot edge is outside the available evidence")
        if not any(
            edge.relation == "RETRIEVED_FOR" and edge.left == ("experience", self.target_id)
            and edge.right == ("task", self.task_id) and edge.timestamp == self.timestamp
            for edge in self.edges
        ):
            raise ValueError("Snapshot requires the current retrieval edge")

    @classmethod
    def from_dict(cls, row: dict) -> GraphSnapshot:
        return cls(
            row["boundary_id"], row["timestamp"], row["trajectory_id"], row["task_id"], row["target_id"],
            tuple(Node(**item) for item in row["nodes"]),
            tuple(Edge(**{**item, "left": tuple(item["left"]), "right": tuple(item["right"])})
                  for item in row["edges"]),
        )

    def shuffled(self, seed: int) -> GraphSnapshot:
        relations = defaultdict(list)
        for index, edge in enumerate(self.edges):
            if edge.source != self.trajectory_id and edge.relation in DEPENDENCIES:
                relations[edge.kind].append(index)
        rng = random.Random(seed)
        edges = list(self.edges)
        for indices in relations.values():
            destinations = [edges[index].right for index in indices]
            shuffled = destinations.copy()
            rng.shuffle(shuffled)
            if shuffled == destinations and len(set(destinations)) > 1:
                shift = next(index for index, value in enumerate(shuffled) if value != shuffled[0])
                shuffled = shuffled[shift:] + shuffled[:shift]
            for index, destination in zip(indices, shuffled, strict=True):
                edges[index] = replace(edges[index], right=destination)
        return replace(self, edges=tuple(edges))

    def to_heterodata(self, encoder) -> HeteroData:
        graph = HeteroData()
        embedding = encoder.encode([node.text for node in self.nodes])
        positions = {}
        by_type = {name: [] for name in NODE_TYPES}
        for index, node in enumerate(self.nodes):
            positions[node.key] = len(by_type[node.kind])
            by_type[node.kind].append((index, node))
        for kind, entries in by_type.items():
            indices, nodes = zip(*entries) if entries else ((), ())
            graph[kind].x = embedding.index_select(0, torch.tensor(indices, dtype=torch.long))
            graph[kind].attributes = torch.tensor(
                [node.features(self.timestamp, self.trajectory_id) for node in nodes],
                dtype=torch.float32,
            ).reshape(-1, ATTRIBUTE_DIM)
            graph[kind].is_anchor = torch.tensor(
                [(kind == "experience" and node.id == self.target_id)
                 or (kind == "task" and node.id == self.task_id) for node in nodes],
                dtype=torch.bool,
            )
        links = {kind: [] for kind in EDGE_TYPES}
        for edge in self.message_edges():
            links[edge.kind].append((positions[edge.left], positions[edge.right]))
        for kind, endpoints in links.items():
            graph[kind].edge_index = torch.tensor(endpoints, dtype=torch.long).reshape(-1, 2).T.contiguous()
        return graph

    def message_edges(self) -> tuple[Edge, ...]:
        return (*self.edges, *(edge.reverse() for edge in self.edges))


class GraphBuilder:
    def __init__(
        self, trajectories: tuple[Trajectory, ...] = (),
        retrievals: tuple[Retrieval, ...] = (),
        *, max_nodes: int = 64, max_edges: int = 256,
    ):
        if max_nodes < 2 or max_edges < 2:
            raise ValueError("Graph budget must contain both anchors and their retrieval link")
        self.trajectories = trajectories
        self.retrievals = retrievals
        self.max_nodes = max_nodes
        self.max_edges = max_edges

    def snapshot(
        self, target: Experience, boundary: Boundary, versions: tuple[Experience, ...],
        *, hops: int = 2,
    ) -> GraphSnapshot:
        if hops not in (1, 2):
            raise ValueError("Neighborhood depth must be one or two")
        timestamp = boundary.timestamp
        metadata = target.version_metadata
        if (
            metadata.operator not in {"extraction", "repair"}
            or metadata.timestamp > timestamp
            or (metadata.timestamp == timestamp and metadata.boundary != boundary.id)
        ):
            raise ValueError("The target experience is unavailable at the boundary")

        nodes: dict[tuple[str, str], Node] = {}
        edges: dict[tuple, Edge] = {}
        actions_by_trajectory = defaultdict(dict)
        observations_by_action = defaultdict(set)

        def add_node(node: Node) -> tuple[str, str]:
            existing = nodes.get(node.key)
            if existing is None or node.timestamp < existing.timestamp:
                nodes[node.key] = node
            return node.key

        def add_edge(relation: str, left: tuple[str, str], right: tuple[str, str],
                     time: int, source: str, checkpoint: str = "") -> None:
            edge = Edge(relation, left, right, time, source, checkpoint)
            edges[(relation, left, right, source, checkpoint, time)] = edge

        def add_artifact(artifact: Artifact, checkpoint: str) -> tuple[str, str]:
            key = ("artifact", artifact.key)
            existing = nodes.get(key)
            if existing is not None and existing.text != artifact.text:
                raise ValueError(f"Conflicting artifact version {artifact.key}")
            return add_node(Node(
                "artifact", artifact.key, artifact.text, artifact.timestamp, artifact.source,
                checkpoint, artifact.version,
            ))

        def add_actions(steps: tuple[Action, ...], source: str, checkpoint: str) -> list[tuple[str, str]]:
            action_keys = []
            previous = None
            seen = set()
            for step in sorted(steps, key=lambda action: action.timestamp):
                if step.timestamp >= timestamp:
                    continue
                if step.id in seen:
                    raise ValueError(f"Duplicate action {step.id} in {source}")
                seen.add(step.id)
                key = add_node(Node("action", event_key(source, step.id), step.text, step.timestamp, source, checkpoint))
                actions_by_trajectory[source][step.id] = key
                if previous is not None:
                    add_edge("NEXT", previous, key, step.timestamp, source, checkpoint)
                previous = key
                action_keys.append(key)
                for obs in step.observations:
                    if step.timestamp <= obs.timestamp < timestamp:
                        obs_key = add_node(Node(
                            "observation", event_key(source, obs.id), obs.text, obs.timestamp, source, checkpoint,
                        ))
                        add_edge("PRODUCES", key, obs_key, obs.timestamp, source, checkpoint)
                        observations_by_action[key].add(obs.id)
                for relation, artifacts in (("READS", step.reads), ("WRITES", step.writes)):
                    for artifact in artifacts:
                        if artifact.timestamp <= step.timestamp:
                            add_edge(relation, key, add_artifact(artifact, checkpoint), step.timestamp, source, checkpoint)
                for identity, version in step.applications:
                    add_edge("APPLIES_TO", ("experience", event_key(identity, version)),
                             key, step.timestamp, source, checkpoint)
            return action_keys

        def add_task(task: Task, time: int, source: str, checkpoint: str) -> tuple[str, str]:
            return add_node(Node("task", task.id, task.semantic_text(), time, source, checkpoint))

        for trajectory in self.trajectories:
            if trajectory.completed_at < timestamp and trajectory.id != boundary.trajectory_id:
                add_actions(trajectory.actions, trajectory.id, trajectory.checkpoint_id)
        prefix = add_actions(boundary.prefix, boundary.trajectory_id, boundary.id)
        current_artifacts = [
            add_artifact(artifact, boundary.id) for artifact in boundary.artifacts
            if artifact.timestamp < timestamp
        ]
        visible_versions = {
            item.key: item for item in versions
            if item.version_metadata.timestamp < timestamp
            and item.version_metadata.operator in {"extraction", "repair"}
        }
        visible_versions[target.key] = target
        for item in visible_versions.values():
            provenance = item.provenance
            key = add_node(Node(
                "experience", item.key, item.semantic_text(), item.version_metadata.timestamp,
                provenance.source_trajectory, item.version_metadata.boundary or "", item.version,
                provenance.extraction_record,
            ))
            supporting_observations = set(provenance.supporting_observations)
            supporting_actions = set(provenance.supporting_actions)
            for action_id, action_key in actions_by_trajectory[provenance.source_trajectory].items():
                if action_id in supporting_actions or supporting_observations & observations_by_action[action_key]:
                    add_edge("DERIVED_FROM", key, action_key, item.version_metadata.timestamp,
                             provenance.source_trajectory, item.version_metadata.boundary or "")

        for retrieval in self.retrievals:
            key = ("experience", event_key(retrieval.experience_id, retrieval.version))
            if retrieval.timestamp < timestamp and key in nodes and nodes[key].timestamp <= retrieval.timestamp:
                task = add_task(retrieval.task, retrieval.timestamp, retrieval.task.id, retrieval.boundary_id)
                add_edge("RETRIEVED_FOR", key, task, retrieval.timestamp, retrieval.task.id, retrieval.boundary_id)

        target_key = ("experience", target.key)
        task_key = add_task(boundary.task, timestamp, boundary.trajectory_id, boundary.id)
        nodes[task_key] = Node(
            "task", boundary.task.id, boundary.task.semantic_text(), timestamp, boundary.trajectory_id, boundary.id,
        )
        add_edge("RETRIEVED_FOR", target_key, task_key, timestamp, boundary.trajectory_id, boundary.id)
        observable = tuple(edge for edge in edges.values() if (
            edge.left in nodes and edge.right in nodes and nodes[edge.left].timestamp <= edge.timestamp
            and (nodes[edge.right].timestamp <= edge.timestamp
                 or (edge.relation == "RETRIEVED_FOR" and edge.right == task_key))
        ))
        selected, selected_edges = self._neighborhood(
            nodes, observable, target_key, task_key, prefix, current_artifacts, hops, boundary.trajectory_id,
        )
        return GraphSnapshot(
            boundary.id, timestamp, boundary.trajectory_id, boundary.task.id, target.key,
            tuple(nodes[key] for key in selected), selected_edges,
        )

    def _neighborhood(self, nodes, edges, target, task, prefix, artifacts, hops, current_source):
        adjacency = defaultdict(set)
        for edge in edges:
            adjacency[edge.left].add(edge.right)
            adjacency[edge.right].add(edge.left)

        def order(key):
            return -nodes[key].timestamp, key

        auxiliary_budget = max(0, (self.max_nodes - 2) // 4)
        prefix_anchors = sorted(set(prefix + artifacts), key=order)[:auxiliary_budget]
        seeds = tuple(dict.fromkeys((target, task, *prefix_anchors)))
        distances = dict.fromkeys(seeds, 0)
        frontier = set(seeds)
        for distance in range(1, hops + 1):
            frontier = {other for key in frontier for other in adjacency[key]} - distances.keys()
            distances.update(dict.fromkeys(frontier, distance))
        selected = tuple(sorted(distances, key=lambda key: (
            distances[key], 0 if key in (target, task) else 1, order(key),
        ))[:self.max_nodes])
        chosen = set(selected)

        def edge_priority(edge: Edge):
            return (
                0 if edge.relation == "RETRIEVED_FOR" and edge.left == target
                and edge.right == task and edge.source == current_source else 1,
                max(distances[edge.left], distances[edge.right]),
                min(distances[edge.left], distances[edge.right]),
                -edge.timestamp, edge.kind, edge.left, edge.right,
            )

        available = (edge for edge in edges if edge.left in chosen and edge.right in chosen)
        return selected, tuple(sorted(available, key=edge_priority)[:self.max_edges // 2])


def event_key(source: str, identity: str | int) -> str:
    return json_text((source, identity))


def repair_evidence(snapshot: GraphSnapshot) -> dict:
    return {
        "nodes": [asdict(node) for node in snapshot.nodes],
        "edges": [asdict(edge) for edge in snapshot.edges],
    }
