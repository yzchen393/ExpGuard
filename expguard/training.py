from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, HeteroData

from .metrics import detection_metrics
from .model import HEADS, BGEM3Encoder, RiskPredictor
from .records import json_text
from .supervision import ReplayRecord, RiskLabels, observed

BATCH_SIZE = 32
MAX_EPOCHS = 100
PATIENCE = 10


@dataclass(frozen=True, slots=True)
class ExperimentSetting:
    name: str = "full"
    architecture: str = "hgtnn"
    heads: tuple[str, ...] = HEADS
    hops: int = 2
    shuffle_edges: bool = False
    repetitions: int = 3
    attack_weight: float = 1.0
    task_weight: float = 1.0

    def weights(self) -> dict[str, float]:
        return {"attack": self.attack_weight, "task": self.task_weight}


@dataclass(frozen=True, slots=True)
class Sample:
    graph: HeteroData
    labels: RiskLabels
    identity: str


def prepare(
    records: list[ReplayRecord], encoder: BGEM3Encoder, setting: ExperimentSetting, seed: int,
) -> list[Sample]:
    samples = []
    for record in records:
        graph = record.graph_one_hop if setting.hops == 1 else record.graph
        identity = json_text((graph.boundary_id, graph.target_id))
        if setting.shuffle_edges:
            shuffle_seed = int.from_bytes(hashlib.sha256(json_text((seed, identity)).encode("utf-8")).digest())
            graph = graph.shuffled(shuffle_seed)
        samples.append(Sample(graph.to_heterodata(encoder), record.labels(setting.repetitions), identity))
    return samples


def collate(samples: list[Sample]):
    return Batch.from_data_list([item.graph for item in samples]), [item.labels for item in samples]


def masked_multitask_bce(logits, labels: list[RiskLabels], weights, epoch_counts, batches: int):
    terms = []
    for head, scores in logits.items():
        values = [getattr(item, head) for item in labels]
        keep = [index for index, value in enumerate(values) if observed(value)]
        if keep:
            target = scores.new_tensor([values[index] for index in keep])
            loss = F.binary_cross_entropy_with_logits(scores[keep], target, reduction="sum")
            terms.append(loss * weights[head] * batches / max(1, epoch_counts[head]))
    return torch.stack(terms).sum() if terms else None


@torch.no_grad()
def evaluate(model: RiskPredictor, samples: list[Sample], device: str) -> dict:
    model.eval()
    results = {"overall": ([], [])}
    for head in model.head_names:
        results[head] = ([], [])
    for graph, labels in DataLoader(samples, batch_size=BATCH_SIZE, collate_fn=collate):
        predictions = model.probabilities(graph.to(device))
        risk = torch.stack(tuple(predictions.values())).amax(0).cpu().tolist()
        results["overall"][0].extend(item.any_risk for item in labels)
        results["overall"][1].extend(risk)
        for head, scores in predictions.items():
            results[head][0].extend(getattr(item, head) for item in labels)
            results[head][1].extend(scores.cpu().tolist())
    return {head: detection_metrics(*values) for head, values in results.items()}


def save_training_state(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        torch.save(state, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def model_state(model: RiskPredictor) -> dict:
    return {name: weight.detach().cpu().clone() for name, weight in model.state_dict().items()}


def load_predictor(path: Path, device: str = "cpu") -> RiskPredictor:
    state = torch.load(path, weights_only=True, map_location="cpu")
    setting = state["setting"]
    model = RiskPredictor(setting["architecture"], tuple(setting["heads"]))
    model.load_state_dict(state["best_model"])
    return model.to(device).eval()


def train(
    training: list[Sample], validation: list[Sample], setting: ExperimentSetting,
    path: Path, *, device: str, seed: int, resume: bool = False,
) -> RiskPredictor:
    if not training or not validation:
        raise ValueError("Training and validation splits must be nonempty")
    counts = {
        head: sum(observed(getattr(sample.labels, head)) for sample in training)
        for head in setting.heads
    }
    if not any(counts.values()) or not any(sample.labels.any_risk == 1 for sample in validation):
        raise ValueError("Training requires observable labels and validation requires observable positives")
    signature = hashlib.sha256(json_text([
        [(sample.identity, asdict(sample.labels)) for sample in split] for split in (training, validation)
    ]).encode("utf-8")).hexdigest()
    torch.manual_seed(seed)
    model = RiskPredictor(setting.architecture, setting.heads).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    first_epoch = 0
    best_auprc = -1.0
    best_epoch = -1
    best_model = None
    history = []
    if resume:
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["setting"] != asdict(setting) or saved["seed"] != seed or saved["signature"] != signature:
            raise ValueError("Training checkpoint does not match the experiment or dataset")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        first_epoch = saved["epoch"] + 1
        best_auprc = saved["best_auprc"]
        best_epoch = saved["best_epoch"]
        best_model = saved["best_model"]
        history = saved["history"]
    weights = setting.weights()
    for epoch in range(first_epoch, MAX_EPOCHS):
        if epoch - best_epoch > PATIENCE and best_epoch >= 0:
            break
        torch.manual_seed(seed + epoch)
        generator = torch.Generator().manual_seed(seed + epoch)
        loader = DataLoader(training, batch_size=BATCH_SIZE, shuffle=True, generator=generator, collate_fn=collate)
        model.train()
        running_loss = 0.0
        for graph, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = masked_multitask_bce(model(graph.to(device)), labels, weights, counts, len(loader))
            if loss is not None:
                loss.backward()
                optimizer.step()
                running_loss += float(loss.detach())
        metrics = evaluate(model, validation, device)
        score = metrics["overall"]["auprc"]
        if score is None:
            raise ValueError("Validation AUPRC is undefined")
        if score > best_auprc:
            best_auprc = score
            best_model = model_state(model)
            best_epoch = epoch
        history.append({"epoch": epoch, "loss": running_loss / len(loader), "validation": metrics})
        save_training_state({
            "setting": asdict(setting), "signature": signature, "seed": seed, "epoch": epoch,
            "model": model_state(model), "optimizer": optimizer.state_dict(),
            "best_model": best_model, "best_auprc": best_auprc, "best_epoch": best_epoch,
            "history": history,
        }, path)
    if best_model is None:
        raise RuntimeError("Training produced no validation checkpoint")
    model.load_state_dict(best_model)
    return model.eval()
