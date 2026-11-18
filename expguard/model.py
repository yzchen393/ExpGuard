from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, HeteroConv, HGTConv, RGCNConv, SAGEConv, global_mean_pool
from transformers import AutoModel, AutoTokenizer

from .graph import ATTRIBUTE_DIM, EDGE_TYPES, NODE_TYPES

HEADS = ("attack", "task")
ENCODERS = ("hgtnn", "hetero-sage", "rgcn", "hetero-gat")


class BGEM3Encoder:
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cpu", batch_size: int = 8):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).requires_grad_(False).eval().to(device)
        if self.model.config.hidden_size != 1024:
            raise ValueError("BGE-M3 node embeddings must have 1024 dimensions")
        self.device = device
        self.batch_size = batch_size
        self.cache: OrderedDict[str, torch.Tensor] = OrderedDict()

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        if not texts:
            return torch.empty((0, 1024), dtype=torch.float32)
        vectors = {text: self.cache[text] for text in texts if text in self.cache}
        missing = list(dict.fromkeys(text for text in texts if text not in vectors))
        for start in range(0, len(missing), self.batch_size):
            chunk = missing[start:start + self.batch_size]
            tokens = self.tokenizer(
                chunk, padding=True, truncation=True, max_length=8192, return_tensors="pt",
            ).to(self.device)
            hidden = self.model(**tokens).last_hidden_state[:, 0].float()
            values = F.normalize(hidden, dim=-1).cpu()
            for text, value in zip(chunk, values, strict=True):
                vectors[text] = value.clone()
        for text, value in vectors.items():
            self.cache[text] = value
            self.cache.move_to_end(text)
        while len(self.cache) > 16384:
            self.cache.popitem(last=False)
        return torch.stack([vectors[text] for text in texts])


class RelationalLayer(nn.Module):
    def __init__(self, architecture: str):
        super().__init__()
        self.architecture = architecture
        if architecture == "hgtnn":
            self.convolution = HGTConv(256, 256, (list(NODE_TYPES), list(EDGE_TYPES)), heads=4)
        elif architecture == "rgcn":
            self.convolution = RGCNConv(256, 256, len(EDGE_TYPES))
        else:
            self.convolution = HeteroConv({
                relation: (
                    SAGEConv((256, 256), 256, root_weight=False, bias=False)
                    if architecture == "hetero-sage"
                    else GATConv((256, 256), 64, heads=4, add_self_loops=False, bias=False)
                )
                for relation in EDGE_TYPES
            }, aggr="mean")
            self.residual = nn.ModuleDict({kind: nn.Linear(256, 256) for kind in NODE_TYPES})

    def forward(self, nodes, edges):
        nonempty = {kind: values for kind, values in edges.items() if values.size(1)}
        if self.architecture == "hgtnn":
            result = self.convolution(nodes, nonempty)
            return {kind: result[kind] if result.get(kind) is not None else values for kind, values in nodes.items()}
        if self.architecture == "rgcn":
            sizes = [nodes[kind].size(0) for kind in NODE_TYPES]
            offsets = dict(zip(NODE_TYPES, (sum(sizes[:i]) for i in range(len(sizes))), strict=True))
            shifted, relation_ids = [], []
            for relation_index, kind in enumerate(EDGE_TYPES):
                if kind in nonempty:
                    left, _, right = kind
                    edge_index = nonempty[kind]
                    shift = edge_index.new_tensor([[offsets[left]], [offsets[right]]])
                    shifted.append(edge_index + shift)
                    relation_ids.append(edge_index.new_full((edge_index.size(1),), relation_index))
            homogeneous = torch.cat([nodes[kind] for kind in NODE_TYPES])
            updated = self.convolution(homogeneous, torch.cat(shifted, dim=1), torch.cat(relation_ids))
            return dict(zip(NODE_TYPES, F.relu(updated).split(sizes), strict=True))
        messages = self.convolution(nodes, nonempty)
        updated = {}
        for kind, values in nodes.items():
            residual = self.residual[kind](values)
            updated[kind] = F.relu(residual + messages[kind] if kind in messages else residual)
        return updated


class RiskPredictor(nn.Module):
    def __init__(self, architecture: str = "hgtnn", heads: tuple[str, ...] = HEADS):
        super().__init__()
        if architecture not in ENCODERS or not heads or len(set(heads)) != len(heads) or set(heads) - set(HEADS):
            raise ValueError("Invalid graph encoder or risk heads")
        self.architecture = architecture
        self.head_names = heads
        self.project = nn.ModuleDict({kind: nn.Linear(1024 + ATTRIBUTE_DIM, 256) for kind in NODE_TYPES})
        self.layers = nn.ModuleList(RelationalLayer(architecture) for _ in range(2))
        self.dropout = nn.Dropout(0.2)
        self.risk_heads = nn.ModuleDict({
            head: nn.Sequential(nn.Linear(768, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1))
            for head in heads
        })

    def forward(self, graph):
        nodes = {
            kind: self.project[kind](torch.cat((graph[kind].x, graph[kind].attributes), dim=-1))
            for kind in NODE_TYPES
        }
        for layer in self.layers:
            nodes = {kind: self.dropout(values) for kind, values in layer(nodes, graph.edge_index_dict).items()}
        pooled = global_mean_pool(
            torch.cat([nodes[kind] for kind in NODE_TYPES]),
            torch.cat([graph[kind].batch for kind in NODE_TYPES]),
            size=graph.num_graphs,
        )
        vector = torch.cat((
            nodes["experience"][graph["experience"].is_anchor],
            nodes["task"][graph["task"].is_anchor], pooled,
        ), dim=-1)
        return {name: head(vector).squeeze(-1) for name, head in self.risk_heads.items()}

    def probabilities(self, graph) -> dict[str, torch.Tensor]:
        return {name: logits.sigmoid() for name, logits in self(graph).items()}
