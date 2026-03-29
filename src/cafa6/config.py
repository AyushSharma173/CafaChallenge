"""Experiment configuration management."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class EmbeddingConfig:
    model_name: str = "esm2_t33_650M_UR50D"
    repr_layer: int = 33
    batch_size: int = 4
    max_seq_len: int = 1022
    dtype: str = "float16"


@dataclass
class LightGBMConfig:
    n_estimators: int = 500
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_child_samples: int = 20


@dataclass
class MLPConfig:
    hidden_dims: list[int] = field(default_factory=lambda: [2048, 1024])
    dropout: float = 0.3
    lr: float = 0.001
    epochs: int = 30
    batch_size: int = 256


@dataclass
class Config:
    # Paths
    data_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    embeddings_dir: str = "data/embeddings"
    models_dir: str = "models"
    submissions_dir: str = "submissions"

    # Competition
    competition_slug: str = "cafa-6-protein-function-prediction"
    ontologies: list[str] = field(default_factory=lambda: ["P", "F", "C"])

    # Training
    min_term_count: int = 50
    val_fraction: float = 0.1
    seed: int = 42

    # Model
    model_type: str = "lightgbm"

    # Sub-configs
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    lightgbm: LightGBMConfig = field(default_factory=LightGBMConfig)
    mlp: MLPConfig = field(default_factory=MLPConfig)

    # Submission
    propagate: bool = True
    min_confidence: float = 0.01

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f)

        # Flatten nested structure from YAML
        cfg = cls()
        if "paths" in raw:
            for k, v in raw["paths"].items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        if "competition" in raw:
            if "slug" in raw["competition"]:
                cfg.competition_slug = raw["competition"]["slug"]
            if "ontologies" in raw["competition"]:
                cfg.ontologies = raw["competition"]["ontologies"]
        if "training" in raw:
            for k, v in raw["training"].items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        if "embeddings" in raw:
            cfg.embeddings = EmbeddingConfig(**raw["embeddings"])
        if "model" in raw:
            cfg.model_type = raw["model"].get("type", cfg.model_type)
            if "lightgbm" in raw["model"]:
                cfg.lightgbm = LightGBMConfig(**raw["model"]["lightgbm"])
            if "mlp" in raw["model"]:
                cfg.mlp = MLPConfig(**raw["model"]["mlp"])
        if "submission" in raw:
            cfg.propagate = raw["submission"].get("propagate", cfg.propagate)
            cfg.min_confidence = raw["submission"].get("min_confidence", cfg.min_confidence)

        return cfg
