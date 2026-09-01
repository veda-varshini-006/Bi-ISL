"""Typed Hierarchical Configuration System for Bi-ISL.

Provides pydantic schema definitions for 11 configuration groups:
dataset, preprocessing, visualencoder, decoder, context, ugsa, training,
evaluation, mobile, avatar, and experiment.

No hyperparameter is hard-coded inside model source files.
Supports YAML loading, command-line overrides, validation of invalid hyperparameter
combinations, and dictionary/YAML exporting.
"""

import os
import yaml
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, model_validator


class DatasetConfig(BaseModel):
    dataset_name: str = "ISLTranslate"
    data_dir: str = "./data"
    split_manifest_path: str = "./data/splits/isltranslate_splits.json"
    num_classes: int = 263
    max_frames: int = 300
    target_vocab_size: int = 10000


class PreprocessingConfig(BaseModel):
    fps: float = 30.0
    img_size: int = 224
    norm_mean: List[float] = Field(default_factory=lambda: [0.485, 0.456, 0.406])
    norm_std: List[float] = Field(default_factory=lambda: [0.229, 0.224, 0.225])
    landmark_types: List[str] = Field(default_factory=lambda: ["pose", "hands", "face"])
    apply_augmentation: bool = True


class VisualEncoderConfig(BaseModel):
    backbone_type: str = "i3d_landmark_hybrid"
    hidden_dim: int = 512
    in_channels: int = 3
    num_layers: int = 6
    dropout: float = 0.1


class DecoderConfig(BaseModel):
    decoder_type: str = "transformer"
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 6
    vocab_size: int = 10000
    max_seq_len: int = 256

    @model_validator(mode="after")
    def validate_head_dim(self) -> "DecoderConfig":
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
            )
        return self


class ContextConfig(BaseModel):
    enable_context: bool = True
    sbds_dim: int = 256
    max_history_turns: int = 5
    reliability_threshold: float = 0.5
    context_gate_type: str = "evidence_gated"

    @model_validator(mode="after")
    def validate_context_gating(self) -> "ContextConfig":
        if not (0.0 <= self.reliability_threshold <= 1.0):
            raise ValueError("reliability_threshold must be between 0.0 and 1.0")
        return self


class UGSAConfig(BaseModel):
    enable_ugsa: bool = True
    uncertainty_threshold: float = 0.3
    max_update_steps: int = 10
    learning_rate: float = 0.0001
    protected_rollback_drop_tolerance: float = 0.02

    @model_validator(mode="after")
    def validate_rollback_tolerance(self) -> "UGSAConfig":
        if self.protected_rollback_drop_tolerance < 0.0 or self.protected_rollback_drop_tolerance > 0.5:
            raise ValueError("protected_rollback_drop_tolerance must be between 0.0 and 0.5")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be greater than 0.0")
        return self


class TrainingConfig(BaseModel):
    seed: int = 42
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    clip_grad: float = 1.0

    @model_validator(mode="after")
    def validate_training_params(self) -> "TrainingConfig":
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        return self


class EvaluationConfig(BaseModel):
    primary_metrics: List[str] = Field(default_factory=lambda: ["bleu4", "chrf", "bertscore"])
    secondary_metrics: List[str] = Field(default_factory=lambda: ["ger", "rouge_l", "fps"])
    compute_usr: bool = True
    evaluate_context_attack: bool = True


class MobileConfig(BaseModel):
    quantization_format: str = "int8"
    target_device: str = "android"
    max_p95_latency_ms: float = 200.0
    max_ram_mb: float = 500.0
    export_onnx: bool = True


class AvatarConfig(BaseModel):
    enable_avatar: bool = True
    rig_type: str = "humanoid_rig"
    fps: float = 60.0
    non_manual_markers_enabled: bool = True
    blendshape_count: int = 52


class ExperimentConfig(BaseModel):
    exp_id: str = "E1"
    title: str = "Baseline SLT Reproduction"
    output_dir: str = "./artifacts/e1_baseline"
    save_checkpoints: bool = True
    seed: int = 42


class BiISLConfig(BaseModel):
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    visualencoder: VisualEncoderConfig = Field(default_factory=VisualEncoderConfig)
    decoder: DecoderConfig = Field(default_factory=DecoderConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    ugsa: UGSAConfig = Field(default_factory=UGSAConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    mobile: MobileConfig = Field(default_factory=MobileConfig)
    avatar: AvatarConfig = Field(default_factory=AvatarConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    def to_yaml(self) -> str:
        return yaml.dump(self.to_dict(), default_flow_style=False)


def apply_override(data: Dict[str, Any], override_str: str) -> None:
    """Apply dot-notation override (e.g. 'training.batch_size=64')."""
    if "=" not in override_str:
        return
    key_path, value_str = override_str.split("=", 1)
    keys = key_path.strip().split(".")

    value: Any = value_str.strip()
    if value.lower() == "true":
        value = True
    elif value.lower() == "false":
        value = False
    else:
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass

    current = data
    for k in keys[:-1]:
        if k not in current or not isinstance(current[k], dict):
            current[k] = {}
        current = current[k]
    current[keys[-1]] = value


def load_config(yaml_path: Optional[str] = None, overrides: Optional[List[str]] = None) -> BiISLConfig:
    """Load configuration from YAML file and apply optional command-line overrides."""
    data: Dict[str, Any] = {}
    if yaml_path and os.path.exists(yaml_path):
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

    if overrides:
        for override in overrides:
            apply_override(data, override)

    return BiISLConfig.model_validate(data)
