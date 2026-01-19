# Model configuration and calibration dataclasses
# Re-exported from the main package for demo use

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ModelConfig:
    """Configuration for a trained early exit model."""

    model_name: str
    num_heads: int
    head_layer_indices: List[int]
    quantization: str  # "none", "4bit", "8bit"
    hidden_size: int
    vocab_size: int
    num_hidden_layers: int
    training_config: Optional[Dict] = None

    @classmethod
    def from_json(cls, path: str) -> "ModelConfig":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(
            model_name=data["model_name"],
            num_heads=data["num_heads"],
            head_layer_indices=data["head_layer_indices"],
            quantization=data["quantization"],
            hidden_size=data["hidden_size"],
            vocab_size=data["vocab_size"],
            num_hidden_layers=data["num_hidden_layers"],
            training_config=data.get("training_config"),
        )


@dataclass
class CalibrationResult:
    """Calibration results with thresholds per head per accuracy level."""

    model_config_path: str
    calibration_dataset: str
    calibration_samples: int
    uncertainty_metric: str  # "entropy" or "confidence"
    accuracy_levels: List[float]
    thresholds: Dict[str, Dict[str, float]] = field(default_factory=dict)
    statistics: Dict[str, Dict] = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: str) -> "CalibrationResult":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    def get_thresholds_for_level(self, accuracy_level: float) -> Dict[int, float]:
        """Get all thresholds for a given accuracy level."""
        level_key = f"{accuracy_level:.2f}"
        return {int(k): v for k, v in self.thresholds[level_key].items()}
