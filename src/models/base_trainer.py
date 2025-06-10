from __future__ import annotations

"""Base classes and model registry for TTS trainers."""

import abc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import yaml


@dataclass
class TrainResult:
    """Result returned by training methods."""

    success: bool
    model_path: Optional[str] = None
    final_metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class InferenceResult:
    """Result returned by synthesis methods."""

    success: bool
    audio_path: Optional[str] = None
    generation_time: float = 0.0
    error: Optional[str] = None


class BaseTrainer(abc.ABC):
    """Abstract base class for all TTS trainers."""

    model_name: str = "base"

    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load YAML configuration if provided."""
        if config_path and Path(config_path).exists():
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    @abc.abstractmethod
    async def train(
        self,
        dataset_path: str,
        output_path: str,
        config: Optional[Dict[str, Any]] = None,
        resume_from: Optional[str] = None,
    ) -> TrainResult:
        """Train a model using the provided dataset."""

    @abc.abstractmethod
    async def synthesize(
        self,
        text: str,
        reference_audio: Optional[str] = None,
        character: Optional[str] = None,
        output_path: str = "output.wav",
        streaming: bool = False,
    ) -> InferenceResult:
        """Generate speech from text."""


class ModelRegistry:
    """Registry for available TTS trainers."""

    _registry: Dict[str, Type[BaseTrainer]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(trainer_cls: Type[BaseTrainer]):
            cls._registry[name] = trainer_cls
            trainer_cls.model_name = name
            return trainer_cls

        return decorator

    @classmethod
    def get_trainer(cls, name: str) -> BaseTrainer:
        if name not in cls._registry:
            raise ValueError(f"Unknown model type: {name}")
        return cls._registry[name]()

    @classmethod
    def list_available_models(cls) -> List[str]:
        return list(cls._registry.keys())

    @classmethod
    def detect_model_type(cls, model_path: str) -> str:
        path = Path(model_path)
        if path.suffix == ".json":
            return "xtts_v2"
        return "xtts_v2"

    @classmethod
    def load_model(cls, model_type: str, model_path: str) -> BaseTrainer:
        trainer = cls.get_trainer(model_type)
        if hasattr(trainer, "load_character_voices") and model_path.endswith(".json"):
            trainer.load_character_voices(model_path)
        elif hasattr(trainer, "load"):
            trainer.load(model_path)
        return trainer
