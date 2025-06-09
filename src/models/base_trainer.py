from __future__ import annotations

"""Base classes and registry for TTS models."""

from dataclasses import dataclass
from typing import Dict, Optional, Type, List
from abc import ABC, abstractmethod


@dataclass
class TrainingResult:
    success: bool
    model_path: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    error: Optional[str] = None


@dataclass
class InferenceResult:
    success: bool
    audio_path: Optional[str] = None
    error: Optional[str] = None


class BaseTrainer(ABC):
    """Abstract base class for all model trainers."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.config_path = config_path

    @abstractmethod
    async def train(self, dataset_path: str, output_dir: str) -> TrainingResult:
        """Train the model with the given dataset."""

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        reference_audio: Optional[str] = None,
        output_path: Optional[str] = None,
        streaming: bool = False,
    ) -> InferenceResult:
        """Generate speech from text."""


class ModelRegistry:
    """Registry for available model trainers."""

    _registry: Dict[str, Type[BaseTrainer]] = {}

    @classmethod
    def register(cls, model_type: str, trainer_cls: Type[BaseTrainer]) -> None:
        cls._registry[model_type] = trainer_cls

    @classmethod
    def get_trainer(cls, model_type: str) -> BaseTrainer:
        if model_type not in cls._registry:
            raise ValueError(f"Model not registered: {model_type}")
        trainer_cls = cls._registry[model_type]
        return trainer_cls()

    @classmethod
    def list_available_models(cls) -> List[str]:
        return sorted(cls._registry.keys())

    @classmethod
    def detect_model_type(cls, identifier: str) -> str:
        ident = identifier.lower()
        if "xtts" in ident:
            return "xtts_v2"
        if "vits" in ident:
            return "vits"
        if "tortoise" in ident:
            return "tortoise"
        raise ValueError("Unable to detect model type")

    @classmethod
    def load_model(cls, model_type: str, identifier: str) -> BaseTrainer:
        trainer = cls.get_trainer(model_type)
        # Concrete trainers may implement actual loading logic. Here we just
        # return the instantiated trainer.
        return trainer
