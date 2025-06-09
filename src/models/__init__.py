"""Model registry and trainer implementations."""

from .base_trainer import BaseTrainer, ModelRegistry, TrainingResult, InferenceResult

# Import trainer implementations to register them if available
try:
    from .xtts.xtts_trainer import XTTSTrainer
    ModelRegistry.register("xtts_v2", XTTSTrainer)
except Exception:  # pragma: no cover - optional dependency may be missing
    pass
