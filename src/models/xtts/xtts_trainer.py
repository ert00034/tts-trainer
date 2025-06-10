from __future__ import annotations

"""XTTS v2 trainer using Coqui TTS for voice cloning."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import soundfile as sf
from TTS.api import TTS

from ..base_trainer import BaseTrainer, InferenceResult, ModelRegistry, TrainResult


@ModelRegistry.register("xtts_v2")
class XTTSTrainer(BaseTrainer):
    """Trainer and inference helper for the XTTS v2 model."""

    def __init__(self, model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.tts: Optional[TTS] = None
        self.character_voices: Dict[str, str] = {}

    def initialize_model(self) -> None:
        if self.tts is None:
            self.tts = TTS(self.model_name, gpu=self.device.startswith("cuda"))

    def load_character_voices(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            self.character_voices = json.load(f)

    def list_available_characters(self) -> List[str]:
        return list(self.character_voices.keys())

    async def train(
        self,
        dataset_path: str,
        output_path: str,
        config: Optional[Dict[str, Any]] = None,
        resume_from: Optional[str] = None,
    ) -> TrainResult:
        # Training not implemented in this demo
        return TrainResult(success=False, model_path="", final_metrics={}, error="Training not implemented")

    async def synthesize(
        self,
        text: str,
        reference_audio: Optional[str] = None,
        character: Optional[str] = None,
        output_path: str = "output.wav",
        streaming: bool = False,
    ) -> InferenceResult:
        try:
            self.initialize_model()
            if character and character in self.character_voices:
                reference_audio = self.character_voices[character]

            start = time.time()
            if reference_audio:
                wav = self.tts.tts(text, speaker_wav=reference_audio, language="en")
            else:
                wav = self.tts.tts(text, language="en")

            sf.write(output_path, wav, self.tts.synthesizer.output_sample_rate)
            return InferenceResult(success=True, audio_path=output_path, generation_time=time.time() - start)
        except Exception as e:
            return InferenceResult(success=False, audio_path="", generation_time=0.0, error=str(e))
