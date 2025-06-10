from __future__ import annotations

"""XTTS v2 trainer using Coqui TTS for voice cloning."""

import json
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import defaultdict

import soundfile as sf
import torch
import torchaudio
import librosa
import numpy as np
from TTS.api import TTS

from ..base_trainer import BaseTrainer, InferenceResult, ModelRegistry, TrainResult

logger = logging.getLogger(__name__)


@ModelRegistry.register("xtts_v2")
class XTTSTrainer(BaseTrainer):
    """Trainer and inference helper for the XTTS v2 model."""

    def __init__(self, model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2", device: str = "cuda"):
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.tts: Optional[TTS] = None
        self.character_voices: Dict[str, str] = {}
        
        # Character-specific synthesis parameters
        self.character_synthesis_params = {
            'ash': {'temperature': 0.7, 'speed': 1.1},
            'brock': {'temperature': 0.5, 'speed': 0.9},
            'misty': {'temperature': 0.8, 'speed': 1.0},
            'jessie': {'temperature': 0.9, 'speed': 1.0},
            'james': {'temperature': 0.8, 'speed': 0.95},
            'meowth': {'temperature': 0.9, 'speed': 1.2},
            'narrator': {'temperature': 0.3, 'speed': 0.85},
            'pokédex': {'temperature': 0.1, 'speed': 0.8}
        }

    def initialize_model(self) -> None:
        if self.tts is None:
            logger.info(f"Initializing XTTS model: {self.model_name}")
            self.tts = TTS(self.model_name).to(self.device)
            logger.info(f"✅ XTTS model loaded successfully on {self.device}")

    def load_character_voices(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            self.character_voices = json.load(f)

    def list_available_characters(self) -> List[str]:
        return list(self.character_voices.keys())

    def parse_manual_references(self, refs_file: str) -> Dict[str, str]:
        """Parse manual references file into character:audio_path mapping."""
        character_refs = defaultdict(list)  # Changed to collect multiple refs per character
        refs_path = Path(refs_file)
        
        if not refs_path.exists():
            raise FileNotFoundError(f"Manual references file not found: {refs_file}")
        
        with open(refs_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                if ':' not in line:
                    logger.warning(f"Line {line_num}: Invalid format (missing ':'): {line}")
                    continue
                
                character, audio_path = line.split(':', 1)
                character = character.strip()
                audio_path = audio_path.strip()
                
                # Normalize path separators for cross-platform compatibility
                audio_path = Path(audio_path).as_posix()
                
                if not Path(audio_path).exists():
                    logger.warning(f"Line {line_num}: Audio file not found: {audio_path}")
                    continue
                
                character_refs[character].append(audio_path)
                logger.debug(f"Added reference: {character} -> {audio_path}")
        
        # Process multiple references per character
        final_refs = {}
        for character, audio_files in character_refs.items():
            if len(audio_files) == 1:
                final_refs[character] = audio_files[0]
                logger.info(f"Using single reference for {character}: {Path(audio_files[0]).name}")
            else:
                # Concatenate multiple references
                concatenated_path = self._concatenate_references(character, audio_files)
                final_refs[character] = concatenated_path
                logger.info(f"Concatenated {len(audio_files)} references for {character}: {Path(concatenated_path).name}")
        
        logger.info(f"Parsed {len(final_refs)} character references from {refs_file}")
        return final_refs

    def _concatenate_references(self, character: str, audio_files: List[str]) -> str:
        """Concatenate multiple reference audio files for a character."""
        import librosa
        import soundfile as sf
        from collections import defaultdict
        
        logger.info(f"🔗 Concatenating {len(audio_files)} reference files for {character}")
        
        # Create output directory
        concat_dir = Path("temp/concatenated_references")
        concat_dir.mkdir(parents=True, exist_ok=True)
        
        # Output file path
        output_path = concat_dir / f"{character}_concatenated_reference.wav"
        
        # Load and concatenate audio files
        concatenated_audio = []
        target_sr = 22050  # Standard sample rate for XTTS
        total_duration = 0
        
        for i, audio_file in enumerate(audio_files):
            try:
                logger.debug(f"  Loading file {i+1}/{len(audio_files)}: {Path(audio_file).name}")
                
                # Load audio
                audio, sr = librosa.load(audio_file, sr=target_sr)
                duration = len(audio) / sr
                total_duration += duration
                
                # Add a small silence gap between clips (0.3 seconds)
                if i > 0:
                    silence_samples = int(0.3 * target_sr)
                    concatenated_audio.extend([0.0] * silence_samples)
                
                # Add the audio
                concatenated_audio.extend(audio.tolist())
                
                logger.debug(f"    Duration: {duration:.2f}s")
                
            except Exception as e:
                logger.warning(f"  Failed to load {audio_file}: {e}")
                continue
        
        if not concatenated_audio:
            raise ValueError(f"No valid audio files found for {character}")
        
        # Convert to numpy array and save
        import numpy as np
        concatenated_array = np.array(concatenated_audio, dtype=np.float32)
        
        # Save concatenated audio
        sf.write(output_path, concatenated_array, target_sr)
        
        logger.info(f"  ✅ Concatenated audio saved: {output_path}")
        logger.info(f"  📊 Total duration: {total_duration:.2f}s from {len(audio_files)} files")
        
        return str(output_path)

    def create_character_voice_profile(self, character: str, reference_audio: str) -> Dict[str, Any]:
        """Create a voice profile for a character using reference audio."""
        self.initialize_model()
        
        if not Path(reference_audio).exists():
            raise FileNotFoundError(f"Reference audio not found: {reference_audio}")
        
        logger.info(f"🎭 Creating voice profile for {character}")
        logger.info(f"🎯 Using reference: {Path(reference_audio).name}")
        
        # Get character-specific synthesis parameters
        synth_params = self.character_synthesis_params.get(character, {
            'temperature': 0.7,
            'speed': 1.0
        })
        
        # Test synthesis
        test_text = self._get_character_test_text(character)
        start_time = time.time()
        
        try:
            # Synthesize test audio
            audio_output = self.tts.tts(
                text=test_text,
                speaker_wav=reference_audio,
                language="en"
            )
            synthesis_time = time.time() - start_time
            
            # Handle audio format
            if isinstance(audio_output, list):
                audio_tensor = torch.tensor(audio_output, dtype=torch.float32)
            else:
                audio_tensor = torch.from_numpy(audio_output).float()
            
            # Save test audio
            output_dir = Path("artifacts/character_voices")
            output_dir.mkdir(parents=True, exist_ok=True)
            test_audio_path = output_dir / f"{character}_test.wav"
            
            torchaudio.save(test_audio_path, audio_tensor.unsqueeze(0), 22050)
            
            logger.info(f"✅ Voice profile created in {synthesis_time:.2f}s")
            logger.info(f"🎵 Test audio saved: {test_audio_path}")
            
            # Create profile
            profile = {
                'character': character,
                'reference_audio': reference_audio,
                'synthesis_params': synth_params,
                'test_synthesis_time': synthesis_time,
                'test_audio_path': str(test_audio_path),
                'test_text': test_text,
                'metrics': {
                    'synthesis_speed': synthesis_time,
                    'voice_quality': 'good'
                },
                'created_at': time.strftime('%Y-%m-%dT%H:%M:%S')
            }
            
            return profile
            
        except Exception as e:
            logger.error(f"❌ Voice profile creation failed for {character}: {e}")
            raise

    def _get_character_test_text(self, character: str) -> str:
        """Get character-appropriate test text."""
        test_texts = {
            'ash': "Pikachu, I choose you! We're gonna be the very best, like no one ever was!",
            'brock': "Pokemon are amazing creatures. I want to be the best Pokemon breeder in the world!",
            'misty': "Water Pokemon are the most beautiful! Psyduck, come back here!",
            'jessie': "Prepare for trouble! Team Rocket blasts off at the speed of light!",
            'james': "Make it double! To protect the world from devastation!",
            'meowth': "Meowth, that's right! The boss is gonna love this plan!",
            'narrator': "Our heroes continue their journey through the world of Pokemon, seeking new adventures.",
            'pokédex': "Pokemon data entry complete. Analysis shows this species has unique characteristics."
        }
        return test_texts.get(character, "Hello, this is a test of the text to speech system.")

    async def train(
        self,
        dataset_path: str,
        output_path: str,
        config: Optional[Dict[str, Any]] = None,
        resume_from: Optional[str] = None,
    ) -> TrainResult:
        """Train/Setup XTTS model with character voice profiles."""
        try:
            logger.info(f"🏋️ Setting up XTTS model with dataset: {dataset_path}")
            
            dataset_path_obj = Path(dataset_path)
            output_path_obj = Path(output_path)
            output_path_obj.mkdir(parents=True, exist_ok=True)
            
            # Check if dataset_path is a manual references file
            if dataset_path_obj.is_file() and dataset_path_obj.suffix == '.txt':
                logger.info("📝 Detected manual references file, parsing character references...")
                character_refs = self.parse_manual_references(dataset_path)
                
                if not character_refs:
                    return TrainResult(
                        success=False,
                        error="No valid character references found in manual references file"
                    )
                
                # Create voice profiles for each character
                character_profiles = {}
                
                for character, reference_audio in character_refs.items():
                    try:
                        profile = self.create_character_voice_profile(character, reference_audio)
                        character_profiles[character] = profile
                        
                        # Update character voices mapping
                        self.character_voices[character] = reference_audio
                        
                    except Exception as e:
                        logger.error(f"Failed to create profile for {character}: {e}")
                        continue
                
                if not character_profiles:
                    return TrainResult(
                        success=False,
                        error="Failed to create any character voice profiles"
                    )
                
                # Save character voice mappings
                character_voices_file = output_path_obj / "character_voices.json"
                with open(character_voices_file, 'w', encoding='utf-8') as f:
                    json.dump(self.character_voices, f, indent=2, ensure_ascii=False)
                
                # Save detailed profiles
                profiles_file = output_path_obj / "character_profiles.json"
                with open(profiles_file, 'w', encoding='utf-8') as f:
                    json.dump(character_profiles, f, indent=2, ensure_ascii=False, default=str)
                
                logger.info(f"✅ Character voice setup completed!")
                logger.info(f"📁 Character mappings saved: {character_voices_file}")
                logger.info(f"📁 Detailed profiles saved: {profiles_file}")
                
                # Calculate metrics
                total_synthesis_time = sum(p['test_synthesis_time'] for p in character_profiles.values())
                avg_synthesis_time = total_synthesis_time / len(character_profiles)
                
                return TrainResult(
                    success=True,
                    model_path=str(character_voices_file),
                    final_metrics={
                        'characters_trained': len(character_profiles),
                        'total_synthesis_time': total_synthesis_time,
                        'avg_synthesis_time': avg_synthesis_time,
                        'character_list': list(character_profiles.keys())
                    }
                )
            
            else:
                # Handle directory-based training (not implemented for this version)
                return TrainResult(
                    success=False,
                    error="Directory-based training not implemented. Please use manual references file."
                )
                
        except Exception as e:
            logger.error(f"Training failed with exception: {e}")
            return TrainResult(
                success=False,
                error=f"Training failed: {str(e)}"
            )

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
            
            # Use character voice if specified
            if character and character in self.character_voices:
                reference_audio = self.character_voices[character]
                logger.info(f"🎭 Using {character} voice: {Path(reference_audio).name}")
            
            if not reference_audio:
                return InferenceResult(
                    success=False,
                    error="No reference audio specified and no character voice available"
                )

            start = time.time()
            
            # Get character-specific synthesis parameters
            synth_params = {}
            if character in self.character_synthesis_params:
                synth_params = {k: v for k, v in self.character_synthesis_params[character].items() 
                              if k in ['temperature']}  # Only include supported parameters
            
            # Synthesize with character parameters
            wav = self.tts.tts(
                text=text,
                speaker_wav=reference_audio,
                language="en",
                **synth_params
            )

            # Save audio
            sf.write(output_path, wav, self.tts.synthesizer.output_sample_rate)
            
            generation_time = time.time() - start
            logger.info(f"🎵 Audio generated in {generation_time:.2f}s: {output_path}")
            
            return InferenceResult(
                success=True, 
                audio_path=output_path, 
                generation_time=generation_time
            )
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return InferenceResult(
                success=False, 
                audio_path="", 
                generation_time=0.0, 
                error=str(e)
            )
