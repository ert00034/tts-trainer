from __future__ import annotations

"""IndexTTS trainer for English voice cloning."""

import json
import time
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import tempfile

import soundfile as sf
import torch
import torchaudio
import librosa
import numpy as np

from ..base_trainer import BaseTrainer, InferenceResult, ModelRegistry, TrainResult

logger = logging.getLogger(__name__)


@ModelRegistry.register("indextts")
class IndexTTSTrainer(BaseTrainer):
    """Trainer and inference helper for the IndexTTS model (English-focused)."""

    def __init__(self, 
                 model_dir: str = "checkpoints/IndexTTS-1.5", 
                 config_path: str = None,
                 device: str = "cuda"):
        self.model_dir = Path(model_dir)
        self.config_path = config_path
        self.device = device if torch.cuda.is_available() else "cpu"
        self.tts = None
        self.character_voices: Dict[str, str] = {}
        self._setup_indextts()
        
        # Character-specific synthesis parameters (English optimized)
        self.character_synthesis_params = {
            'meowth': {'temperature': 0.9, 'speed': 1.2},
            'ash': {'temperature': 0.7, 'speed': 1.1},
            'brock': {'temperature': 0.5, 'speed': 0.9},
            'misty': {'temperature': 0.8, 'speed': 1.0},
            'jessie': {'temperature': 0.9, 'speed': 1.0},
            'james': {'temperature': 0.8, 'speed': 0.95},
            'narrator': {'temperature': 0.3, 'speed': 0.85},
        }

    def _setup_indextts(self) -> None:
        """Setup IndexTTS environment and check installation."""
        try:
            # Try to import IndexTTS
            self._import_indextts()
            logger.info("✅ IndexTTS found and imported successfully")
        except ImportError as e:
            logger.warning(f"IndexTTS not found: {e}")
            logger.info("🔧 Attempting to install IndexTTS...")
            self._install_indextts()

    def _import_indextts(self):
        """Import IndexTTS components."""
        try:
            from indextts.infer import IndexTTS
            self.IndexTTS = IndexTTS
            logger.debug("IndexTTS import successful")
        except ImportError:
            # Try alternative import paths
            try:
                # Add the cloned repository to Python path
                repo_path = Path("checkpoints/index-tts")
                if repo_path.exists():
                    import sys
                    sys.path.insert(0, str(repo_path))
                
                from indextts.infer import IndexTTS
                self.IndexTTS = IndexTTS
                logger.debug("IndexTTS import successful (with path adjustment)")
            except ImportError as e:
                raise ImportError(f"Could not import IndexTTS: {e}")

    def _install_indextts(self) -> None:
        """Install IndexTTS if not available."""
        try:
            logger.info("📦 Installing IndexTTS...")
            
            # First, try to clone the repository
            import subprocess
            
            # Create checkpoints directory if it doesn't exist
            self.model_dir.parent.mkdir(parents=True, exist_ok=True)
            
            # Clone IndexTTS repository
            repo_dir = self.model_dir.parent / "index-tts"
            if not repo_dir.exists():
                logger.info("🔄 Cloning IndexTTS repository...")
                subprocess.run([
                    "git", "clone", 
                    "https://github.com/index-tts/index-tts.git",
                    str(repo_dir)
                ], check=True)
            
            # Install IndexTTS package
            logger.info("📥 Installing IndexTTS package...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-e", str(repo_dir)
            ], check=True)
            
            # Import after installation
            self._import_indextts()
            
            logger.info("✅ IndexTTS installation completed")
            
        except Exception as e:
            logger.error(f"❌ Failed to install IndexTTS: {e}")
            logger.info("🔧 Manual installation required. Please run:")
            logger.info("   git clone https://github.com/index-tts/index-tts.git")
            logger.info("   cd index-tts")
            logger.info("   pip install -e .")
            raise RuntimeError(f"IndexTTS installation failed: {e}")

    def _download_models(self) -> None:
        """Download IndexTTS models if they don't exist."""
        if not self.model_dir.exists():
            logger.info(f"📥 Downloading IndexTTS models to {self.model_dir}...")
            self.model_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                # Use huggingface-cli to download models
                import subprocess
                subprocess.run([
                    "huggingface-cli", "download", "IndexTeam/IndexTTS-1.5",
                    "config.yaml", "bigvgan_discriminator.pth", "bigvgan_generator.pth", 
                    "bpe.model", "dvae.pth", "gpt.pth", "unigram_12000.vocab",
                    "--local-dir", str(self.model_dir)
                ], check=True)
                
                logger.info("✅ IndexTTS models downloaded successfully")
                
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Failed to download models: {e}")
                logger.info("🔧 Manual model download required:")
                logger.info(f"   huggingface-cli download IndexTeam/IndexTTS-1.5 --local-dir {self.model_dir}")
                raise RuntimeError(f"Model download failed: {e}")

    def initialize_model(self) -> None:
        """Initialize the IndexTTS model."""
        if self.tts is None:
            logger.info(f"🔄 Initializing IndexTTS model from {self.model_dir}")
            
            # Ensure models are downloaded
            if not (self.model_dir / "gpt.pth").exists():
                self._download_models()
            
            try:
                # Initialize IndexTTS
                config_path = self.config_path or str(self.model_dir / "config.yaml")
                
                self.tts = self.IndexTTS(
                    model_dir=str(self.model_dir),
                    cfg_path=config_path
                )
                
                logger.info(f"✅ IndexTTS model loaded successfully on {self.device}")
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize IndexTTS: {e}")
                raise RuntimeError(f"IndexTTS initialization failed: {e}")

    def load_character_voices(self, path: str) -> None:
        """Load character voice profiles from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            self.character_voices = json.load(f)
        logger.info(f"📚 Loaded {len(self.character_voices)} character voices from {path}")

    def list_available_characters(self) -> List[str]:
        """List available character voices."""
        return list(self.character_voices.keys())

    async def train(
        self,
        dataset_path: str,
        output_path: str,
        config: Optional[Dict[str, Any]] = None,
        resume_from: Optional[str] = None,
    ) -> TrainResult:
        """
        Train IndexTTS model.
        
        Note: IndexTTS typically doesn't require training for voice cloning.
        This method creates voice profiles instead.
        """
        logger.info("🏋️ Starting IndexTTS voice profile creation...")
        
        dataset_path = Path(dataset_path)
        output_path = Path(output_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        
        # Ensure model is initialized
        self.initialize_model()
        
        # Create voice profiles for characters found in dataset
        voice_profiles = {}
        
        # Scan for character directories
        for char_dir in dataset_path.iterdir():
            if char_dir.is_dir():
                character = char_dir.name
                logger.info(f"🎭 Processing character: {character}")
                
                # Find audio files in character directory
                audio_files = []
                for ext in ['*.wav', '*.mp3', '*.flac']:
                    audio_files.extend(char_dir.glob(ext))
                
                if audio_files:
                    # Use the first suitable audio file as reference
                    reference_audio = str(audio_files[0])
                    logger.info(f"  📄 Using reference: {Path(reference_audio).name}")
                    
                    # Test synthesis to validate the voice
                    test_text = "Hello, this is a test of my voice."
                    try:
                        result = await self.synthesize(
                            text=test_text,
                            reference_audio=reference_audio,
                            output_path=str(output_path / f"{character}_test.wav")
                        )
                        
                        if result.success:
                            voice_profiles[character] = {
                                'reference_audio': reference_audio,
                                'test_synthesis_time': result.generation_time,
                                'parameters': self.character_synthesis_params.get(character, {
                                    'temperature': 0.7,
                                    'speed': 1.0
                                })
                            }
                            logger.info(f"  ✅ Voice profile created for {character}")
                        else:
                            logger.warning(f"  ❌ Failed to create voice profile for {character}")
                            
                    except Exception as e:
                        logger.warning(f"  ❌ Error testing {character}: {e}")
                else:
                    logger.warning(f"  ❌ No audio files found for {character}")
        
        # Save voice profiles
        output_path.mkdir(parents=True, exist_ok=True)
        profiles_file = output_path / "indextts_voice_profiles.json"
        
        with open(profiles_file, 'w', encoding='utf-8') as f:
            json.dump(voice_profiles, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Saved voice profiles to {profiles_file}")
        
        return TrainResult(
            success=True,
            model_path=str(profiles_file),
            metrics={
                'characters_processed': len(voice_profiles),
                'total_characters_found': len(list(dataset_path.iterdir())),
            },
            training_time=0.0,  # No actual training time
            message=f"Created voice profiles for {len(voice_profiles)} characters"
        )

    async def synthesize(
        self,
        text: str,
        reference_audio: Optional[str] = None,
        character: Optional[str] = None,
        output_path: str = "output.wav",
        streaming: bool = False,
    ) -> InferenceResult:
        """
        Synthesize speech using IndexTTS.
        
        Args:
            text: Text to synthesize
            reference_audio: Path to reference audio file for voice cloning
            character: Character name (if using pre-loaded voice profiles)
            output_path: Where to save the generated audio
            streaming: Whether to use streaming (not implemented for IndexTTS yet)
        """
        logger.info(f"🎙️ Synthesizing text with IndexTTS: '{text[:50]}...'")
        
        # Ensure model is initialized
        self.initialize_model()
        
        # Determine reference audio
        if character and character in self.character_voices:
            reference_audio = self.character_voices[character]['reference_audio']
            logger.info(f"🎭 Using character voice: {character}")
        elif not reference_audio:
            raise ValueError("Either 'reference_audio' or 'character' must be provided")
        
        # Validate reference audio
        if not Path(reference_audio).exists():
            raise FileNotFoundError(f"Reference audio not found: {reference_audio}")
        
        logger.info(f"🎯 Using reference audio: {Path(reference_audio).name}")
        
        start_time = time.time()
        
        try:
            # Get character-specific parameters
            char_params = self.character_synthesis_params.get(character, {
                'temperature': 0.7,
                'speed': 1.0
            })
            
            # Synthesize using IndexTTS
            logger.info(f"🔄 Generating audio...")
            
            # Use IndexTTS inference
            audio_output = self.tts.infer(
                audio_prompt=reference_audio,
                text=text,
                output_path=output_path
            )
            
            generation_time = time.time() - start_time
            
            # Verify output file was created
            if Path(output_path).exists():
                # Get audio info
                with sf.SoundFile(output_path) as f:
                    duration = len(f) / f.samplerate
                    sample_rate = f.samplerate
                
                logger.info(f"✅ Audio generated successfully!")
                logger.info(f"📊 Duration: {duration:.2f}s, Sample rate: {sample_rate}Hz")
                logger.info(f"⏱️ Generation time: {generation_time:.2f}s")
                logger.info(f"💾 Saved to: {output_path}")
                
                return InferenceResult(
                    success=True,
                    audio_path=output_path,
                    generation_time=generation_time,
                    error=None
                )
            else:
                raise RuntimeError(f"Output file was not created: {output_path}")
                
        except Exception as e:
            generation_time = time.time() - start_time
            logger.error(f"❌ IndexTTS synthesis failed: {e}")
            
            return InferenceResult(
                success=False,
                audio_path=None,
                generation_time=generation_time,
                error=str(e)
            )

    def create_character_voice_profile(self, character: str, reference_audio: str) -> Dict[str, Any]:
        """Create a voice profile for a character using reference audio."""
        logger.info(f"🎭 Creating voice profile for {character}")
        logger.info(f"🎯 Using reference: {Path(reference_audio).name}")
        
        if not Path(reference_audio).exists():
            raise FileNotFoundError(f"Reference audio not found: {reference_audio}")
        
        # Test synthesis to validate the voice
        test_text = self._get_character_test_text(character)
        
        # Create temp output for test
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            test_output = tmp.name
        
        try:
            # Use asyncio to run synthesis
            import asyncio
            result = asyncio.run(self.synthesize(
                text=test_text,
                reference_audio=reference_audio,
                output_path=test_output
            ))
            
            if result.success:
                profile = {
                    'character': character,
                    'reference_audio': reference_audio,
                    'test_synthesis_time': result.generation_time,
                    'parameters': self.character_synthesis_params.get(character, {
                        'temperature': 0.7,
                        'speed': 1.0
                    }),
                    'test_text': test_text,
                    'model_type': 'indextts'
                }
                
                logger.info(f"✅ Voice profile created for {character}")
                logger.info(f"⏱️ Test synthesis: {result.generation_time:.2f}s")
                
                return profile
            else:
                raise RuntimeError(f"Test synthesis failed: {result.error}")
                
        finally:
            # Clean up temp file
            if Path(test_output).exists():
                Path(test_output).unlink()

    def _get_character_test_text(self, character: str) -> str:
        """Get character-specific test text for English synthesis."""
        character_texts = {
            'meowth': "That's right! Team Rocket's here to steal your Pokemon!",
            'ash': "I choose you, Pikachu! Let's win this battle!",
            'brock': "I'll cook up something delicious for everyone!",
            'misty': "Water Pokemon are the most beautiful of all!",
            'jessie': "Prepare for trouble, and make it double!",
            'james': "Team Rocket blasts off at the speed of light!",
            'narrator': "Meanwhile, our heroes continue their Pokemon journey.",
        }
        
        return character_texts.get(character, 
            "Hello, this is a test of my voice. How do I sound today?") 