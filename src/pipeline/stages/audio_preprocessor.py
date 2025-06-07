"""
Audio Preprocessor Pipeline Stage
Handles audio cleaning, normalization, and enhancement
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
from concurrent.futures import ThreadPoolExecutor
import yaml
from dataclasses import dataclass
import subprocess
import tempfile
import shutil

from ..validators.audio_quality import AudioQualityValidator

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result of processing a single audio file."""
    input_file: str
    output_file: str
    success: bool
    error: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None


class AudioPreprocessor:
    """Preprocess audio files for TTS training."""
    
    def __init__(self, config_path: str = "config/audio/preprocessing.yaml"):
        """Initialize preprocessor with configuration."""
        self.config = self._load_config(config_path)
        self.validator = AudioQualityValidator(config_path)
        self.supported_formats = ['.wav', '.mp3', '.flac', '.ogg', '.aac', '.m4a']
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load audio preprocessing configuration."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            # Return default config
            return {
                'input': {
                    'sample_rate': 48000,
                    'format': 'wav',
                    'channels': 'mono'
                },
                'preprocessing': {
                    'trim_silence': {'enabled': True, 'threshold_db': -40},
                    'denoise': {'enabled': True, 'strength': 0.5},
                    'normalize': {'enabled': True, 'method': 'lufs', 'target_lufs': -23},
                    'resample': {'target_rate': 24000, 'quality': 'high'}
                },
                'output': {
                    'sample_rate': 24000,
                    'format': 'wav',
                    'bit_depth': 16,
                    'channels': 1
                },
                'performance': {
                    'parallel_jobs': 4
                }
            }
    
    async def process_directory(self, input_dir: str, output_dir: str) -> Dict[str, Any]:
        """Process all audio files in directory."""
        logger.info(f"Preprocessing audio from {input_dir} to {output_dir}")
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Validate input directory
        if not input_path.exists():
            return {
                'success': False,
                'error': f"Input directory not found: {input_dir}",
                'files_processed': 0,
                'output_path': output_dir
            }
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all audio files
        audio_files = self._find_audio_files(input_path)
        
        if not audio_files:
            return {
                'success': True,
                'message': 'No audio files found to process',
                'files_processed': 0,
                'output_path': output_dir
            }
        
        logger.info(f"Found {len(audio_files)} audio files to process")
        
        # Process files in parallel
        processing_results = await self._process_files_parallel(audio_files, output_path)
        
        # Calculate results
        successful_files = [r for r in processing_results if r.success]
        failed_files = [r for r in processing_results if not r.success]
        
        # Validate processed audio quality
        if successful_files:
            validation_result = await self.validator.validate_directory(str(output_path))
            logger.info(f"Audio quality validation: {validation_result.overall_score:.1f}/10 "
                       f"({validation_result.passed_files} passed, {validation_result.failed_files} failed)")
        
        return {
            'success': True,
            'files_processed': len(successful_files),
            'files_failed': len(failed_files),
            'output_path': str(output_path),
            'processing_results': processing_results,
            'validation_score': validation_result.overall_score if successful_files else 0.0,
            'errors': [r.error for r in failed_files if r.error]
        }
    
    def _find_audio_files(self, directory: Path) -> List[Path]:
        """Find all supported audio files in directory."""
        audio_files = []
        for ext in self.supported_formats:
            audio_files.extend(directory.glob(f"*{ext}"))
            audio_files.extend(directory.glob(f"**/*{ext}"))
        return audio_files
    
    async def _process_files_parallel(self, audio_files: List[Path], output_dir: Path) -> List[ProcessingResult]:
        """Process audio files in parallel."""
        max_workers = self.config['performance']['parallel_jobs']
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            tasks = [
                asyncio.get_event_loop().run_in_executor(
                    executor, self._process_single_file, file_path, output_dir
                )
                for file_path in audio_files
            ]
            return await asyncio.gather(*tasks)
    
    def _process_single_file(self, input_file: Path, output_dir: Path) -> ProcessingResult:
        """Process a single audio file."""
        try:
            # Generate output filename
            output_file = output_dir / f"{input_file.stem}_processed.wav"
            
            logger.debug(f"Processing {input_file} -> {output_file}")
            
            # Load audio
            audio, sr = librosa.load(str(input_file), sr=None)
            original_duration = len(audio) / sr
            
            # Track processing metrics
            metrics = {
                'original_duration': original_duration,
                'original_sample_rate': sr,
                'original_channels': 1 if audio.ndim == 1 else audio.shape[0]
            }
            
            # Convert to mono if needed
            if audio.ndim > 1:
                audio = librosa.to_mono(audio)
                metrics['converted_to_mono'] = True
            
            # Step 1: Trim silence
            if self.config['preprocessing']['trim_silence']['enabled']:
                audio = self._trim_silence(audio, sr)
                metrics['silence_trimmed'] = True
            
            # Step 2: Denoise
            if self.config['preprocessing']['denoise']['enabled']:
                audio = self._denoise_audio(audio, sr)
                metrics['denoised'] = True
            
            # Step 3: Normalize volume
            if self.config['preprocessing']['normalize']['enabled']:
                audio = self._normalize_audio(audio, sr)
                metrics['normalized'] = True
            
            # Step 4: Resample to target rate
            target_sr = self.config['output']['sample_rate']
            if sr != target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr,
                                       res_type='kaiser_best')
                sr = target_sr
                metrics['resampled'] = True
                metrics['target_sample_rate'] = target_sr
            
            # Step 5: Apply final quality checks and adjustments
            audio = self._apply_final_processing(audio, sr)
            
            # Calculate final metrics
            metrics.update({
                'final_duration': len(audio) / sr,
                'final_sample_rate': sr,
                'duration_change': len(audio) / sr - original_duration
            })
            
            # Save processed audio
            sf.write(str(output_file), audio, sr, 
                    subtype=f"PCM_{self.config['output']['bit_depth']}")
            
            logger.debug(f"Successfully processed {input_file.name}")
            
            return ProcessingResult(
                input_file=str(input_file),
                output_file=str(output_file),
                success=True,
                metrics=metrics
            )
            
        except Exception as e:
            logger.error(f"Failed to process {input_file}: {e}")
            return ProcessingResult(
                input_file=str(input_file),
                output_file="",
                success=False,
                error=str(e)
            )
    
    def _trim_silence(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Trim silence from beginning and end of audio."""
        try:
            config = self.config['preprocessing']['trim_silence']
            threshold_db = config.get('threshold_db', -40)
            
            # Convert dB to amplitude threshold
            threshold = 10**(threshold_db / 20)
            
            # Use librosa to trim silence
            audio_trimmed, _ = librosa.effects.trim(
                audio, 
                top_db=-threshold_db,
                frame_length=2048,
                hop_length=512
            )
            
            return audio_trimmed
            
        except Exception as e:
            logger.warning(f"Failed to trim silence: {e}")
            return audio
    
    def _denoise_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply noise reduction to audio."""
        try:
            config = self.config['preprocessing']['denoise']
            
            if config.get('method') == 'rnnoise':
                # For RNNoise, we'd need rnnoise-python which has compatibility issues
                # Fall back to spectral gating method
                return self._spectral_denoise(audio, sr)
            else:
                return self._spectral_denoise(audio, sr)
                
        except Exception as e:
            logger.warning(f"Failed to denoise audio: {e}")
            return audio
    
    def _spectral_denoise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply spectral noise reduction."""
        try:
            # Use noisereduce library for spectral gating
            reduced_noise = nr.reduce_noise(y=audio, sr=sr, prop_decrease=0.8)
            return reduced_noise
        except Exception as e:
            logger.warning(f"Spectral denoising failed: {e}")
            return audio
    
    def _normalize_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Normalize audio volume."""
        try:
            config = self.config['preprocessing']['normalize']
            method = config.get('method', 'peak')
            
            if method == 'peak':
                # Peak normalization
                max_val = np.max(np.abs(audio))
                if max_val > 0:
                    target_peak = 10**(config.get('max_peak', -3.0) / 20)
                    audio = audio * (target_peak / max_val)
            
            elif method == 'rms':
                # RMS normalization
                rms = np.sqrt(np.mean(audio**2))
                if rms > 0:
                    target_rms = 0.1  # -20dB RMS
                    audio = audio * (target_rms / rms)
            
            elif method == 'lufs':
                # LUFS normalization (simplified implementation)
                # In production, you'd use pyloudnorm library
                rms = np.sqrt(np.mean(audio**2))
                if rms > 0:
                    # Approximate LUFS to RMS conversion
                    target_lufs = config.get('target_lufs', -23)
                    target_rms = 10**((target_lufs + 20) / 20)  # Rough conversion
                    audio = audio * (target_rms / rms)
            
            # Ensure no clipping
            audio = np.clip(audio, -1.0, 1.0)
            
            return audio
            
        except Exception as e:
            logger.warning(f"Failed to normalize audio: {e}")
            return audio
    
    def _apply_final_processing(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply final processing steps."""
        # Remove DC offset
        audio = audio - np.mean(audio)
        
        # Apply soft limiting to prevent any clipping
        audio = np.tanh(audio * 0.95) * 0.95
        
        # Ensure audio is in valid range
        audio = np.clip(audio, -1.0, 1.0)
        
        return audio
    
    async def validate_output(self, output_dir: str) -> Dict[str, Any]:
        """Validate the quality of processed audio files."""
        validation_result = await self.validator.validate_directory(output_dir)
        
        return {
            'overall_score': validation_result.overall_score,
            'passed_files': validation_result.passed_files,
            'failed_files': validation_result.failed_files,
            'issues': validation_result.issues[:10],  # Show first 10 issues
            'total_files': validation_result.passed_files + validation_result.failed_files
        } 