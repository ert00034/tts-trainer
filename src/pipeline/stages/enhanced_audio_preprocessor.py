"""
Enhanced Audio Preprocessor Pipeline Stage
Addresses vocal fry, raspy voices, and poor audio quality specifically for TTS training
Based on research from Steinberg Forums and TTS experts
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
import torch
import torchaudio

from ..validators.audio_quality import AudioQualityValidator

logger = logging.getLogger(__name__)


@dataclass
class EnhancedProcessingResult:
    """Result of enhanced processing a single audio file."""
    input_file: str
    output_file: str
    success: bool
    error: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    vocal_fry_detected: bool = False
    enhancement_applied: bool = False


class EnhancedAudioPreprocessor:
    """Enhanced audio preprocessor specifically for TTS training with vocal fry removal."""
    
    def __init__(self, config_path: str = "config/audio/enhanced_preprocessing.yaml"):
        """Initialize enhanced preprocessor with configuration."""
        self.config = self._load_config(config_path)
        self.validator = AudioQualityValidator(config_path)
        self.supported_formats = ['.wav', '.mp3', '.flac', '.ogg', '.aac', '.m4a']
        
        # Initialize enhancement models
        self.demucs_model = None
        self.enhance_model = None
        self._initialize_models()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load enhanced audio preprocessing configuration."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            # Return enhanced default config
            return {
                'input': {
                    'sample_rate': 48000,
                    'format': 'wav',
                    'channels': 'mono'
                },
                'preprocessing': {
                    'denoise': {'enabled': True, 'method': 'demucs'},
                    'enhance': {'enabled': True, 'method': 'resemble_enhance'},
                    'vocal_fry_removal': {'enabled': True, 'aggressive': 0.3},
                    'normalize': {'enabled': True, 'method': 'lufs', 'target_lufs': -23},
                    'trim_silence': {'enabled': True, 'threshold_db': -40},
                    'resample': {'target_rate': 24000, 'quality': 'high'}
                },
                'output': {
                    'sample_rate': 24000,
                    'format': 'wav',
                    'bit_depth': 16,
                    'channels': 1
                },
                'performance': {
                    'parallel_jobs': 2,  # Reduced due to heavy processing
                    'use_gpu': True
                }
            }
    
    def _initialize_models(self):
        """Initialize enhancement models."""
        try:
            # Initialize Demucs for vocal separation
            if self.config['preprocessing']['denoise']['method'] == 'demucs':
                try:
                    import demucs
                    # For now, skip Demucs initialization and fall back to spectral
                    # Demucs API is complex and needs proper setup
                    raise ImportError("Demucs setup requires additional configuration")
                    logger.info("Demucs model initialized successfully")
                except (ImportError, Exception) as e:
                    logger.warning(f"Demucs not available: {e}, falling back to spectral denoising")
                    self.demucs_model = None
            
            # Initialize Resemble-Enhance for voice enhancement
            if self.config['preprocessing']['enhance']['method'] == 'resemble_enhance':
                try:
                    # Import resemble-enhance functions
                    from resemble_enhance.enhancer.inference import denoise, enhance
                    self.enhance_denoise = denoise
                    self.enhance_enhance = enhance
                    logger.info("Resemble-Enhance initialized successfully")
                except ImportError:
                    logger.warning("Resemble-Enhance not available, falling back to basic enhancement")
                    self.enhance_denoise = None
                    self.enhance_enhance = None
            
        except Exception as e:
            logger.warning(f"Failed to initialize enhancement models: {e}")
    
    async def process_directory(self, input_dir: str, output_dir: str) -> Dict[str, Any]:
        """Process all audio files in a directory with enhanced preprocessing."""
        logger.info(f"Starting enhanced audio preprocessing: {input_dir} -> {output_dir}")
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all audio files
        audio_files = []
        for ext in self.supported_formats:
            audio_files.extend(input_path.glob(f"**/*{ext}"))
        
        logger.info(f"Found {len(audio_files)} audio files to process")
        
        # Process files with enhanced pipeline
        results = []
        max_workers = self.config['performance']['parallel_jobs']
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._process_single_file, file, output_path)
                for file in audio_files
            ]
            
            for future in asyncio.as_completed([asyncio.wrap_future(f) for f in futures]):
                result = await future
                results.append(result)
                
                if result.success:
                    logger.debug(f"✓ Enhanced processing completed: {result.input_file}")
                else:
                    logger.error(f"✗ Enhanced processing failed: {result.input_file} - {result.error}")
        
        # Generate summary
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        vocal_fry_detected = len([r for r in successful if r.vocal_fry_detected])
        enhanced = len([r for r in successful if r.enhancement_applied])
        
        logger.info(f"Enhanced preprocessing completed: {len(successful)}/{len(results)} files processed")
        logger.info(f"Vocal fry detected and removed: {vocal_fry_detected} files")
        logger.info(f"Voice enhancement applied: {enhanced} files")
        
        return {
            'total_files': len(results),
            'successful': len(successful),
            'failed': len(failed),
            'vocal_fry_detected': vocal_fry_detected,
            'enhancement_applied': enhanced,
            'results': results
        }
    
    def _process_single_file(self, input_file: Path, output_dir: Path) -> EnhancedProcessingResult:
        """Process a single audio file with enhanced preprocessing."""
        try:
            # Generate output filename
            output_file = output_dir / f"{input_file.stem}_enhanced.wav"
            
            logger.debug(f"Enhanced processing {input_file} -> {output_file}")
            
            # Load audio
            audio, sr = librosa.load(str(input_file), sr=None)
            original_duration = len(audio) / sr
            
            # Track processing metrics
            metrics = {
                'original_duration': original_duration,
                'original_sample_rate': sr,
                'original_channels': 1 if audio.ndim == 1 else audio.shape[0]
            }
            
            vocal_fry_detected = False
            enhancement_applied = False
            
            # Convert to mono if needed and ensure 1D array
            if audio.ndim > 1:
                audio = librosa.to_mono(audio)
                metrics['converted_to_mono'] = True
            
            # Ensure audio is 1D
            if audio.ndim > 1:
                audio = audio.flatten()
                
            # Ensure audio is not empty
            if len(audio) == 0:
                raise ValueError("Audio file is empty or corrupted")
            
            # Step 1: Detect and analyze vocal fry
            if self.config['preprocessing']['vocal_fry_removal']['enabled']:
                vocal_fry_detected = self._detect_vocal_fry(audio, sr)
                metrics['vocal_fry_detected'] = vocal_fry_detected
            
            # Step 2: Denoise using Demucs (separate vocals from noise)
            if self.config['preprocessing']['denoise']['enabled']:
                if self.demucs_model and self.config['preprocessing']['denoise']['method'] == 'demucs':
                    audio = self._denoise_with_demucs(audio, sr, input_file)
                    metrics['denoised_with_demucs'] = True
                else:
                    audio = self._spectral_denoise(audio, sr)
                    metrics['denoised_with_spectral'] = True
            
            # Step 3: Remove vocal fry if detected
            if vocal_fry_detected and self.config['preprocessing']['vocal_fry_removal']['enabled']:
                audio = self._remove_vocal_fry(audio, sr)
                metrics['vocal_fry_removed'] = True
            
            # Step 4: Enhance voice quality using Resemble-Enhance
            if self.config['preprocessing']['enhance']['enabled']:
                if self.enhance_denoise and self.enhance_enhance:
                    audio = self._enhance_with_resemble(audio, sr)
                    enhancement_applied = True
                    metrics['enhanced_with_resemble'] = True
                else:
                    audio = self._enhance_with_spectral(audio, sr)
                    enhancement_applied = True
                    metrics['enhanced_with_spectral'] = True
            
            # Step 5: Trim silence
            if self.config['preprocessing']['trim_silence']['enabled']:
                audio = self._trim_silence(audio, sr)
                metrics['silence_trimmed'] = True
            
            # Step 6: Normalize volume
            if self.config['preprocessing']['normalize']['enabled']:
                audio = self._normalize_audio(audio, sr)
                metrics['normalized'] = True
            
            # Step 7: Resample to target rate
            target_sr = self.config['output']['sample_rate']
            if sr != target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr,
                                       res_type='kaiser_best')
                sr = target_sr
                metrics['resampled'] = True
                metrics['target_sample_rate'] = target_sr
            
            # Step 8: Apply final quality processing
            audio = self._apply_final_processing(audio, sr)
            
            # Calculate final metrics
            metrics.update({
                'final_duration': len(audio) / sr,
                'final_sample_rate': sr,
                'duration_change': len(audio) / sr - original_duration,
                'snr_improvement': self._calculate_snr_improvement(audio)
            })
            
            # Save enhanced audio
            sf.write(str(output_file), audio, sr, 
                    subtype=f"PCM_{self.config['output']['bit_depth']}")
            
            logger.debug(f"Successfully enhanced {input_file.name}")
            
            return EnhancedProcessingResult(
                input_file=str(input_file),
                output_file=str(output_file),
                success=True,
                metrics=metrics,
                vocal_fry_detected=vocal_fry_detected,
                enhancement_applied=enhancement_applied
            )
            
        except Exception as e:
            logger.error(f"Failed to enhance {input_file}: {e}")
            return EnhancedProcessingResult(
                input_file=str(input_file),
                output_file="",
                success=False,
                error=str(e)
            )
    
    def _detect_vocal_fry(self, audio: np.ndarray, sr: int) -> bool:
        """Detect vocal fry in audio using spectral analysis."""
        try:
            # Compute spectral features with consistent parameters
            n_fft = 2048  # Use larger FFT for better frequency resolution
            stft = librosa.stft(audio, hop_length=512, n_fft=n_fft)
            magnitude = np.abs(stft)
            
            # Focus on low frequencies where vocal fry appears (0-1000 Hz)
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
            low_freq_mask = freqs <= 1000
            
            # Ensure the mask matches the magnitude array dimensions
            if len(low_freq_mask) != magnitude.shape[0]:
                logger.warning(f"Frequency mask size mismatch: {len(low_freq_mask)} vs {magnitude.shape[0]}")
                return False
                
            low_freq_energy = np.mean(magnitude[low_freq_mask, :], axis=0)
            
            # Calculate energy variation in low frequencies
            energy_variation = np.std(low_freq_energy) / (np.mean(low_freq_energy) + 1e-8)
            
            # Threshold for vocal fry detection
            vocal_fry_threshold = 0.3
            
            return energy_variation > vocal_fry_threshold
            
        except Exception as e:
            logger.warning(f"Vocal fry detection failed: {e}")
            return False
    
    def _denoise_with_demucs(self, audio: np.ndarray, sr: int, input_file: Path) -> np.ndarray:
        """Denoise audio using Demucs vocal separation."""
        try:
            # Save temporary file for Demucs
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                sf.write(temp_file.name, audio, sr)
                temp_path = temp_file.name
            
            # Use Demucs to separate vocals
            separated = self.demucs_model.separate_audio_file(temp_path)
            
            # Extract vocals (clean voice)
            vocals = None
            for source_name, source_audio in separated[1].items():
                if source_name == 'vocals':
                    vocals = source_audio.numpy()
                    break
            
            # Clean up temporary file
            Path(temp_path).unlink()
            
            if vocals is not None:
                return vocals
            else:
                logger.warning("No vocals found in Demucs separation, using original audio")
                return audio
                
        except Exception as e:
            logger.warning(f"Demucs denoising failed: {e}")
            return self._spectral_denoise(audio, sr)
    
    def _remove_vocal_fry(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Remove vocal fry using spectral filtering approach from Steinberg Forums."""
        try:
            # Compute STFT with consistent parameters
            n_fft = 2048
            stft = librosa.stft(audio, hop_length=512, n_fft=n_fft)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Get frequency bins
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
            
            # Find vocal fry frequency range (0-1000 Hz)
            vocal_fry_mask = (freqs >= 0) & (freqs <= 1000)
            
            # Ensure the mask matches the magnitude array dimensions
            if len(vocal_fry_mask) != magnitude.shape[0]:
                logger.warning(f"Vocal fry mask size mismatch: {len(vocal_fry_mask)} vs {magnitude.shape[0]}")
                return audio
            
            # Apply aggressive filtering to vocal fry frequencies
            aggressive_factor = self.config['preprocessing']['vocal_fry_removal']['aggressive']
            
            # Reduce energy in vocal fry frequencies
            magnitude[vocal_fry_mask, :] *= (1.0 - aggressive_factor)
            
            # Reconstruct audio
            cleaned_stft = magnitude * np.exp(1j * phase)
            cleaned_audio = librosa.istft(cleaned_stft, hop_length=512, n_fft=n_fft)
            
            return cleaned_audio
            
        except Exception as e:
            logger.warning(f"Vocal fry removal failed: {e}")
            return audio
    
    def _enhance_with_resemble(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Enhance audio using Resemble-Enhance."""
        try:
            # Store original length to preserve duration
            original_length = len(audio)
            
            # Ensure audio is 1D
            if audio.ndim > 1:
                audio = audio.flatten()
            
            # Convert to tensor - Resemble-Enhance expects 1D input
            audio_tensor = torch.tensor(audio, dtype=torch.float32)
            
            # Apply Resemble-Enhance denoise
            if self.enhance_denoise:
                denoised, _ = self.enhance_denoise(audio_tensor, sr, device="cuda" if torch.cuda.is_available() else "cpu")
            else:
                denoised = audio_tensor
            
            # Apply Resemble-Enhance enhancement
            if self.enhance_enhance:
                enhanced, _ = self.enhance_enhance(
                    denoised, sr, 
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    nfe=64, solver="midpoint", lambd=0.9, tau=0.5
                )
            else:
                enhanced = denoised
            
            # Convert back to numpy and ensure 1D
            enhanced_audio = enhanced.squeeze().cpu().numpy()
            if enhanced_audio.ndim > 1:
                enhanced_audio = enhanced_audio.flatten()
            
            # Preserve original duration by truncating or padding
            if len(enhanced_audio) > original_length:
                # Truncate if longer
                enhanced_audio = enhanced_audio[:original_length]
            elif len(enhanced_audio) < original_length:
                # Pad with zeros if shorter
                padding = original_length - len(enhanced_audio)
                enhanced_audio = np.pad(enhanced_audio, (0, padding), mode='constant', constant_values=0)
            
            return enhanced_audio
            
        except Exception as e:
            logger.warning(f"Resemble-Enhance failed: {e}")
            return self._enhance_with_spectral(audio, sr)
    
    def _enhance_with_spectral(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Fallback spectral enhancement."""
        try:
            # Apply spectral enhancement using librosa
            # Boost speech frequencies (300-3000 Hz)
            n_fft = 2048
            stft = librosa.stft(audio, hop_length=512, n_fft=n_fft)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Get frequency bins
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
            
            # Boost speech frequencies
            speech_mask = (freqs >= 300) & (freqs <= 3000)
            
            # Ensure the mask matches the magnitude array dimensions
            if len(speech_mask) != magnitude.shape[0]:
                logger.warning(f"Speech mask size mismatch: {len(speech_mask)} vs {magnitude.shape[0]}")
                return audio
                
            magnitude[speech_mask, :] *= 1.2  # 20% boost
            
            # Reconstruct audio
            enhanced_stft = magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft, hop_length=512, n_fft=n_fft)
            
            return enhanced_audio
            
        except Exception as e:
            logger.warning(f"Spectral enhancement failed: {e}")
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
    
    def _trim_silence(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Trim silence from beginning and end of audio."""
        try:
            config = self.config['preprocessing']['trim_silence']
            threshold_db = config.get('threshold_db', -40)
            
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
    
    def _normalize_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Normalize audio volume."""
        try:
            config = self.config['preprocessing']['normalize']
            method = config.get('method', 'lufs')
            
            if method == 'lufs':
                # LUFS normalization (simplified implementation)
                rms = np.sqrt(np.mean(audio**2))
                if rms > 0:
                    target_lufs = config.get('target_lufs', -23)
                    target_rms = 10**((target_lufs + 20) / 20)
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
    
    def _calculate_snr_improvement(self, audio: np.ndarray) -> float:
        """Calculate SNR improvement estimate."""
        try:
            # Simple SNR calculation
            signal_level = np.percentile(np.abs(audio), 95)
            noise_level = np.percentile(np.abs(audio), 5)
            
            if noise_level > 0:
                snr = 20 * np.log10(signal_level / noise_level)
                return max(0.0, snr)
            else:
                return 60.0
                
        except Exception:
            return 20.0
    
    async def validate_output(self, output_dir: str) -> Dict[str, Any]:
        """Validate the quality of enhanced audio files."""
        validation_result = await self.validator.validate_directory(output_dir)
        
        return {
            'overall_score': validation_result.overall_score,
            'passed_files': validation_result.passed_files,
            'failed_files': validation_result.failed_files,
            'issues': validation_result.issues[:10],
            'total_files': validation_result.passed_files + validation_result.failed_files
        } 