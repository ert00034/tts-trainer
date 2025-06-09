"""
Audio Quality Validator
Validates audio quality for TTS training
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import numpy as np
import librosa
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor
import yaml

logger = logging.getLogger(__name__)


@dataclass
class AudioFileResult:
    """Result of individual audio file validation."""
    file_path: str
    passed: bool
    issues: List[str]
    metrics: Dict[str, float]


@dataclass
class ValidationResult:
    """Result of audio quality validation."""
    overall_score: float
    passed_files: int
    failed_files: int
    issues: List[str]
    file_results: List[AudioFileResult]


class AudioQualityValidator:
    """Validate audio quality for TTS training."""
    
    def __init__(self, config_path: str = "config/audio/preprocessing.yaml"):
        """Initialize validator with configuration."""
        self.config = self._load_config(config_path)
        self.min_quality_score = 7.0
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
            config = {
                'preprocessing': {
                    'quality_check': {
                        'min_duration': 1.0,
                        'max_duration': 30.0,
                        'snr_threshold': 15
                    }
                },
                'output': {
                    'sample_rate': 24000
                }
            }
            return config
    
    async def validate_directory(self, audio_dir: str) -> ValidationResult:
        """Validate all audio files in directory."""
        logger.info(f"Validating audio quality in {audio_dir}")
        
        audio_path = Path(audio_dir)
        if not audio_path.exists():
            return ValidationResult(
                overall_score=0.0,
                passed_files=0,
                failed_files=0,
                issues=[f"Directory not found: {audio_dir}"],
                file_results=[]
            )
        
        # Find all audio files
        audio_files = []
        for ext in self.supported_formats:
            audio_files.extend(audio_path.glob(f"*{ext}"))
            audio_files.extend(audio_path.glob(f"**/*{ext}"))
        
        if not audio_files:
            return ValidationResult(
                overall_score=0.0,
                passed_files=0,
                failed_files=0,
                issues=["No audio files found in directory"],
                file_results=[]
            )
        
        # Validate files in parallel
        file_results = await self._validate_files_parallel(audio_files)
        
        # Calculate overall results
        passed_files = sum(1 for result in file_results if result.passed)
        failed_files = len(file_results) - passed_files
        
        # Calculate overall score (weighted by file count)
        if file_results:
            scores = [self._calculate_file_score(result.metrics) for result in file_results]
            overall_score = np.mean(scores)
        else:
            overall_score = 0.0
        
        # Collect all issues
        all_issues = []
        for result in file_results:
            if result.issues:
                for issue in result.issues:
                    all_issues.append(f"{result.file_path}: {issue}")
        
        return ValidationResult(
            overall_score=overall_score,
            passed_files=passed_files,
            failed_files=failed_files,
            issues=all_issues,
            file_results=file_results
        )
    
    async def _validate_files_parallel(self, audio_files: List[Path]) -> List[AudioFileResult]:
        """Validate audio files in parallel."""
        with ThreadPoolExecutor(max_workers=4) as executor:
            tasks = [
                asyncio.get_event_loop().run_in_executor(
                    executor, self._validate_single_file, file_path
                )
                for file_path in audio_files
            ]
            return await asyncio.gather(*tasks)
    
    def _validate_single_file(self, file_path: Path) -> AudioFileResult:
        """Validate a single audio file."""
        issues = []
        metrics = {}
        
        try:
            # Load audio file
            audio, sr = librosa.load(str(file_path), sr=None)
            duration = librosa.get_duration(y=audio, sr=sr)
            
            # Basic file metrics
            metrics['duration'] = duration
            metrics['sample_rate'] = sr
            metrics['channels'] = 1 if audio.ndim == 1 else audio.shape[0]
            metrics['file_size'] = file_path.stat().st_size
            
            # Ensure quality_check config exists
            quality_config = self.config.get('preprocessing', {}).get('quality_check', {
                'min_duration': 1.0,
                'max_duration': 30.0,
                'snr_threshold': 15
            })
            
            # Validate duration
            min_duration = quality_config.get('min_duration', 1.0)
            max_duration = quality_config.get('max_duration', 30.0)
            
            if duration < min_duration:
                issues.append(f"Too short: {duration:.2f}s (min: {min_duration}s)")
            elif duration > max_duration:
                issues.append(f"Too long: {duration:.2f}s (max: {max_duration}s)")
            
            # Validate sample rate
            expected_sr = self.config['output']['sample_rate']
            if sr < 16000:
                issues.append(f"Sample rate too low: {sr}Hz (min: 16000Hz)")
            
            # Calculate SNR (Signal-to-Noise Ratio)
            snr = self._calculate_snr(audio)
            metrics['snr'] = snr
            
            snr_threshold = quality_config.get('snr_threshold', 15)
            if snr < snr_threshold:
                issues.append(f"Low SNR: {snr:.1f}dB (min: {snr_threshold}dB)")
            
            # Detect clipping
            clipping_ratio = self._detect_clipping(audio)
            metrics['clipping_ratio'] = clipping_ratio
            
            if clipping_ratio > 0.01:  # More than 1% clipping
                issues.append(f"Clipping detected: {clipping_ratio:.2%} of samples")
            
            # Detect silence ratio
            silence_ratio = self._calculate_silence_ratio(audio)
            metrics['silence_ratio'] = silence_ratio
            
            if silence_ratio > 0.5:  # More than 50% silence
                issues.append(f"Too much silence: {silence_ratio:.1%}")
            
            # Calculate dynamic range
            dynamic_range = self._calculate_dynamic_range(audio)
            metrics['dynamic_range'] = dynamic_range
            
            if dynamic_range < 10:  # Less than 10dB dynamic range
                issues.append(f"Low dynamic range: {dynamic_range:.1f}dB")
            
            # Check for DC offset
            dc_offset = np.mean(audio)
            metrics['dc_offset'] = float(dc_offset)
            
            if abs(dc_offset) > 0.01:
                issues.append(f"DC offset detected: {dc_offset:.3f}")
            
        except Exception as e:
            issues.append(f"Failed to analyze: {str(e)}")
            metrics['error'] = str(e)
        
        # Determine if file passed validation
        passed = len(issues) == 0
        
        return AudioFileResult(
            file_path=str(file_path),
            passed=passed,
            issues=issues,
            metrics=metrics
        )
    
    def _calculate_snr(self, audio: np.ndarray) -> float:
        """Calculate Signal-to-Noise Ratio."""
        try:
            # Use a more robust SNR calculation for enhanced audio
            # Calculate percentile-based signal and noise levels
            
            # Signal level: 90th percentile of absolute values
            signal_level = np.percentile(np.abs(audio), 90)
            
            # Noise level: 10th percentile of absolute values  
            noise_level = np.percentile(np.abs(audio), 10)
            
            # Ensure we don't divide by zero
            if noise_level <= 0:
                noise_level = np.finfo(float).eps
            
            # Calculate SNR in dB
            snr = 20 * np.log10(signal_level / noise_level)
            
            # Clamp to reasonable range for enhanced audio
            snr = max(0.0, min(60.0, snr))
                
            return float(snr)
            
        except Exception:
            return 20.0  # Default reasonable SNR
    
    def _detect_clipping(self, audio: np.ndarray) -> float:
        """Detect clipping in audio signal."""
        # Detect samples at or near the maximum amplitude
        max_val = np.max(np.abs(audio))
        threshold = 0.99 * max_val if max_val > 0.9 else 0.99
        
        clipped_samples = np.sum(np.abs(audio) >= threshold)
        total_samples = len(audio)
        
        return clipped_samples / total_samples if total_samples > 0 else 0.0
    
    def _calculate_silence_ratio(self, audio: np.ndarray) -> float:
        """Calculate ratio of silence in audio."""
        # Use a threshold based on the audio's own dynamic range
        audio_rms = np.sqrt(np.mean(audio**2))
        silence_threshold = audio_rms * 0.01  # 1% of RMS as silence threshold
        
        # Find samples below threshold
        silent_samples = np.sum(np.abs(audio) < silence_threshold)
        total_samples = len(audio)
        
        return silent_samples / total_samples if total_samples > 0 else 0.0
    
    def _calculate_dynamic_range(self, audio: np.ndarray) -> float:
        """Calculate dynamic range of audio in dB."""
        if len(audio) == 0:
            return 0.0
            
        # Calculate 95th percentile (loud) and 5th percentile (quiet) levels
        loud_level = np.percentile(np.abs(audio), 95)
        quiet_level = np.percentile(np.abs(audio), 5)
        
        if quiet_level > 0:
            dynamic_range = 20 * np.log10(loud_level / quiet_level)
        else:
            dynamic_range = 60.0  # Assume high dynamic range
            
        return float(dynamic_range)
    
    def _calculate_file_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall quality score for a file (0-10)."""
        score = 10.0
        
        # Penalize low SNR
        snr = metrics.get('snr', 20)
        if snr < 20:
            score -= (20 - snr) * 0.2
        
        # Penalize clipping
        clipping = metrics.get('clipping_ratio', 0)
        score -= clipping * 50  # Heavy penalty for clipping
        
        # Penalize excessive silence
        silence = metrics.get('silence_ratio', 0)
        if silence > 0.3:
            score -= (silence - 0.3) * 10
        
        # Penalize low dynamic range
        dynamic_range = metrics.get('dynamic_range', 40)
        if dynamic_range < 20:
            score -= (20 - dynamic_range) * 0.1
        
        return max(0.0, min(10.0, score)) 