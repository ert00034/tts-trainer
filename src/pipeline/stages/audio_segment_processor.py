"""
Audio Segment Processor Pipeline Stage
Improves quality of speaker segments by refining boundaries and filtering poor quality clips
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import numpy as np
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import librosa
import soundfile as sf

logger = logging.getLogger(__name__)


def convert_numpy_types(obj):
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    else:
        return obj


@dataclass
class SegmentQuality:
    """Quality metrics for an audio segment."""
    duration: float
    snr_db: float
    speech_ratio: float  # Ratio of speech vs silence
    word_count: int
    complete_words: bool  # Whether segment ends with complete words
    energy_stability: float  # Consistency of audio energy
    quality_score: float  # Overall quality score 0-1


@dataclass
class ProcessedSegment:
    """A processed audio segment with improved boundaries."""
    original_file: str
    speaker: str
    start: float
    end: float
    extended_start: float
    extended_end: float
    text: str
    confidence: float
    quality: SegmentQuality
    segment_file: str
    accepted: bool
    rejection_reason: Optional[str] = None


class AudioSegmentProcessor:
    """Improves quality of speaker segments for TTS training."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with quality processing configuration."""
        self.config = config or self._get_default_config()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for segment processing."""
        return {
            'quality_thresholds': {
                'min_duration': 2.0,          # Minimum segment duration in seconds
                'max_duration': 15.0,         # Maximum segment duration
                'min_snr_db': 12.0,          # Minimum signal-to-noise ratio
                'min_speech_ratio': 0.5,      # Minimum ratio of speech vs silence
                'min_word_count': 3,          # Minimum number of words
                'min_quality_score': 0.5,     # Overall quality threshold
            },
            'boundary_extension': {
                'padding_seconds': 0.2,       # Seconds to extend on each side
                'max_extension': 1.0,         # Maximum extension in seconds
            },
            'audio_processing': {
                'target_sample_rate': 24000,
                'normalize_volume': True,
                'fade_in_ms': 50,            # Fade in/out to avoid clicks
                'fade_out_ms': 50,
            }
        }
    
    async def process_segments(self, segments_dir: str, output_dir: str) -> Dict[str, Any]:
        """
        Process existing speaker segments to improve quality for TTS training.
        
        Args:
            segments_dir: Directory containing speaker segment folders
            output_dir: Directory to save improved segments
        """
        logger.info(f"Processing speaker segments: {segments_dir} → {output_dir}")
        
        segments_path = Path(segments_dir)
        output_path = Path(output_dir)
        
        if not segments_path.exists():
            return {
                'success': False,
                'error': f"Segments directory not found: {segments_dir}",
                'speakers_processed': 0
            }
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all speaker directories
        speaker_dirs = [d for d in segments_path.iterdir() if d.is_dir()]
        
        if not speaker_dirs:
            return {
                'success': True,
                'message': 'No speaker directories found',
                'speakers_processed': 0,
                'output_path': str(output_path)
            }
        
        logger.info(f"Found {len(speaker_dirs)} speaker directories to process")
        
        # Process speakers in parallel
        max_workers = 3  # Conservative to avoid memory issues
        processing_results = await self._process_speakers_parallel(
            speaker_dirs, output_path, max_workers
        )
        
        # Calculate overall results
        successful_speakers = [r for r in processing_results if r['success']]
        failed_speakers = [r for r in processing_results if not r['success']]
        
        # Aggregate statistics
        total_segments_input = sum(r.get('segments_input', 0) for r in successful_speakers)
        total_segments_output = sum(r.get('segments_accepted', 0) for r in successful_speakers)
        total_duration_output = sum(r.get('total_duration', 0.0) for r in successful_speakers)
        
        # Generate quality report
        quality_report = self._generate_quality_report(processing_results, output_path)
        
        return {
            'success': True,
            'speakers_processed': len(successful_speakers),
            'speakers_failed': len(failed_speakers),
            'segments_input': total_segments_input,
            'segments_accepted': total_segments_output,
            'segments_rejected': total_segments_input - total_segments_output,
            'rejection_rate': (total_segments_input - total_segments_output) / total_segments_input if total_segments_input > 0 else 0,
            'total_duration_hours': total_duration_output / 3600,
            'output_path': str(output_path),
            'quality_report': quality_report,
            'errors': [r.get('error') for r in failed_speakers if r.get('error')]
        }
    
    async def _process_speakers_parallel(self, speaker_dirs: List[Path], 
                                       output_path: Path, max_workers: int) -> List[Dict[str, Any]]:
        """Process speaker directories in parallel."""
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            tasks = [
                asyncio.get_event_loop().run_in_executor(
                    executor, self._process_single_speaker, speaker_dir, output_path
                )
                for speaker_dir in speaker_dirs
            ]
            return await asyncio.gather(*tasks)
    
    def _process_single_speaker(self, speaker_dir: Path, output_path: Path) -> Dict[str, Any]:
        """Process all segments for a single speaker."""
        try:
            speaker_id = speaker_dir.name
            logger.info(f"Processing speaker: {speaker_id}")
            
            # Create output directory for this speaker
            speaker_output_dir = output_path / speaker_id
            speaker_output_dir.mkdir(exist_ok=True)
            
            # Find all audio segments for this speaker
            audio_files = []
            for ext in ['.wav', '.mp3', '.flac']:
                audio_files.extend(speaker_dir.glob(f"*{ext}"))
            
            if not audio_files:
                return {
                    'success': True,
                    'speaker': speaker_id,
                    'segments_input': 0,
                    'segments_accepted': 0,
                    'message': 'No audio files found'
                }
            
            processed_segments = []
            
            # Process each segment
            for audio_file in audio_files:
                # Load metadata
                metadata_file = audio_file.with_suffix(audio_file.suffix + '.json')
                if not metadata_file.exists():
                    logger.warning(f"No metadata found for {audio_file.name}")
                    continue
                
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # Process this segment
                processed_segment = self._process_single_segment(
                    audio_file, metadata, speaker_output_dir
                )
                
                if processed_segment:
                    processed_segments.append(processed_segment)
            
            # Calculate statistics
            accepted_segments = [s for s in processed_segments if s.accepted]
            rejected_segments = [s for s in processed_segments if not s.accepted]
            
            total_duration = sum(s.quality.duration for s in accepted_segments)
            
            # Save processing report
            self._save_speaker_report(processed_segments, speaker_output_dir)
            
            logger.info(f"Speaker {speaker_id}: {len(accepted_segments)}/{len(processed_segments)} segments accepted "
                       f"({total_duration:.1f}s total)")
            
            return {
                'success': True,
                'speaker': speaker_id,
                'segments_input': len(processed_segments),
                'segments_accepted': len(accepted_segments),
                'segments_rejected': len(rejected_segments),
                'total_duration': total_duration,
                'rejection_reasons': [s.rejection_reason for s in rejected_segments if s.rejection_reason],
                'avg_quality_score': np.mean([s.quality.quality_score for s in accepted_segments]) if accepted_segments else 0.0
            }
            
        except Exception as e:
            logger.error(f"Failed to process speaker {speaker_dir.name}: {e}")
            return {
                'success': False,
                'speaker': speaker_dir.name,
                'error': str(e)
            }
    
    def _process_single_segment(self, audio_file: Path, metadata: Dict[str, Any], 
                              output_dir: Path) -> Optional[ProcessedSegment]:
        """Process a single audio segment to improve quality."""
        try:
            # Load original audio file
            original_audio_path = Path(metadata['original_file'])
            if not original_audio_path.exists():
                logger.warning(f"Original audio file not found: {original_audio_path}")
                return None
            
            # Load full episode audio
            full_audio, sr = librosa.load(str(original_audio_path), sr=None)
            
            # Get original segment boundaries
            start_time = metadata['start']
            end_time = metadata['end']
            text = metadata['text']
            speaker = metadata['speaker']
            confidence = metadata.get('confidence', 1.0)
            
            # Extend boundaries with padding
            padding = self.config['boundary_extension']['padding_seconds']
            max_extension = self.config['boundary_extension']['max_extension']
            
            extended_start = max(0, start_time - min(padding, max_extension))
            extended_end = min(len(full_audio) / sr, end_time + min(padding, max_extension))
            
            # Extract extended segment
            start_sample = int(extended_start * sr)
            end_sample = int(extended_end * sr)
            
            if start_sample >= len(full_audio) or end_sample > len(full_audio):
                logger.warning(f"Extended segment boundaries exceed audio length")
                return None
            
            segment_audio = full_audio[start_sample:end_sample]
            
            # Analyze segment quality
            quality = self._analyze_segment_quality(segment_audio, sr, text)
            
            # Determine if segment should be accepted
            accepted, rejection_reason = self._should_accept_segment(quality)
            
            # Generate output filename
            base_name = audio_file.stem.replace('.wav', '')  # Remove .wav if present
            output_filename = f"{base_name}_processed.wav"
            output_path = output_dir / output_filename
            
            # Create processed segment object
            processed_segment = ProcessedSegment(
                original_file=str(original_audio_path),
                speaker=speaker,
                start=start_time,
                end=end_time,
                extended_start=extended_start,
                extended_end=extended_end,
                text=text,
                confidence=confidence,
                quality=quality,
                segment_file=str(output_path),
                accepted=accepted,
                rejection_reason=rejection_reason
            )
            
            # Save accepted segments
            if accepted:
                # Apply audio processing
                processed_audio = self._apply_audio_processing(segment_audio, sr)
                
                # Save audio file
                sf.write(str(output_path), processed_audio, sr)
                
                # Save metadata
                metadata_path = output_path.with_suffix('.json')
                self._save_segment_metadata(processed_segment, metadata_path)
            
            return processed_segment
            
        except Exception as e:
            logger.error(f"Failed to process segment {audio_file.name}: {e}")
            return None
    
    def _analyze_segment_quality(self, audio: np.ndarray, sr: int, text: str) -> SegmentQuality:
        """Analyze the quality of an audio segment."""
        duration = len(audio) / sr
        
        # Calculate SNR (signal-to-noise ratio)
        snr_db = self._calculate_snr(audio)
        
        # Calculate speech ratio using energy
        speech_ratio = self._calculate_speech_ratio(audio, sr)
        
        # Analyze text quality
        word_count = len(text.split())
        complete_words = self._check_complete_words(text)
        
        # Calculate energy stability
        energy_stability = self._calculate_energy_stability(audio, sr)
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(
            duration, snr_db, speech_ratio, word_count, complete_words, energy_stability
        )
        
        return SegmentQuality(
            duration=duration,
            snr_db=snr_db,
            speech_ratio=speech_ratio,
            word_count=word_count,
            complete_words=complete_words,
            energy_stability=energy_stability,
            quality_score=quality_score
        )
    
    def _calculate_snr(self, audio: np.ndarray) -> float:
        """Calculate signal-to-noise ratio estimate."""
        if len(audio) == 0:
            return 0.0
        
        # Use 95th percentile as signal, 5th percentile as noise
        signal_level = np.percentile(np.abs(audio), 95)
        noise_level = np.percentile(np.abs(audio), 5)
        
        if noise_level == 0:
            return 60.0  # Very high SNR
        
        snr = 20 * np.log10(signal_level / noise_level)
        return max(0.0, snr)
    
    def _calculate_speech_ratio(self, audio: np.ndarray, sr: int) -> float:
        """Calculate ratio of speech vs silence in segment."""
        if len(audio) == 0:
            return 0.0
        
        # Simple energy-based speech detection
        frame_length = int(0.025 * sr)
        hop_length = int(0.010 * sr)
        
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
        energy = np.sum(frames ** 2, axis=0)
        
        # Energy threshold for speech detection
        energy_threshold = np.percentile(energy, 30)
        speech_frames = np.sum(energy > energy_threshold)
        
        return speech_frames / len(energy) if len(energy) > 0 else 0.0
    
    def _check_complete_words(self, text: str) -> bool:
        """Check if segment contains complete words/phrases."""
        text = text.strip()
        
        if not text:
            return False
        
        # Check for common incomplete patterns
        incomplete_patterns = [
            text.endswith('...'),
            text.startswith('...'),
            len(text.split()) < 2,
            text.lower() in ['uh', 'um', 'ah', 'oh', 'the', 'and', 'to', 'a'],
            text.endswith(' the'),
            text.endswith(' a'),
            text.endswith(' and'),
            text.startswith('the ') and len(text.split()) < 4,
        ]
        
        return not any(incomplete_patterns)
    
    def _calculate_energy_stability(self, audio: np.ndarray, sr: int) -> float:
        """Calculate how stable the energy is throughout the segment."""
        if len(audio) == 0:
            return 0.0
        
        # Calculate RMS energy in windows
        window_size = int(0.1 * sr)  # 100ms windows
        rms_values = []
        
        for i in range(0, len(audio) - window_size, window_size // 2):
            window = audio[i:i + window_size]
            rms = np.sqrt(np.mean(window ** 2))
            rms_values.append(rms)
        
        if len(rms_values) < 2:
            return 1.0
        
        # Calculate coefficient of variation (lower = more stable)
        mean_rms = np.mean(rms_values)
        std_rms = np.std(rms_values)
        
        if mean_rms == 0:
            return 0.0
        
        cv = std_rms / mean_rms
        # Convert to stability score (1 = very stable, 0 = very unstable)
        return max(0.0, 1.0 - min(1.0, cv))
    
    def _calculate_quality_score(self, duration: float, snr_db: float, speech_ratio: float,
                               word_count: int, complete_words: bool, energy_stability: float) -> float:
        """Calculate overall quality score from 0-1."""
        
        # Duration score (optimal around 3-8 seconds)
        if duration < 1.0:
            duration_score = 0.0
        elif duration < 2.0:
            duration_score = 0.3
        elif duration <= 8.0:
            duration_score = 1.0
        elif duration <= 15.0:
            duration_score = 0.8
        else:
            duration_score = 0.4
        
        # SNR score
        snr_score = min(1.0, max(0.0, (snr_db - 8.0) / 20.0))  # 8-28 dB range
        
        # Speech ratio score
        speech_score = speech_ratio
        
        # Word count score
        word_score = min(1.0, word_count / 5.0)  # Optimal at 5+ words
        
        # Complete words bonus
        complete_bonus = 1.0 if complete_words else 0.4
        
        # Energy stability score
        stability_score = energy_stability
        
        # Weighted combination
        quality_score = (
            duration_score * 0.25 +
            snr_score * 0.25 +
            speech_score * 0.20 +
            word_score * 0.15 +
            stability_score * 0.05 +
            complete_bonus * 0.10
        )
        
        return max(0.0, min(1.0, quality_score))
    
    def _should_accept_segment(self, quality: SegmentQuality) -> Tuple[bool, Optional[str]]:
        """Determine if segment should be accepted based on quality thresholds."""
        thresholds = self.config['quality_thresholds']
        
        # Check each threshold
        if quality.duration < thresholds['min_duration']:
            return False, f"Too short: {quality.duration:.1f}s < {thresholds['min_duration']}s"
        
        if quality.duration > thresholds['max_duration']:
            return False, f"Too long: {quality.duration:.1f}s > {thresholds['max_duration']}s"
        
        if quality.snr_db < thresholds['min_snr_db']:
            return False, f"Low SNR: {quality.snr_db:.1f}dB < {thresholds['min_snr_db']}dB"
        
        if quality.speech_ratio < thresholds['min_speech_ratio']:
            return False, f"Low speech ratio: {quality.speech_ratio:.2f} < {thresholds['min_speech_ratio']}"
        
        if quality.word_count < thresholds['min_word_count']:
            return False, f"Too few words: {quality.word_count} < {thresholds['min_word_count']}"
        
        if not quality.complete_words:
            return False, "Incomplete words/phrases"
        
        if quality.quality_score < thresholds['min_quality_score']:
            return False, f"Low quality score: {quality.quality_score:.2f} < {thresholds['min_quality_score']}"
        
        return True, None
    
    def _apply_audio_processing(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply audio processing improvements."""
        processed = audio.copy()
        
        # Normalize volume if requested
        if self.config['audio_processing']['normalize_volume']:
            max_val = np.max(np.abs(processed))
            if max_val > 0:
                processed = processed / max_val * 0.95  # Normalize to 95% to avoid clipping
        
        # Apply fade in/out to avoid clicks
        fade_in_samples = int(self.config['audio_processing']['fade_in_ms'] * sr / 1000)
        fade_out_samples = int(self.config['audio_processing']['fade_out_ms'] * sr / 1000)
        
        if len(processed) > fade_in_samples + fade_out_samples:
            # Fade in
            if fade_in_samples > 0:
                fade_in = np.linspace(0, 1, fade_in_samples)
                processed[:fade_in_samples] *= fade_in
            
            # Fade out
            if fade_out_samples > 0:
                fade_out = np.linspace(1, 0, fade_out_samples)
                processed[-fade_out_samples:] *= fade_out
        
        return processed
    
    def _save_segment_metadata(self, segment: ProcessedSegment, metadata_path: Path):
        """Save processed segment metadata."""
        metadata = {
            'original_file': segment.original_file,
            'speaker': segment.speaker,
            'original_start': float(segment.start),
            'original_end': float(segment.end),
            'extended_start': float(segment.extended_start),
            'extended_end': float(segment.extended_end),
            'duration': float(segment.quality.duration),
            'text': segment.text,
            'confidence': float(segment.confidence),
            'quality': {
                'snr_db': float(segment.quality.snr_db),
                'speech_ratio': float(segment.quality.speech_ratio),
                'word_count': int(segment.quality.word_count),
                'complete_words': bool(segment.quality.complete_words),
                'energy_stability': float(segment.quality.energy_stability),
                'quality_score': float(segment.quality.quality_score)
            },
            'accepted': bool(segment.accepted),
            'rejection_reason': segment.rejection_reason
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(metadata), f, indent=2, ensure_ascii=False)
    
    def _save_speaker_report(self, segments: List[ProcessedSegment], output_dir: Path):
        """Save processing report for a speaker."""
        accepted = [s for s in segments if s.accepted]
        rejected = [s for s in segments if not s.accepted]
        
        report = {
            'speaker': output_dir.name,
            'processing_summary': {
                'total_segments': len(segments),
                'accepted_segments': len(accepted),
                'rejected_segments': len(rejected),
                'acceptance_rate': len(accepted) / len(segments) if segments else 0,
                'total_duration_accepted': sum(s.quality.duration for s in accepted),
                'avg_quality_score': np.mean([s.quality.quality_score for s in accepted]) if accepted else 0
            },
            'rejection_reasons': {},
            'quality_distribution': {
                'duration_range': [min(s.quality.duration for s in accepted), max(s.quality.duration for s in accepted)] if accepted else [0, 0],
                'snr_range': [min(s.quality.snr_db for s in accepted), max(s.quality.snr_db for s in accepted)] if accepted else [0, 0],
                'avg_word_count': np.mean([s.quality.word_count for s in accepted]) if accepted else 0
            }
        }
        
        # Count rejection reasons
        for segment in rejected:
            reason = segment.rejection_reason or 'Unknown'
            report['rejection_reasons'][reason] = report['rejection_reasons'].get(reason, 0) + 1
        
        # Save report
        report_path = output_dir / 'processing_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(report), f, indent=2, ensure_ascii=False)
    
    def _generate_quality_report(self, processing_results: List[Dict[str, Any]], output_path: Path) -> Dict[str, Any]:
        """Generate overall quality report."""
        successful_results = [r for r in processing_results if r['success']]
        
        if not successful_results:
            return {'error': 'No successful processing results'}
        
        total_input = sum(r.get('segments_input', 0) for r in successful_results)
        total_accepted = sum(r.get('segments_accepted', 0) for r in successful_results)
        total_duration = sum(r.get('total_duration', 0) for r in successful_results)
        
        # Collect all rejection reasons
        all_rejection_reasons = {}
        for result in successful_results:
            for reason in result.get('rejection_reasons', []):
                all_rejection_reasons[reason] = all_rejection_reasons.get(reason, 0) + 1
        
        report = {
            'overall_statistics': {
                'speakers_processed': len(successful_results),
                'total_segments_input': total_input,
                'total_segments_accepted': total_accepted,
                'overall_acceptance_rate': total_accepted / total_input if total_input > 0 else 0,
                'total_duration_hours': total_duration / 3600,
                'avg_quality_score': np.mean([r.get('avg_quality_score', 0) for r in successful_results])
            },
            'rejection_analysis': all_rejection_reasons,
            'recommendations': self._generate_recommendations(successful_results, all_rejection_reasons)
        }
        
        # Save overall report
        report_path = output_path / 'overall_quality_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(report), f, indent=2, ensure_ascii=False)
        
        return report
    
    def _generate_recommendations(self, results: List[Dict[str, Any]], rejection_reasons: Dict[str, int]) -> List[str]:
        """Generate recommendations based on processing results."""
        recommendations = []
        
        total_rejected = sum(rejection_reasons.values())
        
        # Analyze common rejection reasons
        if 'Too short' in str(rejection_reasons) and total_rejected > 0:
            short_count = sum(count for reason, count in rejection_reasons.items() if 'Too short' in reason)
            if short_count / total_rejected > 0.3:
                recommendations.append("Consider reducing min_duration threshold - many segments rejected for being too short")
        
        if 'Low SNR' in str(rejection_reasons):
            snr_count = sum(count for reason, count in rejection_reasons.items() if 'Low SNR' in reason)
            if snr_count / total_rejected > 0.2:
                recommendations.append("Consider improving audio preprocessing or reducing min_snr_db threshold")
        
        if 'Incomplete words' in str(rejection_reasons):
            incomplete_count = sum(count for reason, count in rejection_reasons.items() if 'Incomplete words' in reason)
            if incomplete_count / total_rejected > 0.2:
                recommendations.append("Consider improving diarization boundaries or reducing word completeness requirements")
        
        # Overall acceptance rate recommendations
        avg_acceptance = np.mean([r.get('segments_accepted', 0) / max(1, r.get('segments_input', 1)) for r in results])
        if avg_acceptance < 0.3:
            recommendations.append("Low overall acceptance rate - consider relaxing quality thresholds")
        elif avg_acceptance > 0.9:
            recommendations.append("Very high acceptance rate - consider stricter quality thresholds for better TTS training")
        
        return recommendations 