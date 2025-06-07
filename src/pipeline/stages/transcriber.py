"""
Transcriber Pipeline Stage
Generates text transcripts from audio using Whisper and performs speaker diarization
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import json
import numpy as np
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import tempfile
import shutil

# Progress monitoring
from tqdm import tqdm

# Core audio processing
import librosa
import soundfile as sf

# Whisper for transcription
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    logging.warning("faster-whisper not available")

# Speaker diarization 
try:
    from pyannote.audio import Pipeline as DiarizationPipeline
    from pyannote.core import Annotation, Segment
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    logging.warning("pyannote.audio not available - speaker diarization will be disabled")

logger = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    """A transcript segment with speaker and timing information."""
    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    confidence: Optional[float] = None


@dataclass
class SpeakerSegment:
    """A speaker segment with timing information."""
    start: float
    end: float
    speaker: str
    confidence: float = 1.0


@dataclass
class TranscriptionResult:
    """Result of transcription and speaker diarization."""
    segments: List[TranscriptSegment]
    speakers: List[str]
    audio_duration: float
    success: bool
    error: Optional[str] = None


class Transcriber:
    """Generate transcripts from audio files using Whisper with optional speaker diarization."""
    
    def __init__(self, model_size: str = "large-v3", speaker_diarization: bool = False, 
                 device: str = "cpu", compute_type: str = "int8", diarization_config: Optional[str] = None):
        self.model_size = model_size
        self.speaker_diarization = speaker_diarization
        self.device = device
        self.compute_type = compute_type
        self.diarization_config_path = diarization_config or "config/audio/speaker_diarization.yaml"
        
        # Load diarization configuration
        self.diarization_config = self._load_diarization_config()
        
        # Initialize models
        self.whisper_model = None
        self.diarization_pipeline = None
        
        if not FASTER_WHISPER_AVAILABLE:
            logger.error("faster-whisper is required but not available")
            raise ImportError("faster-whisper is required for transcription")
        
        if speaker_diarization and not PYANNOTE_AVAILABLE:
            logger.warning("pyannote.audio not available - disabling speaker diarization")
            self.speaker_diarization = False
    
    def _load_diarization_config(self) -> Dict[str, Any]:
        """Load speaker diarization configuration."""
        try:
            import yaml
            with open(self.diarization_config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load diarization config from {self.diarization_config_path}: {e}")
            # Return default configuration
            return {
                'clustering': {
                    'min_speakers': 2,
                    'max_speakers': 30,
                    'threshold': 0.45
                },
                'model': {
                    'name': "pyannote/speaker-diarization-3.1"
                }
            }
    
    def _initialize_models(self):
        """Initialize Whisper and diarization models."""
        if self.whisper_model is None:
            logger.info(f"Loading Whisper model: {self.model_size}")
            self.whisper_model = WhisperModel(
                self.model_size, 
                device=self.device,
                compute_type=self.compute_type
            )
        
        if self.speaker_diarization and self.diarization_pipeline is None and PYANNOTE_AVAILABLE:
            logger.info("Loading speaker diarization pipeline")
            try:
                # Try to load with HuggingFace authentication
                import os
                auth_token = os.environ.get('HUGGINGFACE_TOKEN') or True
                model_name = self.diarization_config.get('model', {}).get('name', "pyannote/speaker-diarization-3.1")
                
                self.diarization_pipeline = DiarizationPipeline.from_pretrained(
                    model_name,
                    use_auth_token=auth_token
                )
                
                # Apply device configuration for diarization pipeline  
                diarization_device = self.diarization_config.get('model', {}).get('device', 'cpu')
                if diarization_device == 'cpu':
                    logger.info("🖥️  Using CPU for speaker diarization (safer for compatibility)")
                    import torch
                    self.diarization_pipeline.to(torch.device("cpu"))
                elif diarization_device == 'cuda' and self.device in ['cuda', 'auto']:
                    logger.info("🚀 Using CUDA for speaker diarization")
                    import torch
                    self.diarization_pipeline.to(torch.device("cuda"))
                else:
                    logger.info(f"⚙️  Using {diarization_device} for speaker diarization")
                
                # Log configuration (min/max speakers will be applied at inference time)
                clustering_config = self.diarization_config.get('clustering', {})
                logger.info(f"Diarization pipeline loaded on {diarization_device}. Will use runtime params: "
                           f"min_speakers={clustering_config.get('min_speakers', 'auto')}, "
                           f"max_speakers={clustering_config.get('max_speakers', 'auto')}")
                        
            except Exception as e:
                logger.error(f"Failed to load diarization pipeline: {e}")
                logger.error("Please set up HuggingFace authentication:")
                logger.error("1. Visit https://huggingface.co/pyannote/speaker-diarization-3.1 and accept terms")
                logger.error("2. Get token from https://huggingface.co/settings/tokens")
                logger.error("3. Run: export HUGGINGFACE_TOKEN=your_token_here")
                logger.error("4. Or run: huggingface-cli login")
                self.speaker_diarization = False
    
    async def process_directory(self, input_dir: str, output_dir: str) -> Dict[str, Any]:
        """Process all audio files in directory with transcription and speaker diarization."""
        logger.info(f"Processing audio from {input_dir} to {output_dir}")
        
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
        
        # Create output directories
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different outputs
        transcripts_dir = output_path / "transcripts"
        speakers_dir = output_path / "speakers"
        segments_dir = output_path / "segments"
        
        transcripts_dir.mkdir(exist_ok=True)
        if self.speaker_diarization:
            speakers_dir.mkdir(exist_ok=True)
            segments_dir.mkdir(exist_ok=True)
        
        # Find audio files
        audio_files = self._find_audio_files(input_path)
        
        if not audio_files:
            return {
                'success': True,
                'message': 'No audio files found to process',
                'files_processed': 0,
                'output_path': str(output_path)
            }
        
        logger.info(f"Found {len(audio_files)} audio files to process")
        
        # Initialize models
        self._initialize_models()
        
        # Process files in parallel (but limit concurrency for memory)
        max_workers = 2 if self.speaker_diarization else 4
        
        logger.info(f"Processing {len(audio_files)} files with {max_workers} workers...")
        if self.speaker_diarization:
            logger.info("⚠️  Speaker diarization enabled - this may take significantly longer")
            
        processing_results = await self._process_files_parallel(
            audio_files, transcripts_dir, speakers_dir, segments_dir, max_workers
        )
        
        # Calculate results
        successful_files = [r for r in processing_results if r['success']]
        failed_files = [r for r in processing_results if not r['success']]
        
        # Generate summary
        total_speakers = set()
        total_segments = 0
        for result in successful_files:
            if result.get('speakers'):
                total_speakers.update(result['speakers'])
            total_segments += result.get('segment_count', 0)
        
        return {
            'success': True,
            'files_processed': len(successful_files),
            'files_failed': len(failed_files),
            'output_path': str(output_path),
            'total_speakers': len(total_speakers),
            'unique_speakers': list(total_speakers),
            'total_segments': total_segments,
            'speaker_diarization_enabled': self.speaker_diarization,
            'errors': [r.get('error') for r in failed_files if r.get('error')]
        }
    
    def _find_audio_files(self, directory: Path) -> List[Path]:
        """Find all supported audio files in directory."""
        audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.aac', '.m4a']
        audio_files = []
        for ext in audio_extensions:
            # Use recursive glob only - this includes current directory without duplicates
            audio_files.extend(directory.glob(f"**/*{ext}"))
        
        # Remove duplicates and return sorted list
        return sorted(list(set(audio_files)))
    
    async def _process_files_parallel(self, audio_files: List[Path], 
                                    transcripts_dir: Path, speakers_dir: Path, 
                                    segments_dir: Path, max_workers: int) -> List[Dict[str, Any]]:
        """Process audio files in parallel."""
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # For speaker diarization, avoid progress bar conflicts with pyannote ProgressHook
            if self.speaker_diarization:
                logger.info("🎯 Processing with speaker diarization - pyannote will show detailed progress for each file")
                
                # Simple sequential processing to avoid tqdm conflicts
                results = []
                for i, file_path in enumerate(audio_files, 1):
                    logger.info(f"📁 Processing file {i}/{len(audio_files)}: {file_path.name}")
                    result = await asyncio.get_event_loop().run_in_executor(
                        executor, self._process_single_file, file_path, 
                        transcripts_dir, speakers_dir, segments_dir
                    )
                    status = "✅" if result['success'] else "❌"
                    logger.info(f"{status} Completed file {i}/{len(audio_files)}: {file_path.name}")
                    results.append(result)
                return results
            else:
                # Use progress bar for transcription-only (faster processing)
                pbar = tqdm(total=len(audio_files), desc="Processing audio files", unit="file")
                
                async def process_with_progress(file_path):
                    result = await asyncio.get_event_loop().run_in_executor(
                        executor, self._process_single_file, file_path, 
                        transcripts_dir, speakers_dir, segments_dir
                    )
                    pbar.update(1)
                    if result['success']:
                        pbar.set_postfix_str(f"✅ {file_path.name}")
                    else:
                        pbar.set_postfix_str(f"❌ {file_path.name}")
                    return result
                
                tasks = [process_with_progress(file_path) for file_path in audio_files]
                results = await asyncio.gather(*tasks)
                pbar.close()
                return results
    
    def _process_single_file(self, audio_file: Path, transcripts_dir: Path, 
                           speakers_dir: Path, segments_dir: Path) -> Dict[str, Any]:
        """Process a single audio file with transcription and speaker diarization."""
        import time
        start_time = time.time()
        
        try:
            logger.info(f"Processing {audio_file.name}")
            
            # Load audio for processing
            audio, sr = librosa.load(str(audio_file), sr=16000)  # Whisper expects 16kHz
            duration = len(audio) / sr
            logger.info(f"Loaded audio: {duration:.1f}s duration, {sr}Hz sample rate")
            
            # Step 1: Transcribe with Whisper
            logger.debug(f"Transcribing {audio_file.name}")
            segments, info = self.whisper_model.transcribe(
                str(audio_file),
                beam_size=5,
                language="en",  # Assuming English for Pokemon
                word_timestamps=True
            )
            
            # Convert segments to our format
            transcript_segments = []
            for segment in segments:
                transcript_segments.append(TranscriptSegment(
                    start=segment.start,
                    end=segment.end,
                    text=segment.text.strip(),
                    confidence=getattr(segment, 'avg_logprob', None)
                ))
            
            # Step 2: Speaker diarization (if enabled)
            speaker_segments = []
            speakers = []
            
            if self.speaker_diarization and self.diarization_pipeline:
                logger.debug(f"Performing speaker diarization on {audio_file.name}")
                
                # Apply clustering parameters at inference time
                clustering_config = self.diarization_config.get('clustering', {})
                
                # Build inference parameters
                inference_params = {}
                if 'min_speakers' in clustering_config:
                    inference_params['min_speakers'] = clustering_config['min_speakers']
                if 'max_speakers' in clustering_config:
                    inference_params['max_speakers'] = clustering_config['max_speakers']
                
                logger.debug(f"Running diarization with params: {inference_params}")
                
                # Run diarization with progress monitoring and configured parameters
                try:
                    from pyannote.audio.pipelines.utils.hook import ProgressHook
                    with ProgressHook() as hook:
                        diarization = self.diarization_pipeline(str(audio_file), hook=hook, **inference_params)
                except ImportError:
                    logger.warning("ProgressHook not available, running without progress monitoring")
                    diarization = self.diarization_pipeline(str(audio_file), **inference_params)
                
                speaker_labels = list(diarization.labels())
                logger.debug(f"Diarization completed - found {len(speaker_labels)} speakers: {speaker_labels}")
                
                # Convert diarization to our format
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    speaker_segments.append(SpeakerSegment(
                        start=turn.start,
                        end=turn.end,
                        speaker=speaker
                    ))
                    if speaker not in speakers:
                        speakers.append(speaker)
                
                # Align transcripts with speakers
                transcript_segments = self._align_transcripts_with_speakers(
                    transcript_segments, speaker_segments
                )
            
            # Step 3: Save outputs
            base_name = audio_file.stem
            
            # Save transcript
            transcript_file = transcripts_dir / f"{base_name}.json"
            self._save_transcript(transcript_segments, transcript_file, duration)
            
            # Save speaker information and segments
            segment_count = 0
            if self.speaker_diarization:
                # Save speaker timeline
                speakers_file = speakers_dir / f"{base_name}_speakers.json"
                self._save_speaker_timeline(speaker_segments, speakers_file, duration)
                
                # Extract and save speaker segments
                segment_count = self._extract_speaker_segments(
                    audio_file, transcript_segments, segments_dir
                )
            
            processing_time = time.time() - start_time
            rate = duration / processing_time if processing_time > 0 else 0
            logger.info(f"Successfully processed {audio_file.name} in {processing_time:.1f}s "
                       f"({rate:.1f}x real-time): {len(transcript_segments)} transcripts, "
                       f"{len(speakers)} speakers, {segment_count} segments extracted")
            
            return {
                'success': True,
                'file': str(audio_file),
                'transcript_count': len(transcript_segments),
                'speakers': speakers,
                'segment_count': segment_count,
                'duration': duration,
                'processing_time': processing_time
            }
            
        except Exception as e:
            logger.error(f"Failed to process {audio_file}: {e}")
            return {
                'success': False,
                'file': str(audio_file),
                'error': str(e)
            }
    
    def _align_transcripts_with_speakers(self, transcript_segments: List[TranscriptSegment], 
                                       speaker_segments: List[SpeakerSegment]) -> List[TranscriptSegment]:
        """Align transcript segments with speaker segments."""
        aligned_segments = []
        
        for transcript in transcript_segments:
            # Find the speaker segment that overlaps most with this transcript
            best_speaker = None
            best_overlap = 0
            
            for speaker_seg in speaker_segments:
                # Calculate overlap
                overlap_start = max(transcript.start, speaker_seg.start)
                overlap_end = min(transcript.end, speaker_seg.end)
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = speaker_seg.speaker
            
            # Create aligned segment
            aligned_segment = TranscriptSegment(
                start=transcript.start,
                end=transcript.end,
                text=transcript.text,
                speaker=best_speaker,
                confidence=transcript.confidence
            )
            aligned_segments.append(aligned_segment)
        
        return aligned_segments
    
    def _save_transcript(self, segments: List[TranscriptSegment], output_file: Path, duration: float):
        """Save transcript segments to JSON file."""
        transcript_data = {
            'duration': duration,
            'segments': [
                {
                    'start': seg.start,
                    'end': seg.end,
                    'text': seg.text,
                    'speaker': seg.speaker,
                    'confidence': seg.confidence
                }
                for seg in segments
            ]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(transcript_data, f, indent=2, ensure_ascii=False)
    
    def _save_speaker_timeline(self, speaker_segments: List[SpeakerSegment], 
                             output_file: Path, duration: float):
        """Save speaker timeline to JSON file."""
        speaker_data = {
            'duration': duration,
            'speakers': [
                {
                    'start': seg.start,
                    'end': seg.end,
                    'speaker': seg.speaker,
                    'confidence': seg.confidence
                }
                for seg in speaker_segments
            ]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(speaker_data, f, indent=2, ensure_ascii=False)
    
    def _extract_speaker_segments(self, audio_file: Path, transcript_segments: List[TranscriptSegment], 
                                segments_dir: Path) -> int:
        """Extract audio segments for each speaker and save them."""
        if not transcript_segments:
            return 0
        
        # Load original audio
        audio, sr = librosa.load(str(audio_file), sr=None)
        base_name = audio_file.stem
        
        segment_count = 0
        speaker_counts = {}
        
        for segment in transcript_segments:
            if not segment.speaker or not segment.text.strip():
                continue
            
            # Skip very short segments
            if segment.end - segment.start < 1.0:
                continue
            
            # Create speaker directory
            speaker_dir = segments_dir / segment.speaker
            speaker_dir.mkdir(exist_ok=True)
            
            # Extract audio segment
            start_sample = int(segment.start * sr)
            end_sample = int(segment.end * sr)
            segment_audio = audio[start_sample:end_sample]
            
            # Generate segment filename
            if segment.speaker not in speaker_counts:
                speaker_counts[segment.speaker] = 0
            speaker_counts[segment.speaker] += 1
            
            segment_filename = f"{base_name}_{segment.speaker}_{speaker_counts[segment.speaker]:03d}.wav"
            segment_path = speaker_dir / segment_filename
            
            # Save audio segment
            sf.write(str(segment_path), segment_audio, sr)
            
            # Save metadata
            metadata = {
                'original_file': str(audio_file),
                'start': segment.start,
                'end': segment.end,
                'duration': segment.end - segment.start,
                'text': segment.text,
                'speaker': segment.speaker,
                'confidence': segment.confidence
            }
            
            metadata_path = speaker_dir / f"{segment_filename}.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            segment_count += 1
        
        return segment_count
    
    # Legacy method for backwards compatibility
    async def transcribe_directory(self, input_dir: str, output_dir: str) -> Dict[str, Any]:
        """Legacy method - use process_directory instead."""
        return await self.process_directory(input_dir, output_dir) 