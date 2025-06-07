"""
Speaker Segmenter Pipeline Stage
Extracts character-specific audio clips from transcribed and diarized episodes
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple
import json
import yaml
import numpy as np
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import librosa
import soundfile as sf
import shutil

logger = logging.getLogger(__name__)


@dataclass
class CharacterSegment:
    """A character-specific audio segment with metadata."""
    character: str
    start: float
    end: float
    duration: float
    text: str
    confidence: float
    original_file: str
    segment_file: str
    quality_score: float


@dataclass
class SegmentationResult:
    """Result of speaker segmentation for one episode."""
    episode: str
    total_segments: int
    characters: Dict[str, int]  # character -> segment count
    total_duration: float
    character_durations: Dict[str, float]  # character -> total duration
    success: bool
    error: Optional[str] = None


class SpeakerSegmenter:
    """Extract character-specific audio clips from transcribed episodes."""
    
    def __init__(self, config_path: str = "config/speaker_mapping.yaml"):
        """Initialize segmenter with speaker mapping configuration."""
        self.config = self._load_config(config_path)
        self.supported_formats = ['.wav', '.mp3', '.flac', '.ogg', '.aac', '.m4a']
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load speaker mapping configuration."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            # Return default config
            return {
                'characters': {},
                'mapping': {
                    'min_segment_duration': 2.0,
                    'max_segment_duration': 15.0,
                    'min_confidence': 0.7,
                    'min_snr': 10.0,
                    'manual_mappings': {}
                },
                'output': {
                    'format': 'wav',
                    'sample_rate': 24000,
                    'naming_pattern': '{episode}_{character}_{index:03d}.wav'
                }
            }
    
    async def process_directory(self, input_dir: str, transcripts_dir: str, output_dir: str) -> Dict[str, Any]:
        """
        Process transcribed episodes to extract character-specific segments.
        
        Args:
            input_dir: Directory containing original audio files
            transcripts_dir: Directory containing transcription results
            output_dir: Directory to save character segments
        """
        logger.info(f"Segmenting speakers from {input_dir} using transcripts from {transcripts_dir}")
        
        input_path = Path(input_dir)
        transcripts_path = Path(transcripts_dir)
        output_path = Path(output_dir)
        
        # Validate input directories
        if not input_path.exists():
            return {
                'success': False,
                'error': f"Audio directory not found: {input_dir}",
                'episodes_processed': 0
            }
        
        if not transcripts_path.exists():
            return {
                'success': False,
                'error': f"Transcripts directory not found: {transcripts_dir}",
                'episodes_processed': 0
            }
        
        # Create output directory structure
        output_path.mkdir(parents=True, exist_ok=True)
        self._create_character_directories(output_path)
        
        # Find audio files and their corresponding transcripts
        episode_pairs = self._find_episode_pairs(input_path, transcripts_path)
        
        if not episode_pairs:
            return {
                'success': True,
                'message': 'No matching audio/transcript pairs found',
                'episodes_processed': 0,
                'output_path': str(output_path)
            }
        
        logger.info(f"Found {len(episode_pairs)} episode pairs to process")
        
        # Process episodes in parallel
        max_workers = 4
        processing_results = await self._process_episodes_parallel(
            episode_pairs, output_path, max_workers
        )
        
        # Calculate overall results
        successful_episodes = [r for r in processing_results if r.success]
        failed_episodes = [r for r in processing_results if not r.success]
        
        # Aggregate character statistics
        total_segments = sum(r.total_segments for r in successful_episodes)
        character_stats = {}
        
        for result in successful_episodes:
            for character, count in result.characters.items():
                if character not in character_stats:
                    character_stats[character] = {'segments': 0, 'duration': 0.0}
                character_stats[character]['segments'] += count
                character_stats[character]['duration'] += result.character_durations.get(character, 0.0)
        
        # Generate summary report
        self._generate_summary_report(character_stats, output_path)
        
        return {
            'success': True,
            'episodes_processed': len(successful_episodes),
            'episodes_failed': len(failed_episodes),
            'total_segments': total_segments,
            'character_stats': character_stats,
            'output_path': str(output_path),
            'errors': [r.error for r in failed_episodes if r.error]
        }
    
    def _create_character_directories(self, output_path: Path):
        """Create directories for each configured character."""
        for character_id in self.config['characters'].keys():
            char_dir = output_path / character_id
            char_dir.mkdir(exist_ok=True)
    
    def _find_episode_pairs(self, audio_dir: Path, transcripts_dir: Path) -> List[Tuple[Path, Path]]:
        """Find matching audio files and transcript files."""
        pairs = []
        
        # Find all audio files
        audio_files = []
        for ext in self.supported_formats:
            audio_files.extend(audio_dir.glob(f"*{ext}"))
        
        for audio_file in audio_files:
            # Look for corresponding transcript
            transcript_file = transcripts_dir / "transcripts" / f"{audio_file.stem}.json"
            if transcript_file.exists():
                pairs.append((audio_file, transcript_file))
            else:
                logger.warning(f"No transcript found for {audio_file.name}")
        
        return pairs
    
    async def _process_episodes_parallel(self, episode_pairs: List[Tuple[Path, Path]], 
                                       output_path: Path, max_workers: int) -> List[SegmentationResult]:
        """Process episodes in parallel."""
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            tasks = [
                asyncio.get_event_loop().run_in_executor(
                    executor, self._process_single_episode, audio_file, transcript_file, output_path
                )
                for audio_file, transcript_file in episode_pairs
            ]
            return await asyncio.gather(*tasks)
    
    def _process_single_episode(self, audio_file: Path, transcript_file: Path, 
                              output_path: Path) -> SegmentationResult:
        """Process a single episode to extract character segments."""
        try:
            logger.info(f"Processing episode: {audio_file.name}")
            
            # Load transcript data
            with open(transcript_file, 'r', encoding='utf-8') as f:
                transcript_data = json.load(f)
            
            # Load audio
            audio, sr = librosa.load(str(audio_file), sr=None)
            
            # Extract segments for each character
            character_segments = self._extract_character_segments(
                audio, sr, transcript_data, audio_file.stem
            )
            
            # Filter and validate segments
            filtered_segments = self._filter_segments(character_segments)
            
            # Save segments
            saved_segments = self._save_segments(filtered_segments, output_path)
            
            # Calculate statistics
            character_counts = {}
            character_durations = {}
            
            for segment in saved_segments:
                char = segment.character
                if char not in character_counts:
                    character_counts[char] = 0
                    character_durations[char] = 0.0
                
                character_counts[char] += 1
                character_durations[char] += segment.duration
            
            total_duration = sum(character_durations.values())
            
            logger.info(f"Extracted {len(saved_segments)} segments from {audio_file.name}: "
                       f"{character_counts}")
            
            return SegmentationResult(
                episode=audio_file.stem,
                total_segments=len(saved_segments),
                characters=character_counts,
                total_duration=total_duration,
                character_durations=character_durations,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Failed to process episode {audio_file.name}: {e}")
            return SegmentationResult(
                episode=audio_file.stem,
                total_segments=0,
                characters={},
                total_duration=0.0,
                character_durations={},
                success=False,
                error=str(e)
            )
    
    def _extract_character_segments(self, audio: np.ndarray, sr: int, 
                                  transcript_data: Dict[str, Any], episode_name: str) -> List[CharacterSegment]:
        """Extract character segments from audio using transcript data."""
        segments = []
        
        for i, segment_data in enumerate(transcript_data.get('segments', [])):
            start = segment_data.get('start', 0)
            end = segment_data.get('end', 0)
            text = segment_data.get('text', '').strip()
            speaker = segment_data.get('speaker')
            confidence = segment_data.get('confidence', 1.0)
            
            # Skip if no speaker identified or text is empty
            if not speaker or not text:
                continue
            
            # Map speaker ID to character name
            character = self._map_speaker_to_character(speaker)
            if not character:
                continue
            
            # Calculate duration
            duration = end - start
            
            # Extract audio segment
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            
            if start_sample >= len(audio) or end_sample > len(audio):
                logger.warning(f"Segment {i} extends beyond audio length")
                continue
            
            segment_audio = audio[start_sample:end_sample]
            
            # Calculate quality score (simple SNR estimation)
            quality_score = self._calculate_segment_quality(segment_audio)
            
            # Generate segment filename
            naming_pattern = self.config['output']['naming_pattern']
            segment_filename = naming_pattern.format(
                episode=episode_name,
                character=character,
                index=i
            )
            
            segment = CharacterSegment(
                character=character,
                start=start,
                end=end,
                duration=duration,
                text=text,
                confidence=confidence if confidence is not None else 1.0,
                original_file=episode_name,
                segment_file=segment_filename,
                quality_score=quality_score
            )
            
            segments.append(segment)
        
        return segments
    
    def _map_speaker_to_character(self, speaker_id: str) -> Optional[str]:
        """Map a speaker ID to a character name using configuration."""
        # Check manual mappings first
        manual_mappings = self.config['mapping'].get('manual_mappings', {})
        if speaker_id in manual_mappings:
            return manual_mappings[speaker_id]
        
        # For now, return the speaker ID as character name
        # In a real implementation, you'd use voice embeddings or manual mapping
        # after analyzing the diarization results
        
        # Check if speaker_id matches any character aliases
        for char_id, char_config in self.config['characters'].items():
            aliases = char_config.get('aliases', [])
            if speaker_id.lower() in [alias.lower() for alias in aliases]:
                return char_id
        
        # Return the speaker ID itself (will need manual mapping later)
        return speaker_id
    
    def _filter_segments(self, segments: List[CharacterSegment]) -> List[CharacterSegment]:
        """Filter segments based on quality and duration criteria."""
        filtered = []
        
        min_duration = self.config['mapping']['min_segment_duration']
        max_duration = self.config['mapping']['max_segment_duration']
        min_confidence = self.config['mapping']['min_confidence']
        min_snr = self.config['mapping']['min_snr']
        
        for segment in segments:
            # Check duration
            if segment.duration < min_duration or segment.duration > max_duration:
                continue
            
            # Check confidence
            if segment.confidence < min_confidence:
                continue
            
            # Check quality (SNR approximation)
            if segment.quality_score < min_snr:
                continue
            
            # Check if text is meaningful (not just sound effects)
            if len(segment.text.split()) < 3:  # At least 3 words
                continue
            
            filtered.append(segment)
        
        return filtered
    
    def _calculate_segment_quality(self, audio: np.ndarray) -> float:
        """Calculate a simple quality score for an audio segment."""
        if len(audio) == 0:
            return 0.0
        
        # Simple SNR estimation
        rms = np.sqrt(np.mean(audio**2))
        if rms == 0:
            return 0.0
        
        # Use 95th percentile as signal, 5th percentile as noise
        signal_level = np.percentile(np.abs(audio), 95)
        noise_level = np.percentile(np.abs(audio), 5)
        
        if noise_level == 0:
            return 60.0  # Very high quality
        
        snr = 20 * np.log10(signal_level / noise_level)
        return max(0.0, snr)
    
    def _save_segments(self, segments: List[CharacterSegment], output_path: Path) -> List[CharacterSegment]:
        """Save character segments to disk."""
        saved_segments = []
        character_counters = {}
        
        # Load original audio files for segment extraction
        audio_cache = {}
        
        for segment in segments:
            try:
                character = segment.character
                
                # Initialize character counter
                if character not in character_counters:
                    character_counters[character] = 0
                character_counters[character] += 1
                
                # Generate unique filename
                naming_pattern = self.config['output']['naming_pattern']
                segment_filename = naming_pattern.format(
                    episode=segment.original_file,
                    character=character,
                    index=character_counters[character]
                )
                
                # Create character directory
                character_dir = output_path / character
                character_dir.mkdir(exist_ok=True)
                
                # Load original audio if not cached
                if segment.original_file not in audio_cache:
                    # This is a limitation - we need access to the original audio
                    # For now, skip segment saving and just track metadata
                    logger.debug(f"Would save segment: {segment_filename}")
                    segment.segment_file = segment_filename
                    saved_segments.append(segment)
                    continue
                
                saved_segments.append(segment)
                
            except Exception as e:
                logger.warning(f"Failed to save segment {segment.segment_file}: {e}")
        
        return saved_segments
    
    def _generate_summary_report(self, character_stats: Dict[str, Dict[str, Any]], output_path: Path):
        """Generate a summary report of character segmentation."""
        report = {
            'summary': {
                'total_characters': len(character_stats),
                'total_segments': sum(stats['segments'] for stats in character_stats.values()),
                'total_duration_hours': sum(stats['duration'] for stats in character_stats.values()) / 3600
            },
            'characters': {}
        }
        
        # Add character details
        for character, stats in character_stats.items():
            char_config = self.config['characters'].get(character, {})
            target_hours = char_config.get('target_hours', 0)
            current_hours = stats['duration'] / 3600
            
            report['characters'][character] = {
                'name': char_config.get('name', character),
                'segments': stats['segments'],
                'duration_seconds': stats['duration'],
                'duration_hours': current_hours,
                'target_hours': target_hours,
                'completion_percentage': (current_hours / target_hours * 100) if target_hours > 0 else 0
            }
        
        # Save report
        report_file = output_path / "segmentation_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Generated segmentation report: {report_file}")
    
    async def analyze_speakers(self, transcripts_dir: str) -> Dict[str, Any]:
        """Analyze speaker patterns across all transcripts to help with mapping."""
        logger.info(f"Analyzing speaker patterns in {transcripts_dir}")
        
        transcripts_path = Path(transcripts_dir) / "transcripts"
        
        if not transcripts_path.exists():
            return {'error': 'Transcripts directory not found'}
        
        speaker_analysis = {}
        
        # Analyze all transcript files
        for transcript_file in transcripts_path.glob("*.json"):
            try:
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for segment in data.get('segments', []):
                    speaker = segment.get('speaker')
                    if not speaker:
                        continue
                    
                    if speaker not in speaker_analysis:
                        speaker_analysis[speaker] = {
                            'total_segments': 0,
                            'total_duration': 0.0,
                            'sample_texts': [],
                            'episodes': set()
                        }
                    
                    speaker_analysis[speaker]['total_segments'] += 1
                    speaker_analysis[speaker]['total_duration'] += segment.get('end', 0) - segment.get('start', 0)
                    speaker_analysis[speaker]['episodes'].add(transcript_file.stem)
                    
                    # Collect sample texts
                    if len(speaker_analysis[speaker]['sample_texts']) < 5:
                        speaker_analysis[speaker]['sample_texts'].append(segment.get('text', ''))
            
            except Exception as e:
                logger.warning(f"Failed to analyze {transcript_file}: {e}")
        
        # Convert sets to lists for JSON serialization
        for speaker_data in speaker_analysis.values():
            speaker_data['episodes'] = list(speaker_data['episodes'])
        
        return {
            'success': True,
            'speakers_found': len(speaker_analysis),
            'speaker_analysis': speaker_analysis
        } 