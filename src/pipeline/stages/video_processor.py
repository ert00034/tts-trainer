"""
Video Processor Pipeline Stage
Handles video file validation, metadata extraction, and preparation
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List
import subprocess
import json

from ...utils.file_utils import get_video_files, clean_filename


logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Pipeline stage for processing video files.
    
    Responsibilities:
    - Validate video file formats and integrity
    - Extract metadata (duration, resolution, codec, etc.)
    - Prepare videos for audio extraction
    - Filter out unsuitable videos
    """
    
    def __init__(self):
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']
        self.min_duration = 10.0  # Minimum video duration in seconds
        self.max_duration = 3600.0  # Maximum video duration in seconds
    
    async def process_directory(self, input_dir: str, output_dir: str) -> Dict[str, Any]:
        """
        Process all video files in a directory.
        
        Args:
            input_dir: Directory containing video files
            output_dir: Directory to store processed video metadata
            
        Returns:
            Dictionary with processing results
        """
        logger.info(f"Processing videos in {input_dir}")
        
        try:
            video_files = get_video_files(input_dir)
            if not video_files:
                return {
                    'success': False,
                    'error': 'No video files found in input directory',
                    'files_processed': 0
                }
            
            results = {
                'success': True,
                'files_processed': 0,
                'files_valid': 0,
                'files_skipped': 0,
                'output_path': output_dir,
                'video_metadata': [],
                'errors': []
            }
            
            # Create output directory
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Process each video file
            for video_file in video_files:
                try:
                    metadata = await self._process_single_video(video_file)
                    results['files_processed'] += 1
                    
                    if metadata.get('valid', False):
                        results['files_valid'] += 1
                        results['video_metadata'].append(metadata)
                        logger.info(f"✅ Processed: {video_file.name}")
                    else:
                        results['files_skipped'] += 1
                        logger.warning(f"⚠️ Skipped: {video_file.name} - {metadata.get('skip_reason', 'Unknown')}")
                
                except Exception as e:
                    results['files_skipped'] += 1
                    results['errors'].append(f"{video_file.name}: {str(e)}")
                    logger.error(f"❌ Error processing {video_file.name}: {e}")
            
            # Save metadata summary
            metadata_file = Path(output_dir) / "video_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(results['video_metadata'], f, indent=2)
            
            logger.info(f"Video processing complete: {results['files_valid']}/{results['files_processed']} videos valid")
            return results
            
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'files_processed': 0
            }
    
    async def _process_single_video(self, video_file: Path) -> Dict[str, Any]:
        """Process a single video file and extract metadata."""
        try:
            # Extract video metadata using ffprobe
            metadata = await self._extract_metadata(video_file)
            
            # Validate video
            is_valid, skip_reason = self._validate_video(metadata)
            
            # Clean filename for output
            clean_name = clean_filename(video_file.stem)
            
            result = {
                'original_path': str(video_file),
                'clean_filename': clean_name,
                'valid': is_valid,
                'skip_reason': skip_reason if not is_valid else None,
                **metadata
            }
            
            return result
            
        except Exception as e:
            return {
                'original_path': str(video_file),
                'valid': False,
                'skip_reason': f'Processing error: {str(e)}',
                'error': str(e)
            }
    
    async def _extract_metadata(self, video_file: Path) -> Dict[str, Any]:
        """Extract metadata from video using ffprobe."""
        try:
            # Run ffprobe to extract metadata
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                str(video_file)
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise Exception(f"ffprobe failed: {stderr.decode()}")
            
            ffprobe_data = json.loads(stdout.decode())
            
            # Extract relevant information
            format_info = ffprobe_data.get('format', {})
            video_stream = None
            audio_stream = None
            
            for stream in ffprobe_data.get('streams', []):
                if stream.get('codec_type') == 'video' and video_stream is None:
                    video_stream = stream
                elif stream.get('codec_type') == 'audio' and audio_stream is None:
                    audio_stream = stream
            
            # Compile metadata
            metadata = {
                'duration': float(format_info.get('duration', 0)),
                'file_size': int(format_info.get('size', 0)),
                'format_name': format_info.get('format_name', ''),
                'bit_rate': int(format_info.get('bit_rate', 0)),
                'has_video': video_stream is not None,
                'has_audio': audio_stream is not None
            }
            
            if video_stream:
                metadata.update({
                    'video_codec': video_stream.get('codec_name', ''),
                    'width': int(video_stream.get('width', 0)),
                    'height': int(video_stream.get('height', 0)),
                    'fps': self._parse_framerate(video_stream.get('r_frame_rate', '0/1'))
                })
            
            if audio_stream:
                metadata.update({
                    'audio_codec': audio_stream.get('codec_name', ''),
                    'audio_sample_rate': int(audio_stream.get('sample_rate', 0)),
                    'audio_channels': int(audio_stream.get('channels', 0)),
                    'audio_bit_rate': int(audio_stream.get('bit_rate', 0))
                })
            
            return metadata
            
        except Exception as e:
            # Fallback: basic file info
            stat = video_file.stat()
            return {
                'duration': 0,
                'file_size': stat.st_size,
                'format_name': video_file.suffix.lstrip('.'),
                'has_video': True,  # Assume true for video files
                'has_audio': True,  # Assume true for video files
                'extraction_error': str(e)
            }
    
    def _parse_framerate(self, framerate_str: str) -> float:
        """Parse framerate from ffprobe format (e.g., '30/1')."""
        try:
            if '/' in framerate_str:
                num, den = framerate_str.split('/')
                return float(num) / float(den)
            return float(framerate_str)
        except (ValueError, ZeroDivisionError):
            return 0.0
    
    def _validate_video(self, metadata: Dict[str, Any]) -> tuple[bool, str]:
        """
        Validate video metadata to determine if suitable for processing.
        
        Returns:
            Tuple of (is_valid, skip_reason)
        """
        # Check duration
        duration = metadata.get('duration', 0)
        if duration < self.min_duration:
            return False, f"Too short: {duration:.1f}s (min: {self.min_duration}s)"
        
        if duration > self.max_duration:
            return False, f"Too long: {duration:.1f}s (max: {self.max_duration}s)"
        
        # Check for audio stream
        if not metadata.get('has_audio', False):
            return False, "No audio stream found"
        
        # Check file size (avoid corrupted files)
        file_size = metadata.get('file_size', 0)
        if file_size < 1024 * 1024:  # Less than 1MB
            return False, f"File too small: {file_size / 1024:.1f}KB"
        
        # Check audio properties if available
        sample_rate = metadata.get('audio_sample_rate', 0)
        if sample_rate > 0 and sample_rate < 16000:
            return False, f"Audio sample rate too low: {sample_rate}Hz"
        
        # All checks passed
        return True, ""
    
    def get_processing_stats(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Get summary statistics from processing results."""
        if not results.get('success', False):
            return {'error': results.get('error', 'Unknown error')}
        
        total_duration = sum(
            metadata.get('duration', 0) 
            for metadata in results.get('video_metadata', [])
        )
        
        total_size = sum(
            metadata.get('file_size', 0) 
            for metadata in results.get('video_metadata', [])
        )
        
        return {
            'total_files': results['files_processed'],
            'valid_files': results['files_valid'],
            'skipped_files': results['files_skipped'],
            'total_duration_minutes': total_duration / 60,
            'total_size_mb': total_size / (1024 * 1024),
            'average_duration_minutes': (total_duration / results['files_valid'] / 60) if results['files_valid'] > 0 else 0
        } 