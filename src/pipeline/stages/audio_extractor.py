"""
Audio Extractor Pipeline Stage
Extracts audio from video files using FFmpeg
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List
from utils.file_utils import get_video_files, clean_filename

logger = logging.getLogger(__name__)


class AudioExtractor:
    """Extract audio from video files using FFmpeg."""
    
    def __init__(self, output_format: str = "wav", sample_rate: int = 24000):
        self.output_format = output_format
        self.sample_rate = sample_rate
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']
    
    async def extract_from_directory(self, input_dir: str, output_dir: str) -> Dict[str, Any]:
        """Extract audio from all videos in directory."""
        logger.info(f"Extracting audio from {input_dir} to {output_dir}")
        
        try:
            # Get all video files
            video_files = get_video_files(input_dir)
            
            if not video_files:
                logger.warning(f"No video files found in {input_dir}")
                return {
                    'success': True,
                    'files_processed': 0,
                    'files_successful': 0,
                    'files_failed': 0,
                    'output_path': output_dir,
                    'message': 'No video files found in input directory'
                }
            
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Found {len(video_files)} video files to process")
            
            # Process each video file
            results = {
                'success': True,
                'files_processed': len(video_files),
                'files_successful': 0,
                'files_failed': 0,
                'output_path': output_dir,
                'extracted_files': [],
                'failed_files': [],
                'errors': []
            }
            
            for video_file in video_files:
                try:
                    output_file = await self._extract_audio_from_video(video_file, output_path)
                    if output_file:
                        results['files_successful'] += 1
                        results['extracted_files'].append(str(output_file))
                        logger.info(f"✅ Extracted audio: {video_file.name} → {output_file.name}")
                    else:
                        results['files_failed'] += 1
                        results['failed_files'].append(str(video_file))
                        logger.error(f"❌ Failed to extract audio: {video_file.name}")
                        
                except Exception as e:
                    results['files_failed'] += 1
                    results['failed_files'].append(str(video_file))
                    results['errors'].append(f"{video_file.name}: {str(e)}")
                    logger.error(f"❌ Error extracting audio from {video_file.name}: {e}")
            
            logger.info(f"Audio extraction complete: {results['files_successful']}/{results['files_processed']} successful")
            return results
            
        except Exception as e:
            logger.error(f"Audio extraction failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'files_processed': 0,
                'files_successful': 0,
                'files_failed': 0
            }
    
    async def _extract_audio_from_video(self, video_file: Path, output_dir: Path) -> Path:
        """Extract audio from a single video file using FFmpeg."""
        try:
            # Generate clean output filename
            clean_name = clean_filename(video_file.stem)
            output_file = output_dir / f"{clean_name}.{self.output_format}"
            
            # Ensure unique filename
            if output_file.exists():
                counter = 1
                while output_file.exists():
                    output_file = output_dir / f"{clean_name}_{counter}.{self.output_format}"
                    counter += 1
            
            # Build FFmpeg command
            cmd = [
                'ffmpeg',
                '-i', str(video_file),           # Input video file
                '-vn',                           # No video (audio only)
                '-acodec', 'pcm_s16le',         # Audio codec: 16-bit PCM
                '-ar', str(self.sample_rate),    # Sample rate
                '-ac', '1',                      # Mono audio
                '-y',                            # Overwrite output file
                str(output_file)                 # Output audio file
            ]
            
            logger.debug(f"Running FFmpeg command: {' '.join(cmd)}")
            
            # Execute FFmpeg
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr.decode('utf-8', errors='ignore')
                logger.error(f"FFmpeg failed for {video_file.name}: {error_msg}")
                return None
            
            # Verify output file was created
            if not output_file.exists() or output_file.stat().st_size == 0:
                logger.error(f"Output file not created or empty: {output_file}")
                return None
            
            return output_file
            
        except Exception as e:
            logger.error(f"Failed to extract audio from {video_file}: {e}")
            return None
    
    async def extract_single_file(self, video_file: str, output_file: str) -> bool:
        """Extract audio from a single video file."""
        try:
            video_path = Path(video_file)
            output_path = Path(output_file)
            
            if not video_path.exists():
                logger.error(f"Video file not found: {video_file}")
                return False
            
            # Create output directory
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Extract audio
            result = await self._extract_audio_from_video(video_path, output_path.parent)
            
            if result:
                # Rename to desired output filename if different
                if result != output_path:
                    result.rename(output_path)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Single file extraction failed: {e}")
            return False 