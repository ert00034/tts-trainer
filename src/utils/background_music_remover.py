"""
Background Music Remover for TTS Training Validation Samples

Uses audio-separator (based on UVR models) to separate vocals from background music
in Pokemon episode clips for better TTS training quality.
"""

import logging
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import subprocess
import sys

logger = logging.getLogger(__name__)


class BackgroundMusicRemover:
    """Remove background music from audio files using audio-separator."""
    
    def __init__(self):
        """Initialize the background music remover."""
        self.separator_available = self._check_audio_separator()
        
    def _check_audio_separator(self) -> bool:
        """Check if audio-separator is installed and available."""
        try:
            result = subprocess.run(
                ["audio-separator", "--version"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            if result.returncode == 0:
                logger.info("audio-separator is available")
                return True
            else:
                logger.warning("audio-separator command failed")
                return False
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("audio-separator not found or not working")
            return False
    
    def install_audio_separator(self) -> bool:
        """Install audio-separator using pip."""
        try:
            logger.info("Installing audio-separator...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "audio-separator[cpu]"],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                logger.info("audio-separator installed successfully")
                self.separator_available = self._check_audio_separator()
                return self.separator_available
            else:
                logger.error(f"Failed to install audio-separator: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Installation timed out")
            return False
        except Exception as e:
            logger.error(f"Installation failed: {e}")
            return False
    
    def remove_background_music(
        self, 
        input_file: Path, 
        output_dir: Path,
        model: str = "UVR-MDX-NET-Voc_FT.onnx"
    ) -> Optional[Path]:
        """
        Remove background music from an audio file.
        
        Args:
            input_file: Path to input audio file
            output_dir: Directory to save processed audio
            model: Model to use for separation (vocal models work best for our use case)
        
        Returns:
            Path to the vocals-only file, or None if failed
        """
        if not self.separator_available:
            logger.error("audio-separator not available. Run install_audio_separator() first.")
            return None
        
        try:
            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Run audio-separator to extract vocals (clean speech)
            cmd = [
                "audio-separator",
                str(input_file),
                "--model_filename", model,
                "--output_dir", str(output_dir),
                "--output_format", "wav",
                "--single_stem", "Vocals"  # Only output vocals (speech without music)
            ]
            
            logger.info(f"Removing background music from {input_file.name}...")
            logger.debug(f"Command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per file
            )
            
            if result.returncode == 0:
                # Find the output file
                vocals_file = self._find_vocals_file(output_dir, input_file.stem)
                if vocals_file and vocals_file.exists():
                    logger.info(f"Successfully removed background music: {vocals_file}")
                    return vocals_file
                else:
                    logger.error("Vocals file not found after separation")
                    return None
            else:
                logger.error(f"audio-separator failed: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error(f"Background music removal timed out for {input_file}")
            return None
        except Exception as e:
            logger.error(f"Failed to remove background music from {input_file}: {e}")
            return None
    
    def _find_vocals_file(self, output_dir: Path, stem: str) -> Optional[Path]:
        """Find the vocals output file."""
        # audio-separator typically creates files like: filename_(Vocals)_model.wav
        pattern_candidates = [
            f"{stem}_(Vocals)*.wav",
            f"{stem}_Vocals.wav", 
            f"*{stem}*Vocals*.wav",
            f"*Vocals*.wav"
        ]
        
        for pattern in pattern_candidates:
            matches = list(output_dir.glob(pattern))
            if matches:
                return matches[0]  # Return first match
        
        return None
    
    def process_validation_samples(
        self, 
        manual_refs_file: Path,
        output_suffix: str = "_no_music"
    ) -> Dict[str, str]:
        """
        Process validation samples listed in manual_refs.txt.
        
        Args:
            manual_refs_file: Path to manual_refs.txt
            output_suffix: Suffix to add to processed files
            
        Returns:
            Dictionary mapping original files to processed files
        """
        if not self.separator_available:
            logger.error("audio-separator not available")
            return {}
        
        try:
            # Read manual refs file
            with open(manual_refs_file, 'r') as f:
                lines = f.readlines()
            
            results = {}
            
            for line in lines:
                line = line.strip()
                if not line or ':' not in line:
                    continue
                
                # Parse line: "character:path/to/file.wav"
                character, file_path = line.split(':', 1)
                input_file = Path(file_path.strip())
                
                if not input_file.exists():
                    logger.warning(f"File not found: {input_file}")
                    continue
                
                # Create output directory next to input file
                output_dir = input_file.parent / f"processed{output_suffix}"
                
                # Process the file
                vocals_file = self.remove_background_music(input_file, output_dir)
                
                if vocals_file:
                    # Create a more descriptive filename
                    final_name = f"{input_file.stem}{output_suffix}.wav"
                    final_path = input_file.parent / final_name
                    
                    # Move and rename the file
                    shutil.move(str(vocals_file), str(final_path))
                    
                    results[str(input_file)] = str(final_path)
                    logger.info(f"✅ Processed: {input_file.name} -> {final_path.name}")
                else:
                    logger.error(f"❌ Failed to process: {input_file.name}")
            
            # Clean up temporary directories
            for line in lines:
                line = line.strip()
                if ':' in line:
                    _, file_path = line.split(':', 1)
                    input_file = Path(file_path.strip())
                    temp_dir = input_file.parent / f"processed{output_suffix}"
                    if temp_dir.exists():
                        shutil.rmtree(temp_dir, ignore_errors=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to process validation samples: {e}")
            return {}
    
    def get_available_models(self) -> List[str]:
        """Get list of available models for vocal separation."""
        if not self.separator_available:
            return []
        
        try:
            result = subprocess.run(
                ["audio-separator", "--list_models", "--list_filter=vocals"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                # Parse the output to extract model filenames
                lines = result.stdout.split('\n')
                models = []
                for line in lines:
                    if '.onnx' in line or '.pth' in line or '.ckpt' in line:
                        # Extract just the model filename
                        parts = line.split()
                        if parts:
                            models.append(parts[0])
                return models
            else:
                logger.warning("Failed to list models")
                return []
                
        except Exception as e:
            logger.warning(f"Failed to get available models: {e}")
            return []


def main():
    """Command line interface for background music removal."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Remove background music from validation samples")
    parser.add_argument("--manual-refs", type=Path, default="manual_refs.txt",
                       help="Path to manual_refs.txt file")
    parser.add_argument("--install", action="store_true", 
                       help="Install audio-separator first")
    parser.add_argument("--list-models", action="store_true",
                       help="List available vocal separation models")
    parser.add_argument("--model", default="UVR-MDX-NET-Voc_FT.onnx",
                       help="Model to use for separation")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    remover = BackgroundMusicRemover()
    
    if args.install:
        print("Installing audio-separator...")
        if remover.install_audio_separator():
            print("✅ Installation successful!")
        else:
            print("❌ Installation failed!")
            return
    
    if args.list_models:
        print("Available vocal separation models:")
        models = remover.get_available_models()
        for model in models:
            print(f"  - {model}")
        return
    
    if not remover.separator_available:
        print("❌ audio-separator not available. Run with --install first.")
        return
    
    print(f"Processing validation samples from {args.manual_refs}...")
    results = remover.process_validation_samples(args.manual_refs)
    
    print(f"\nProcessed {len(results)} files:")
    for original, processed in results.items():
        print(f"  {Path(original).name} -> {Path(processed).name}")


if __name__ == "__main__":
    main() 