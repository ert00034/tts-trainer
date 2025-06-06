"""
File management utilities for the TTS Trainer project
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional, Generator
import logging


logger = logging.getLogger(__name__)


def validate_input_path(path: str) -> bool:
    """
    Validate that an input path exists and is accessible.
    
    Args:
        path: Path to validate
        
    Returns:
        True if path is valid, False otherwise
    """
    path_obj = Path(path)
    if not path_obj.exists():
        logger.error(f"Path does not exist: {path}")
        return False
    
    if not os.access(path, os.R_OK):
        logger.error(f"Path is not readable: {path}")
        return False
    
    return True


def create_output_dir(path: str, exist_ok: bool = True) -> bool:
    """
    Create output directory if it doesn't exist.
    
    Args:
        path: Directory path to create
        exist_ok: Don't raise error if directory already exists
        
    Returns:
        True if directory was created or already exists, False on error
    """
    try:
        Path(path).mkdir(parents=True, exist_ok=exist_ok)
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {path}: {e}")
        return False


def find_files_by_extension(directory: str, extensions: List[str], 
                           recursive: bool = True) -> List[Path]:
    """
    Find all files with specified extensions in a directory.
    
    Args:
        directory: Directory to search
        extensions: List of file extensions (without dots)
        recursive: Search subdirectories recursively
        
    Returns:
        List of Path objects for matching files
    """
    directory = Path(directory)
    if not directory.exists():
        return []
    
    files = []
    extensions = [ext.lower().lstrip('.') for ext in extensions]
    
    search_pattern = "**/*" if recursive else "*"
    
    for file_path in directory.glob(search_pattern):
        if file_path.is_file() and file_path.suffix.lower().lstrip('.') in extensions:
            files.append(file_path)
    
    return sorted(files)


def get_video_files(directory: str) -> List[Path]:
    """Find all video files in a directory."""
    video_extensions = ['mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm', 'm4v']
    return find_files_by_extension(directory, video_extensions)


def get_audio_files(directory: str) -> List[Path]:
    """Find all audio files in a directory."""
    audio_extensions = ['wav', 'mp3', 'flac', 'ogg', 'aac', 'm4a', 'wma']
    return find_files_by_extension(directory, audio_extensions)


def get_file_size_mb(file_path: str) -> float:
    """Get file size in megabytes."""
    try:
        return Path(file_path).stat().st_size / (1024 * 1024)
    except Exception:
        return 0.0


def clean_filename(filename: str) -> str:
    """
    Clean filename by removing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Cleaned filename safe for filesystem
    """
    # Remove invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove multiple underscores
    while '__' in filename:
        filename = filename.replace('__', '_')
    
    # Trim and ensure not empty
    filename = filename.strip('_').strip()
    if not filename:
        filename = 'unnamed'
    
    return filename


def ensure_unique_filename(file_path: Path) -> Path:
    """
    Ensure filename is unique by adding a number suffix if needed.
    
    Args:
        file_path: Path object for the desired file
        
    Returns:
        Path object with unique filename
    """
    if not file_path.exists():
        return file_path
    
    base = file_path.stem
    suffix = file_path.suffix
    parent = file_path.parent
    
    counter = 1
    while True:
        new_name = f"{base}_{counter}{suffix}"
        new_path = parent / new_name
        if not new_path.exists():
            return new_path
        counter += 1


def copy_file_with_progress(src: str, dst: str, chunk_size: int = 1024*1024) -> bool:
    """
    Copy file with progress logging.
    
    Args:
        src: Source file path
        dst: Destination file path
        chunk_size: Size of chunks to copy at once
        
    Returns:
        True if copy successful, False otherwise
    """
    try:
        src_path = Path(src)
        dst_path = Path(dst)
        
        # Create destination directory if needed
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        
        total_size = src_path.stat().st_size
        copied = 0
        
        with open(src_path, 'rb') as src_file, open(dst_path, 'wb') as dst_file:
            while True:
                chunk = src_file.read(chunk_size)
                if not chunk:
                    break
                dst_file.write(chunk)
                copied += len(chunk)
                
                if total_size > 0:
                    progress = (copied / total_size) * 100
                    if copied % (chunk_size * 10) == 0:  # Log every 10 chunks
                        logger.debug(f"Copying {src_path.name}: {progress:.1f}%")
        
        logger.info(f"Copied {src_path.name} to {dst_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to copy {src} to {dst}: {e}")
        return False


def delete_file_safe(file_path: str) -> bool:
    """
    Safely delete a file with error handling.
    
    Args:
        file_path: Path to file to delete
        
    Returns:
        True if deleted successfully, False otherwise
    """
    try:
        Path(file_path).unlink()
        logger.debug(f"Deleted file: {file_path}")
        return True
    except FileNotFoundError:
        logger.debug(f"File not found (already deleted?): {file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to delete {file_path}: {e}")
        return False


def cleanup_directory(directory: str, pattern: str = "*", dry_run: bool = False) -> int:
    """
    Clean up files in a directory matching a pattern.
    
    Args:
        directory: Directory to clean
        pattern: Glob pattern for files to delete
        dry_run: If True, only log what would be deleted
        
    Returns:
        Number of files deleted (or would be deleted in dry run)
    """
    directory = Path(directory)
    if not directory.exists():
        return 0
    
    files_to_delete = list(directory.glob(pattern))
    
    if dry_run:
        logger.info(f"Would delete {len(files_to_delete)} files matching '{pattern}' in {directory}")
        for file_path in files_to_delete:
            logger.debug(f"Would delete: {file_path}")
        return len(files_to_delete)
    
    deleted_count = 0
    for file_path in files_to_delete:
        if file_path.is_file() and delete_file_safe(str(file_path)):
            deleted_count += 1
    
    logger.info(f"Deleted {deleted_count} files from {directory}")
    return deleted_count


def get_directory_size(directory: str) -> float:
    """
    Calculate total size of directory in MB.
    
    Args:
        directory: Directory path
        
    Returns:
        Size in megabytes
    """
    total_size = 0
    directory = Path(directory)
    
    if not directory.exists():
        return 0.0
    
    for file_path in directory.rglob('*'):
        if file_path.is_file():
            total_size += file_path.stat().st_size
    
    return total_size / (1024 * 1024)


def iter_files_with_progress(directory: str, extensions: Optional[List[str]] = None) -> Generator[Path, None, None]:
    """
    Iterate over files in directory with progress logging.
    
    Args:
        directory: Directory to iterate
        extensions: File extensions to filter by
        
    Yields:
        Path objects for each file
    """
    if extensions:
        files = find_files_by_extension(directory, extensions)
    else:
        files = [f for f in Path(directory).rglob('*') if f.is_file()]
    
    total_files = len(files)
    logger.info(f"Processing {total_files} files in {directory}")
    
    for i, file_path in enumerate(files, 1):
        if i % 10 == 0 or i == total_files:
            progress = (i / total_files) * 100
            logger.debug(f"Processing file {i}/{total_files} ({progress:.1f}%): {file_path.name}")
        
        yield file_path 