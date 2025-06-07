#!/usr/bin/env python3
"""
Archive Transcripts Script
Archives current transcription data to a timestamped zip file and cleans the folder
"""

import os
import shutil
import zipfile
from pathlib import Path
from datetime import datetime
import argparse
import sys

def get_timestamp():
    """Get current timestamp for filename."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_archive_name(custom_name=None):
    """Generate archive filename."""
    timestamp = get_timestamp()
    if custom_name:
        # Clean the custom name for filename safety
        safe_name = "".join(c for c in custom_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_name = safe_name.replace(' ', '_')
        return f"transcripts_{timestamp}_{safe_name}.zip"
    else:
        return f"transcripts_{timestamp}.zip"

def count_files_in_directory(directory):
    """Count total files in directory recursively."""
    if not directory.exists():
        return 0
    
    count = 0
    for root, dirs, files in os.walk(directory):
        count += len(files)
    return count

def create_archive(source_dir, archive_path):
    """Create zip archive of the source directory."""
    print(f"📦 Creating archive: {archive_path}")
    
    file_count = 0
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                file_path = Path(root) / file
                # Store relative path in zip
                arcname = file_path.relative_to(source_dir)
                zipf.write(file_path, arcname)
                file_count += 1
                
                # Show progress for large archives
                if file_count % 50 == 0:
                    print(f"   Added {file_count} files...")
    
    # Get archive size
    archive_size = Path(archive_path).stat().st_size / (1024 * 1024)  # MB
    print(f"✅ Archive created: {archive_path}")
    print(f"   📊 {file_count} files, {archive_size:.1f} MB")
    
    return file_count, archive_size

def clean_directory(directory):
    """Remove all contents of directory but keep the directory itself."""
    print(f"🧹 Cleaning directory: {directory}")
    
    if not directory.exists():
        print(f"   Directory doesn't exist, creating: {directory}")
        directory.mkdir(parents=True, exist_ok=True)
        return
    
    removed_count = 0
    for item in directory.iterdir():
        if item.is_file():
            item.unlink()
            removed_count += 1
        elif item.is_dir():
            shutil.rmtree(item)
            removed_count += 1
    
    print(f"   🗑️  Removed {removed_count} items")

def main():
    parser = argparse.ArgumentParser(description="Archive transcription data and clean folder")
    parser.add_argument(
        "--name", "-n",
        help="Custom name for the archive (will be sanitized for filename)",
        default=None
    )
    parser.add_argument(
        "--output-dir", "-o",
        help="Directory to save archive (default: ./archives/)",
        default="./archives/"
    )
    parser.add_argument(
        "--transcripts-dir", "-t",
        help="Transcripts directory to archive (default: ./resources/transcripts/)",
        default="./resources/transcripts/"
    )
    parser.add_argument(
        "--dry-run", "-d",
        action="store_true",
        help="Show what would be done without actually doing it"
    )
    parser.add_argument(
        "--skip-clean", "-s",
        action="store_true",
        help="Archive only, don't clean the transcripts directory"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    transcripts_dir = Path(args.transcripts_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    print("🎯 TTS Trainer - Archive Transcripts")
    print("=" * 50)
    print(f"📁 Transcripts directory: {transcripts_dir}")
    print(f"📁 Archive output directory: {output_dir}")
    
    # Check if transcripts directory exists and has content
    if not transcripts_dir.exists():
        print(f"❌ Transcripts directory doesn't exist: {transcripts_dir}")
        return 1
    
    file_count = count_files_in_directory(transcripts_dir)
    if file_count == 0:
        print(f"⚠️  No files found in transcripts directory")
        if not args.skip_clean:
            print("   Directory is already clean!")
        return 0
    
    print(f"📊 Found {file_count} files to archive")
    
    # Generate archive name
    archive_name = get_archive_name(args.name)
    archive_path = output_dir / archive_name
    
    if args.dry_run:
        print("\n🔍 DRY RUN - Would perform these actions:")
        print(f"   📦 Create archive: {archive_path}")
        print(f"   📊 Archive {file_count} files")
        if not args.skip_clean:
            print(f"   🧹 Clean directory: {transcripts_dir}")
        print("\nRe-run without --dry-run to perform these actions.")
        return 0
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if archive already exists
    if archive_path.exists():
        print(f"⚠️  Archive already exists: {archive_path}")
        response = input("Overwrite? (y/N): ").strip().lower()
        if response != 'y':
            print("Aborted.")
            return 1
    
    try:
        # Create archive
        archived_files, archive_size = create_archive(transcripts_dir, archive_path)
        
        # Clean directory if requested
        if not args.skip_clean:
            print()
            clean_directory(transcripts_dir)
            print(f"✅ Directory cleaned and ready for fresh transcription")
        
        print(f"\n🎉 Archive operation completed!")
        print(f"   📦 Archive: {archive_path}")
        print(f"   📊 Files: {archived_files} ({archive_size:.1f} MB)")
        
        if not args.skip_clean:
            print(f"   🧹 Transcripts directory cleaned: {transcripts_dir}")
            print(f"\n🚀 Ready to run: python main.py transcribe --input <your_input>")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during archiving: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 