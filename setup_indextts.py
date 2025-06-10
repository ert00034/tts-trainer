#!/usr/bin/env python3
"""
IndexTTS Setup Script
Installs IndexTTS and downloads required models
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description="", check=True):
    """Run a command and handle output."""
    print(f"🔄 {description}")
    print(f"   Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
        print(f"   ✅ Success")
        return result
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed with error: {e}")
        if e.stderr:
            print(f"   Error details: {e.stderr.strip()}")
        if check:
            raise
        return e

def check_git():
    """Check if git is available."""
    try:
        subprocess.run(["git", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def check_huggingface_cli():
    """Check if huggingface-cli is available."""
    try:
        subprocess.run(["huggingface-cli", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def install_huggingface_hub():
    """Install huggingface_hub if not available."""
    print("📦 Installing huggingface_hub...")
    run_command([sys.executable, "-m", "pip", "install", "huggingface_hub[cli]"])

def main():
    """Main setup function."""
    
    print("🚀 IndexTTS Setup Script")
    print("=" * 50)
    
    # Check prerequisites
    print("\n📋 Checking prerequisites...")
    
    # Check git
    if not check_git():
        print("❌ Git is required but not found. Please install git first.")
        return 1
    print("✅ Git is available")
    
    # Check/install huggingface-cli
    if not check_huggingface_cli():
        print("🔧 huggingface-cli not found, installing...")
        install_huggingface_hub()
    else:
        print("✅ huggingface-cli is available")
    
    # Setup directories
    print("\n📁 Setting up directories...")
    checkpoints_dir = Path("checkpoints")
    indextts_dir = checkpoints_dir / "IndexTTS-1.5"
    repo_dir = checkpoints_dir / "index-tts"
    
    checkpoints_dir.mkdir(exist_ok=True)
    print(f"✅ Created {checkpoints_dir}")
    
    # Clone IndexTTS repository
    print("\n📥 Setting up IndexTTS repository...")
    if not repo_dir.exists():
        run_command([
            "git", "clone", 
            "https://github.com/index-tts/index-tts.git",
            str(repo_dir)
        ], "Cloning IndexTTS repository")
    else:
        print(f"✅ Repository already exists at {repo_dir}")
    
    # Install IndexTTS package
    print("\n🔧 Installing IndexTTS package...")
    run_command([
        sys.executable, "-m", "pip", "install", "-e", str(repo_dir)
    ], "Installing IndexTTS package")
    
    # Download models
    print("\n📦 Downloading IndexTTS models...")
    if not (indextts_dir / "gpt.pth").exists():
        indextts_dir.mkdir(exist_ok=True)
        
        model_files = [
            "config.yaml",
            "bigvgan_discriminator.pth", 
            "bigvgan_generator.pth",
            "bpe.model", 
            "dvae.pth", 
            "gpt.pth", 
            "unigram_12000.vocab"
        ]
        
        cmd = [
            "huggingface-cli", "download", "IndexTeam/IndexTTS-1.5",
            *model_files,
            "--local-dir", str(indextts_dir)
        ]
        
        run_command(cmd, "Downloading IndexTTS-1.5 models")
    else:
        print(f"✅ Models already exist at {indextts_dir}")
    
    # Test installation
    print("\n🧪 Testing IndexTTS installation...")
    try:
        import indextts
        from indextts.infer import IndexTTS
        print("✅ IndexTTS import successful")
    except ImportError as e:
        print(f"❌ IndexTTS import failed: {e}")
        return 1
    
    print("\n🎉 IndexTTS setup completed successfully!")
    print("\n📋 Next steps:")
    print("   1. Run: python test_indextts.py")
    print("   2. Or use: python main.py inference --model indextts --text 'Hello world' --reference path/to/reference.wav")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 