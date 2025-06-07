#!/usr/bin/env python3
"""
CUDA Setup Script for TTS Trainer
Installs CUDA-enabled PyTorch with cuDNN for CUDA 12.8/12.9 compatibility
Automatically configures library paths to prevent cuDNN loading issues
"""

import subprocess
import sys
import platform
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout.strip():
            # Only show first few lines to avoid spam
            lines = result.stdout.strip().split('\n')[:3]
            for line in lines:
                if line.strip():
                    print(f"   {line.strip()}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Command: {cmd}")
        if e.stderr:
            print(f"   Error: {e.stderr.strip()}")
        return None

def check_cuda():
    """Check if CUDA is available."""
    print("🔍 Checking CUDA installation...")
    
    # Check nvcc
    nvcc_output = run_command("nvcc --version", "Checking nvcc")
    if nvcc_output:
        for line in nvcc_output.split('\n'):
            if 'release' in line.lower():
                print(f"   Found: {line.strip()}")
    
    # Check nvidia-smi
    smi_output = run_command("nvidia-smi", "Checking nvidia-smi")
    if smi_output:
        lines = smi_output.split('\n')
        for line in lines[:5]:  # First few lines contain version info
            if 'Driver Version' in line or 'CUDA Version' in line:
                print(f"   {line.strip()}")

def install_cudnn():
    """Install cuDNN which is required for neural network operations."""
    print("\n🧠 Installing cuDNN...")
    
    # First try conda (most reliable)
    conda_result = run_command("which conda", "Checking for conda")
    if conda_result:
        print("   Found conda, using conda-forge cuDNN...")
        if run_command("conda install -c conda-forge cudnn -y", "Installing cuDNN via conda"):
            return True
    
    # Fallback to pip
    print("   Using pip for cuDNN...")
    cudnn_cmd = "pip install nvidia-cudnn-cu12"
    return run_command(cudnn_cmd, "Installing cuDNN via pip") is not None

def install_cuda_pytorch():
    """Install CUDA-enabled PyTorch."""
    print("\n🚀 Installing CUDA-enabled PyTorch...")
    
    # Uninstall existing PyTorch
    print("🧹 Removing existing PyTorch installations...")
    run_command("pip uninstall torch torchaudio torchvision -y", "Uninstalling existing PyTorch")
    
    # Install CUDA PyTorch with specific versions known to work
    pytorch_cmd = (
        "pip install torch==2.7.0+cu128 torchaudio==2.7.0+cu128 torchvision==0.22.0+cu128 "
        "--index-url https://download.pytorch.org/whl/cu128"
    )
    
    if not run_command(pytorch_cmd, "Installing CUDA PyTorch"):
        print("❌ Failed to install CUDA PyTorch")
        return False
    
    return True

def install_requirements():
    """Install remaining requirements."""
    print("\n📦 Installing remaining requirements...")
    
    # Install other requirements (excluding PyTorch)
    cmd = "pip install -r requirements.txt"
    return run_command(cmd, "Installing requirements") is not None

def configure_venv_library_paths():
    """Configure virtual environment to automatically set cuDNN library paths."""
    print("\n🔧 Configuring virtual environment library paths...")
    
    # Path to the activate script
    activate_script = Path("venv/bin/activate")
    
    if not activate_script.exists():
        print("❌ Virtual environment activate script not found")
        return False
    
    # The line we want to add
    library_path_line = '''
# TTS Trainer: Add cuDNN library paths to prevent loading issues
if [ -n "$VIRTUAL_ENV" ]; then
    export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cudnn/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH:-}"
fi
'''
    
    try:
        # Read the current activate script
        with open(activate_script, 'r') as f:
            content = f.read()
        
        # Check if our fix is already there
        if "TTS Trainer: Add cuDNN library paths" in content:
            print("✅ Library path configuration already present")
            return True
        
        # Add our configuration before the final section
        # Find a good insertion point (usually before the end)
        if "# This should detect bash and zsh, which have a hash command that must" in content:
            # Insert before the hash detection section
            insertion_point = content.find("# This should detect bash and zsh, which have a hash command that must")
            new_content = content[:insertion_point] + library_path_line + "\n" + content[insertion_point:]
        else:
            # Fallback: append at the end
            new_content = content + library_path_line
        
        # Write the updated activate script
        with open(activate_script, 'w') as f:
            f.write(new_content)
        
        print("✅ Added cuDNN library path configuration to virtual environment")
        print("   ⚠️  You may need to reactivate your virtual environment for changes to take effect")
        return True
        
    except Exception as e:
        print(f"❌ Failed to configure library paths: {e}")
        return False

def test_installation():
    """Test the CUDA installation."""
    print("\n🧪 Testing CUDA installation...")
    
    # Write test script to a temporary file to avoid shell quoting issues
    test_script_content = '''import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA devices: {torch.cuda.device_count()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    
    # Test memory allocation
    try:
        x = torch.randn(1000, 1000).cuda()
        print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
        print("✅ CUDA memory allocation successful!")
    except Exception as e:
        print(f"❌ CUDA memory allocation failed: {e}")
        
    # Test cuDNN
    try:
        import torch.backends.cudnn as cudnn
        print(f"cuDNN version: {cudnn.version()}")
        print("✅ cuDNN is available!")
    except Exception as e:
        print(f"❌ cuDNN test failed: {e}")
else:
    print("❌ CUDA not available - check your installation")
'''
    
    # Write to temporary file
    with open('/tmp/cuda_test.py', 'w') as f:
        f.write(test_script_content)
    
    result = run_command("python /tmp/cuda_test.py", "Testing CUDA setup")
    
    # Clean up
    try:
        os.remove('/tmp/cuda_test.py')
    except:
        pass
    
    return result is not None

def test_library_path_fix():
    """Test that the library path fix works by simulating the transcription environment."""
    print("\n🧪 Testing cuDNN library path fix...")
    
    test_script_content = '''
import os
import sys

# Show current library path
print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")

# Test if we can import cuDNN-dependent libraries
try:
    import torch
    print(f"✅ PyTorch imported: {torch.__version__}")
    
    # Test cuDNN
    import torch.backends.cudnn as cudnn
    print(f"✅ cuDNN accessible: {cudnn.version()}")
    
    # Test if CUDA operations work
    if torch.cuda.is_available():
        x = torch.randn(100, 100).cuda()
        print("✅ CUDA operations working")
    
    print("🎉 Library path fix validation successful!")
    
except Exception as e:
    print(f"❌ Library path fix validation failed: {e}")
    sys.exit(1)
'''
    
    with open('/tmp/test_lib_path.py', 'w') as f:
        f.write(test_script_content)
    
    # Test with the library path that should be set by the activate script
    venv_path = os.path.abspath("venv")
    lib_path = f"{venv_path}/lib/python3.12/site-packages/nvidia/cudnn/lib:{venv_path}/lib/python3.12/site-packages/nvidia/cublas/lib"
    
    test_cmd = f'LD_LIBRARY_PATH="{lib_path}:${{LD_LIBRARY_PATH:-}}" python /tmp/test_lib_path.py'
    result = run_command(test_cmd, "Testing library path fix")
    
    # Clean up
    try:
        os.remove('/tmp/test_lib_path.py')
    except:
        pass
    
    return result is not None

def main():
    """Main setup function."""
    print("🎯 TTS Trainer CUDA Setup with cuDNN and Library Path Fix")
    print("=" * 60)
    
    # Check platform
    if platform.system() != "Linux":
        print(f"⚠️  Warning: This script is designed for Linux. You're running {platform.system()}")
    
    # Check CUDA
    check_cuda()
    
    # Install cuDNN first
    if not install_cudnn():
        print("\n⚠️  cuDNN installation failed, but continuing with PyTorch...")
    
    # Install PyTorch
    if not install_cuda_pytorch():
        print("\n❌ Setup failed during PyTorch installation")
        sys.exit(1)
    
    # Install other requirements
    if not install_requirements():
        print("\n❌ Setup failed during requirements installation")
        sys.exit(1)
    
    # Configure virtual environment library paths
    if not configure_venv_library_paths():
        print("\n⚠️  Failed to configure library paths automatically")
        print("   You may need to run the manual fix later")
    
    # Test installation
    if test_installation():
        print("\n🧪 Testing library path fix...")
        if test_library_path_fix():
            print("\n🎉 CUDA setup with library path fix completed successfully!")
            print("\n🚀 You can now run:")
            print("   # Reactivate your virtual environment first:")
            print("   deactivate && source venv/bin/activate")
            print("   # Then run transcription:")
            print("   python main.py transcribe --input test_clips/ --speaker-diarization")
        else:
            print("\n⚠️  CUDA works but library path fix needs manual configuration")
            print("   Run this command manually:")
            print('   echo \'export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cudnn/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH:-}"\' >> venv/bin/activate')
    else:
        print("\n❌ Setup completed but CUDA test failed")
        print("   Try the manual cuDNN installation steps below")
        print("\n📝 Manual cuDNN installation:")
        print("   conda install -c conda-forge cudnn")
        print("   # OR")
        print("   pip install nvidia-cudnn-cu12")

if __name__ == "__main__":
    main() 