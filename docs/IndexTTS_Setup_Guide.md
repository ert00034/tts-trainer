# IndexTTS Setup Guide

This guide will help you set up IndexTTS for English voice cloning in the TTS Trainer project.

## Quick Setup (Recommended)

The easiest way to get started is using our automated setup script:

```bash
# Copy and run this in your WSL terminal:
cd ~/code/tts-trainer
python setup_indextts.py
```

This script will:
1. ✅ Check prerequisites (git, huggingface-cli)
2. 📥 Clone the IndexTTS repository
3. 🔧 Install the IndexTTS package
4. 📦 Download IndexTTS-1.5 models (~5GB)
5. 🧪 Test the installation

## Manual Setup (Alternative)

If you prefer to set up manually:

### 1. Install Dependencies
```bash
# Copy and run in your WSL terminal:
cd ~/code/tts-trainer
pip install huggingface_hub[cli]
```

### 2. Clone and Install IndexTTS
```bash
# Clone the repository
git clone https://github.com/index-tts/index-tts.git checkpoints/index-tts

# Install IndexTTS
pip install -e checkpoints/index-tts
```

### 3. Download Models
```bash
# Download IndexTTS-1.5 models
huggingface-cli download IndexTeam/IndexTTS-1.5 \
  config.yaml bigvgan_discriminator.pth bigvgan_generator.pth \
  bpe.model dvae.pth gpt.pth unigram_12000.vocab \
  --local-dir checkpoints/IndexTTS-1.5
```

## Testing Your Setup

Once setup is complete, test with your meowth reference:

```bash
# Copy and run in your WSL terminal:
cd ~/code/tts-trainer
python test_indextts.py
```

This will test voice cloning using your `temp/concatenated_references/meowth_concatenated_reference.wav` file.

## Using IndexTTS

### Basic Voice Cloning

```bash
# Using the main CLI
python main.py inference \
  --model indextts \
  --text "That's right! Team Rocket's here to steal your Pokemon!" \
  --reference temp/concatenated_references/meowth_concatenated_reference.wav \
  --output meowth_test.wav
```

### Character Voice Profiles

Create voice profiles for multiple characters:

```bash
# Train character profiles from your dataset
python main.py train \
  --model indextts \
  --dataset resources/validation_samples_v4 \
  --output artifacts/indextts_voices
```

### Python API Usage

```python
import asyncio
from src.models.indextts.indextts_trainer import IndexTTSTrainer

# Initialize trainer
trainer = IndexTTSTrainer()

# Synthesize speech
result = await trainer.synthesize(
    text="Hello world!",
    reference_audio="path/to/reference.wav",
    output_path="output.wav"
)

if result.success:
    print(f"Generated: {result.output_path}")
    print(f"Duration: {result.audio_duration:.2f}s")
```

## Performance Comparison

| Feature | XTTS v2 | IndexTTS | Improvement |
|---------|---------|----------|-------------|
| **Word Error Rate** | 3.0% | 1.2% | 60% better |
| **Speaker Similarity** | 0.573-0.761 | 0.741-0.823 | 10-30% better |
| **Model Size** | ~650MB | ~5GB | Larger but higher quality |
| **Speed** | Real-time | Real-time+ | Comparable or better |

## Troubleshooting

### Common Issues

**1. Import Error: `No module named 'indextts'`**
```bash
# Reinstall IndexTTS
pip install -e checkpoints/index-tts
```

**2. Model Files Not Found**
```bash
# Re-download models
huggingface-cli download IndexTeam/IndexTTS-1.5 --local-dir checkpoints/IndexTTS-1.5
```

**3. CUDA Out of Memory**
```bash
# Use CPU mode (slower but works)
# Edit config/models/indextts.yaml:
# device: "cpu"
```

**4. Git Clone Fails**
```bash
# Use HTTPS instead of SSH
git clone https://github.com/index-tts/index-tts.git checkpoints/index-tts
```

### Getting Help

If you encounter issues:

1. 🐛 Check the error messages carefully
2. 📋 Ensure all dependencies are installed
3. 🔍 Verify model files exist in `checkpoints/IndexTTS-1.5/`
4. 💾 Make sure you have enough disk space (~10GB total)
5. 🔧 Try running the setup script again

## Directory Structure

After setup, your project should have:

```
tts-trainer/
├── checkpoints/
│   ├── IndexTTS-1.5/          # Model files (~5GB)
│   │   ├── config.yaml
│   │   ├── gpt.pth
│   │   ├── dvae.pth
│   │   └── ...
│   └── index-tts/             # Source code
├── src/models/indextts/        # Our integration
├── config/models/indextts.yaml # Configuration
├── setup_indextts.py          # Setup script
└── test_indextts.py           # Test script
```

## Next Steps

1. ✅ Complete setup using this guide
2. 🧪 Run the test script to verify functionality
3. 🎭 Create voice profiles for your characters
4. 🎙️ Start generating high-quality speech!

---

*For more information, see the [IndexTTS Migration Plan](IndexTTS_Migration_Plan.md) document.* 