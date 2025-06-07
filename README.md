# TTS Trainer - Pipeline-Orchestrated Architecture

A comprehensive toolkit for converting video files into training data and fine-tuning Text-to-Speech models with real-time Discord bot integration.

## 🎯 Features

- **Video-to-TTS Pipeline**: Automated conversion from video files to TTS training datasets
- **Multiple Model Support**: XTTS v2, VITS, and Tortoise-TTS with unified interface
- **Audio Processing**: Professional-grade audio preprocessing with denoising and normalization
- **Real-time Transcription**: Faster-Whisper integration with speaker diarization
- **Discord Bot**: Stream TTS output directly to Discord voice channels
- **Experiment Tracking**: Built-in metrics and checkpointing system

## 🏗️ Architecture

This project follows a **Pipeline-Orchestrated Architecture** with clear data flow stages:

```
Video Files → Audio Extraction → Preprocessing → Transcription → Dataset Building → Model Training → Deployment
```

### Key Components

- **Pipeline Orchestrator**: Manages the entire workflow from video to trained model
- **Stage-based Processing**: Each step is isolated and can be run independently
- **Validation Checkpoints**: Quality assurance at each stage
- **Model Adapters**: Unified interface for different TTS architectures

## 📁 Project Structure

```
tts-trainer/
├── config/                 # Configuration files
│   ├── models/            # Model-specific configs (XTTS, VITS, Tortoise)
│   ├── audio/             # Audio processing settings
│   └── training/          # Training hyperparameters
├── docs/                   # Documentation
│   ├── implementation_roadmap.md  # Feature implementation plan
│   └── deep_research_plan.md      # Research and architecture notes
├── resources/             # Data storage
│   ├── videos/           # Input video files
│   ├── audio/            # Extracted/processed audio
│   ├── transcripts/      # Generated transcriptions
│   └── datasets/         # Final training datasets
├── src/                   # Source code
│   ├── pipeline/         # Pipeline orchestration
│   │   ├── stages/       # Individual processing stages
│   │   └── validators/   # Quality validation
│   ├── models/           # Model implementations
│   │   ├── xtts/        # XTTS v2 implementation
│   │   └── vits/        # VITS implementation
│   ├── utils/           # Utility functions
│   └── discord_bot/     # Discord integration
├── notebooks/           # Jupyter notebooks for analysis
├── tests/              # Test suite
├── artifacts/          # Training outputs
│   ├── models/        # Trained models
│   ├── checkpoints/   # Training checkpoints
│   └── metrics/       # Training metrics
└── main.py            # CLI entry point
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.10-3.12**
- **CUDA 12.6+ with compatible drivers** (for GPU acceleration)
- **FFmpeg** for audio processing
- **Git LFS** for model storage (optional)

### Installation

#### Option 1: Automated CUDA Setup (Recommended)

```bash
# Clone the repository
git clone <your-repo-url>
cd tts-trainer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
.\venv\Scripts\Activate.ps1  # Windows PowerShell

# Run automated CUDA setup
python setup_cuda.py
```

#### Option 2: Manual Installation

```bash
# Install CUDA-enabled PyTorch (CUDA 12.8/12.9 compatible)
pip install torch==2.7.0+cu128 torchaudio==2.7.0+cu128 torchvision==0.22.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# Install requirements
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu128

# Verify CUDA setup
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### Option 3: CPU-Only Installation

```bash
# For CPU-only (slower processing)
pip install torch torchaudio torchvision
pip install -r requirements.txt
```

### Speaker Diarization Setup

For speaker identification features, you need HuggingFace authentication:

```bash
# Get token from https://huggingface.co/settings/tokens
export HUGGINGFACE_TOKEN=your_token_here

# Accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1
# Or login via CLI
huggingface-cli login
```

## 📋 Usage

### Basic Pipeline

```bash
# Extract audio from videos
python main.py extract-audio --input resources/videos/ --output resources/audio/

# Transcribe with speaker identification (CUDA recommended)
python main.py transcribe --input resources/audio/ --speaker-diarization --output resources/transcripts/

# Extract speaker-specific audio segments  
python main.py segment-speakers --input resources/transcripts/ --output resources/datasets/

# Preprocess audio for training
python main.py preprocess-audio --input resources/datasets/ --output resources/datasets/processed/

# Train TTS model
python main.py train --model xtts_v2 --dataset resources/datasets/processed/
```

### Performance Comparison

| Configuration | 22-min Episode | Real-time Factor | Notes |
|---------------|----------------|------------------|--------|
| **CPU Only** | ~57 minutes | 2.5x slower | Basic compatibility |
| **CUDA Full** | ~15 minutes | 0.7x faster | **Recommended** |
| **Hybrid CPU/CUDA** | ~25 minutes | 1.1x slower | Fallback option |

### Advanced Features

```bash
# Analyze speaker distribution
python main.py analyze-speakers --input resources/transcripts/

# Run complete pipeline
python main.py run-pipeline --input resources/videos/ --output artifacts/models/

# Start Discord bot
python main.py discord-bot --token YOUR_DISCORD_TOKEN
```

## �� Configuration

### Model Configuration

Edit `config/models/xtts_v2.yaml` for XTTS v2 settings:

```yaml
model:
  name: "xtts_v2"
  device: "cuda"
  precision: "fp16"
  
streaming:
  chunk_size: 2048
  overlap: 256
  
training:
  batch_size: 8
  learning_rate: 1e-4
  epochs: 10
```

### Audio Processing

Configure audio preprocessing in `config/audio/preprocessing.yaml`:

```yaml
input:
  sample_rate: 48000
  format: "wav"
  
preprocessing:
  denoise: true
  normalize_lufs: -23
  trim_silence: true
  
output:
  sample_rate: 24000
  bit_depth: 16
```

### Speaker Diarization Settings

Edit `config/audio/speaker_diarization.yaml`:

```yaml
device: "cuda"  # Use "cpu" if CUDA issues
clustering:
  threshold: 0.45      # Lower = more speakers detected
  min_speakers: 2      # Minimum speakers
  max_speakers: 30     # Maximum speakers (for Pokemon episodes)
batch_size: 4          # GPU batch size
```

## 🎮 Discord Bot Integration

The Discord bot provides real-time TTS streaming:

### Features
- Voice cloning from 6+ seconds of audio
- Real-time speech-to-text transcription
- Speaker identification and separation
- Streaming audio output (200ms latency)

### Commands
- `!say <text>` - Speak text in the voice channel
- `!clone <audio_file>` - Clone voice from audio file
- `!listen` - Start transcribing voice channel
- `!stop` - Stop current operations

## 📊 Supported Models

| Model | Training Time | Quality | Real-time Capable | Use Case |
|-------|---------------|---------|-------------------|----------|
| **XTTS v2** | No training needed | High | ✅ | Quick voice cloning |
| **VITS** | 30 min on 4090 | Very High | ✅ | Fine-tuned quality |
| **Tortoise-TTS** | 2-4 hours | Ultra High | ❌ | Studio-quality offline |

## 🔬 Development

### Running Tests
```bash
# Run all tests
pytest tests/

# Run specific test category
pytest tests/unit/
pytest tests/integration/
```

### Adding New Models
1. Create model adapter in `src/models/{model_name}/`
2. Implement the base trainer interface
3. Add configuration in `config/models/{model_name}.yaml`
4. Register in the pipeline orchestrator

### Pipeline Stages
Each stage implements the `PipelineStage` interface:
- `validate_input()` - Check input requirements
- `process()` - Execute the stage logic
- `validate_output()` - Verify output quality

## 📈 Performance Benchmarks

**RTX 4090 Performance:**
- XTTS v2 Inference: 200ms time-to-first-chunk
- Faster-Whisper Transcription: Real-time with VAD
- VITS Training: 30 minutes for 10-minute dataset
- Peak VRAM Usage: ~7GB (STT + TTS simultaneous)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- [Coqui TTS](https://github.com/coqui-ai/TTS) for XTTS v2 and VITS
- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) for transcription
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) for speaker diarization
- [discord.py](https://github.com/Rapptz/discord.py) for Discord integration 

## 🛠️ Troubleshooting

### CUDA Issues

```bash
# Check CUDA compatibility
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Version: {torch.version.cuda}')"

# Common cuDNN fix - reinstall PyTorch
pip uninstall torch torchaudio torchvision
python setup_cuda.py
```

### Speaker Diarization Problems

```bash
# Check HuggingFace authentication
huggingface-cli whoami

# Test with CPU fallback
# Edit config/audio/speaker_diarization.yaml: device: "cpu"
```

### Audio Quality Issues

```bash
# Validate audio quality
python main.py preprocess-audio --input resources/audio/ --validate-only

# Check supported formats
python -c "import librosa; print('LibROSA OK')"
``` 