# TTS Training Guide: Character Voice Cloning with XTTS v2

This guide explains how to use the new TTS training system to create character voice profiles and synthesize speech using voice cloning.

## 🎯 Overview

The TTS training system uses **Coqui XTTS v2** for zero-shot voice cloning, allowing you to:

1. **Create character voice profiles** from audio datasets
2. **Synthesize speech** in different character voices
3. **Fine-tune voices** for improved quality (optional)
4. **Integrate with Discord bot** for real-time voice synthesis

## 🚀 Quick Start

### 1. Prerequisites

```bash
# Copy and run in your WSL terminal:
cd ~/code/tts-trainer

# Ensure dependencies are installed
pip install -r requirements.txt

# Verify GPU setup (recommended)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 2. Check Available Datasets

```bash
# Quick check of your character datasets:
python test_tts_training.py --check-only
```

This will show you available character datasets and their statistics.

### 3. Train Character Voice Profiles

```bash
# Create voice profiles for all characters:
python main.py train --model xtts_v2 --dataset resources/validation_samples_v4 --output artifacts/character_voices

# Or use the demo script:
python test_tts_training.py
```

### 4. Test Voice Synthesis

```bash
# Test with a specific character:
python main.py inference --model artifacts/character_voices/character_voice_profiles.json --character ash --text "I choose you, Pikachu!"

# Test with reference audio:
python main.py inference --model xtts_v2 --reference path/to/audio.wav --text "Hello world!"
```

## 📊 Understanding Your Data

### Character Dataset Structure

Your character datasets should be organized as:

```
resources/validation_samples_v4/
├── ash/                     # Character name
│   ├── episode_clip_001.wav # Audio files
│   ├── episode_clip_001.json # Metadata files
│   ├── episode_clip_002.wav
│   └── episode_clip_002.json
├── brock/
├── meowth/
└── ...
```

### Metadata Format

Each `.json` file contains:

```json
{
  "original_file": "source_episode.wav",
  "start": 1238.48,
  "end": 1241.42,
  "duration": 2.94,
  "text": "Character dialogue",
  "speaker": "SPEAKER_15",
  "confidence": -0.20,
  "quality_score": 34.65
}
```

The training system uses these metrics to select the best reference audio for each character.

## 🏋️ Training Process

### What Happens During Training

1. **Model Initialization**: Downloads and loads XTTS v2 model
2. **Character Discovery**: Finds character directories in your dataset
3. **Reference Selection**: Chooses best audio samples based on:
   - Audio quality score
   - Duration (prefers 2-8 seconds)
   - Transcription confidence
4. **Voice Profile Creation**: Tests voice cloning with each character
5. **Profile Storage**: Saves character voice profiles for future use

### Training Output

```
artifacts/character_voices/
├── character_voice_profiles.json    # Main profiles file
├── test_synthesis/                  # Test audio samples
│   ├── test_ash_1.wav
│   ├── test_brock_1.wav
│   └── ...
└── training_logs/                   # Training logs
```

### Expected Results

From your Pokemon episode data, you should get profiles for:
- **Ash**: ~100+ samples (main character)
- **Brock**: ~50+ samples 
- **Meowth**: ~40+ samples
- **Team Rocket (Jessie/James)**: ~30+ samples each
- **Narrator**: ~20+ samples
- **Minor characters**: Variable

## 🎤 Voice Synthesis

### Basic Synthesis

```bash
# Using character profiles:
python main.py inference \
  --model artifacts/character_voices/character_voice_profiles.json \
  --character ash \
  --text "Gotta catch 'em all!" \
  --output ash_catchphrase.wav

# Using custom reference audio:
python main.py inference \
  --model xtts_v2 \
  --reference resources/validation_samples_v4/ash/best_sample.wav \
  --text "This is my voice!" \
  --output custom_voice.wav
```

### Available Characters

After training, check available characters:

```bash
python main.py inference --model artifacts/character_voices/character_voice_profiles.json --text "test"
# Will show: Available characters: ['ash', 'brock', 'meowth', ...]
```

### Voice Quality Tips

**For best results:**
- Use **2-8 second** reference clips
- Choose clips with **clear speech** and **minimal background noise**
- Prefer clips with **higher quality scores** from the metadata
- Test different reference clips if quality isn't satisfactory

## ⚙️ Configuration

### Custom Training Configuration

Create custom configs in `config/training/`:

```yaml
# config/training/my_config.yaml
dataset:
  min_samples: 10              # Require more samples per character
  min_quality_score: 30.0      # Higher quality threshold
  target_characters: ["ash", "brock"]  # Only train specific characters

voice_profiles:
  reference_samples: 5         # Use more reference samples
  min_reference_duration: 3.0  # Longer minimum duration

model:
  temperature: 0.7             # More conservative synthesis
  speed: 1.1                   # Slightly faster speech
```

Use with:

```bash
python main.py train --model xtts_v2 --dataset resources/validation_samples_v4 --config config/training/my_config.yaml --output artifacts/my_voices
```

### Character-Specific Overrides

Override settings for specific characters:

```yaml
character_overrides:
  ash:
    model:
      temperature: 0.7      # More energetic voice
      speed: 1.1           # Faster speech rate
  
  brock:
    model:
      temperature: 0.8      # Calmer voice
      speed: 0.9           # Slower speech rate
```

## 🤖 Discord Bot Integration

### Setup Bot with Character Voices

```bash
# Launch bot with character voice profiles:
python main.py discord-bot --token YOUR_DISCORD_TOKEN --model artifacts/character_voices/character_voice_profiles.json
```

### Bot Commands

In Discord:
- `!say ash Hello everyone!` - Speak as Ash
- `!say brock Time to cook!` - Speak as Brock  
- `!clone @user` - Clone voice from attachment
- `!voices` - List available character voices

## 📈 Advanced Features

### Quality Assessment

Monitor voice profile quality:

```python
# In your Python code:
from models.xtts.xtts_trainer import XTTSTrainer

trainer = XTTSTrainer()
trainer.load_character_voices("artifacts/character_voices/character_voice_profiles.json")

# Test synthesis quality
result = await trainer.synthesize(
    text="Quality test phrase",
    character="ash",
    output_path="quality_test.wav"
)

print(f"Synthesis time: {result.generation_time:.2f}s")
```

### Streaming Synthesis

For real-time applications:

```bash
python main.py inference \
  --model artifacts/character_voices/character_voice_profiles.json \
  --character ash \
  --text "This is streaming synthesis" \
  --streaming
```

### Fine-tuning (Experimental)

For improved character-specific quality:

```yaml
# In config file:
fine_tuning:
  enabled: true
  epochs: 3
  learning_rate: 1e-5
  batch_size: 2
```

**Note**: Fine-tuning requires more GPU memory and training time.

## 🔧 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size or use CPU:
export CUDA_VISIBLE_DEVICES=""  # Force CPU
# Or edit config: model.device: "cpu"
```

**2. Poor Voice Quality**
```bash
# Check reference audio quality:
python test_tts_training.py --check-only

# Try different reference samples:
# Edit the character_voice_profiles.json file manually
```

**3. Character Not Found**
```bash
# List available characters:
python main.py inference --model artifacts/character_voices/character_voice_profiles.json --text "test"

# Check dataset structure:
ls -la resources/validation_samples_v4/
```

**4. Model Download Issues**
```bash
# Clear cache and retry:
rm -rf ~/.cache/huggingface/
python main.py train --model xtts_v2 --dataset resources/validation_samples_v4 --output artifacts/character_voices
```

### Performance Tips

**GPU Memory Optimization:**
- Use `fp16` precision (default)
- Reduce batch size in config
- Close other GPU applications

**Synthesis Speed:**
- Use shorter text passages
- Optimize reference audio length (2-5 seconds)
- Use SSD storage for faster file access

**Quality Improvement:**
- Use higher quality reference audio
- Filter by quality_score > 30
- Combine multiple reference samples

## 📝 Examples

### Example 1: Train Specific Characters

```bash
# Only train main characters:
python main.py train \
  --model xtts_v2 \
  --dataset resources/validation_samples_v4 \
  --config config/training/main_characters.yaml \
  --output artifacts/main_character_voices
```

### Example 2: Test Voice Variety

```bash
# Test same text with different characters:
for character in ash brock meowth; do
  python main.py inference \
    --model artifacts/character_voices/character_voice_profiles.json \
    --character $character \
    --text "Welcome to the world of Pokemon!" \
    --output "test_${character}.wav"
done
```

### Example 3: Batch Synthesis

```python
# Python script for batch synthesis:
import asyncio
from models.xtts.xtts_trainer import XTTSTrainer

async def batch_synthesis():
    trainer = XTTSTrainer()
    trainer.initialize_model()
    trainer.load_character_voices("artifacts/character_voices/character_voice_profiles.json")
    
    phrases = [
        "Hello there!",
        "How are you today?",
        "Let's go on an adventure!"
    ]
    
    characters = trainer.list_available_characters()
    
    for character in characters:
        for i, phrase in enumerate(phrases):
            await trainer.synthesize(
                text=phrase,
                character=character,
                output_path=f"batch_{character}_{i+1}.wav"
            )

# Run:
asyncio.run(batch_synthesis())
```

## 🎯 Next Steps

1. **Test the system** with your character datasets
2. **Experiment** with different configurations
3. **Integrate** with your Discord bot
4. **Fine-tune** for specific use cases
5. **Scale up** to more characters and episodes

For additional support, check the implementation roadmap and research plan documents, or review the test script outputs for debugging information. 