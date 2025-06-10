# Background Music Removal Guide

## Overview

This guide explains how to remove background music from TTS training validation samples using the new `remove-background-music` command. This is particularly useful for Pokemon episodes and other TV shows that have constant background music.

## Why Remove Background Music?

Background music in training data can negatively impact TTS model quality by:
- Creating artifacts in generated speech
- Reducing vocal clarity and definition
- Interfering with the model's ability to learn clean voice patterns
- Making voice cloning less accurate

## Installation

The background music removal tool uses `audio-separator`, which needs to be installed first:

```bash
# Copy and run this in your WSL terminal:
cd ~/code/tts-trainer
python main.py remove-background-music --install
```

This will install the `audio-separator` library with CPU support. For better performance with a GPU, you can manually install:

```bash
# For GPU acceleration (if you have CUDA):
pip install "audio-separator[gpu]"
```

## Basic Usage

### Remove Background Music from manual_refs.txt Files

```bash
# Copy and run this in your WSL terminal:
cd ~/code/tts-trainer
python main.py remove-background-music
```

This will process all files listed in `manual_refs.txt` and create new versions with `_no_music` suffix.

### Custom Options

```bash
# Use a different model (for better quality):
python main.py remove-background-music --model "UVR-MDX-NET-Voc_FT.onnx"

# Use a different manual refs file:
python main.py remove-background-music --manual-refs "my_custom_refs.txt"

# Change the output suffix:
python main.py remove-background-music --output-suffix "_clean"
```

### List Available Models

```bash
# See what models are available:
python main.py remove-background-music --list-models
```

## Recommended Models

For Pokemon/anime voice separation, these models work well:

1. **UVR-MDX-NET-Voc_FT.onnx** (default) - Good general vocal isolation
2. **vocals_mel_band_roformer.ckpt** - Excellent for anime/cartoon voices
3. **model_bs_roformer_ep_317_sdr_12.9755.ckpt** - High-quality vocal separation

## Expected Results

### Before Processing
- Original: `Pokemon S01E17 Island of Giant Pokemon_SPEAKER_15_006.wav`
- Contains: Meowth's voice + background music + sound effects

### After Processing  
- Processed: `Pokemon S01E17 Island of Giant Pokemon_SPEAKER_15_006_no_music.wav`
- Contains: Primarily Meowth's voice with greatly reduced background music

## Quality Assessment

The tool works by using AI models trained to separate vocals from instrumentals. Results vary by:

- **Voice prominence**: Louder, clearer voices separate better
- **Music complexity**: Simple background music is easier to remove
- **Audio quality**: Higher quality source audio gives better results

## Integration with Training Workflow

### Step 1: Remove Background Music
```bash
cd ~/code/tts-trainer
python main.py remove-background-music
```

### Step 2: Update References (Optional)
The tool will ask if you want to update `manual_refs.txt` to point to the processed files.

### Step 3: Continue Training
Use the processed files for training as you normally would.

## Sample Level vs Episode Level Processing

### Sample Level (Recommended)
- **Pros**: Faster, targeted processing, easier to manage
- **Cons**: May miss some context for separation
- **Best for**: Individual validation samples, character-specific clips

### Episode Level (Alternative)
- **Pros**: Better context for separation algorithms
- **Cons**: Much slower, more complex processing
- **Best for**: Full episode processing before segmentation

**Recommendation**: Use sample-level processing for your current workflow with `manual_refs.txt`.

## File Organization

### Before Processing
```
resources/validation_samples_v4/meowth/
├── Pokemon S01E17 Island of Giant Pokemon_SPEAKER_15_006.wav
├── Pokemon S01E17 Island of Giant Pokemon_SPEAKER_15_007.wav
└── Pokemon S01E79 Fourth Round Rumble_SPEAKER_20_009.wav
```

### After Processing
```
resources/validation_samples_v4/meowth/
├── Pokemon S01E17 Island of Giant Pokemon_SPEAKER_15_006.wav (original)
├── Pokemon S01E17 Island of Giant Pokemon_SPEAKER_15_006_no_music.wav (processed)
├── Pokemon S01E17 Island of Giant Pokemon_SPEAKER_15_007.wav (original)  
├── Pokemon S01E17 Island of Giant Pokemon_SPEAKER_15_007_no_music.wav (processed)
├── Pokemon S01E79 Fourth Round Rumble_SPEAKER_20_009.wav (original)
└── Pokemon S01E79 Fourth Round Rumble_SPEAKER_20_009_no_music.wav (processed)
```

## Troubleshooting

### Model Download Issues
```bash
# Models are downloaded automatically on first use
# If download fails, try a different model:
python main.py remove-background-music --model "UVR-MDX-NET-Inst_HQ_3.onnx"
```

### Installation Issues
```bash
# If installation fails, try installing manually:
pip install audio-separator[cpu]

# Or for GPU support:
pip install audio-separator[gpu]
```

### Quality Issues
- Try different models - some work better for specific types of content
- Consider the vocal prominence in your source material
- Very quiet voices or very loud music may not separate well

## Performance Notes

- Processing time: ~30-60 seconds per minute of audio
- GPU acceleration significantly speeds up processing
- Models are downloaded once and cached locally
- Temporary files are automatically cleaned up

## Python API Usage

You can also use the background music remover programmatically:

```python
from pathlib import Path
from src.utils.background_music_remover import BackgroundMusicRemover

# Initialize remover
remover = BackgroundMusicRemover()

# Install if needed
if not remover.separator_available:
    remover.install_audio_separator()

# Process a single file
input_file = Path("resources/validation_samples_v4/meowth/sample.wav")
output_dir = Path("resources/validation_samples_v4/meowth/processed")
result = remover.remove_background_music(input_file, output_dir)

# Process validation samples
results = remover.process_validation_samples(Path("manual_refs.txt"))
```

## Best Practices

1. **Test First**: Try processing one file before doing a batch
2. **Compare Results**: Listen to before/after to ensure quality
3. **Keep Originals**: Always keep original files as backup
4. **Model Selection**: Try different models if results aren't satisfactory
5. **Manual Review**: Check a few processed files manually before training

## Example Workflow

```bash
# Copy and run these commands in your WSL terminal:
cd ~/code/tts-trainer

# 1. Install audio-separator
python main.py remove-background-music --install

# 2. List available models
python main.py remove-background-music --list-models

# 3. Process your validation samples
python main.py remove-background-music

# 4. Listen to results and compare quality
# (Use your preferred audio player)

# 5. Continue with training using processed files
```

This approach should significantly improve the quality of your TTS training data by removing the constant background music from Pokemon episodes! 