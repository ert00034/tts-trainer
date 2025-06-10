# Voice Training Guide - TTS Trainer

## Overview

This guide documents the complete voice training workflow for character voices, specifically focusing on Pokemon character voices (Meowth, Ash, Brock, etc.). The system provides multiple preprocessing approaches and training methods to achieve high-quality text-to-speech synthesis.

**⚠️ CURRENT STATUS**: Despite extensive troubleshooting and optimization efforts, voice training is still producing raspy/strange voices for characters like Meowth. The documented workflows and fixes improve certain technical aspects but have not yet resolved the core voice quality issues. This remains an ongoing challenge.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Preprocessing Approaches](#preprocessing-approaches)
3. [Training Scripts](#training-scripts)
4. [Test Scripts](#test-scripts)
5. [Workflow Examples](#workflow-examples)
6. [Troubleshooting](#troubleshooting)
7. [Best Practices](#best-practices)

## Architecture Overview

The voice training system consists of several key components:

- **Audio Preprocessors**: Clean and prepare audio for training
- **Multi-Reference Trainer**: Train on curated high-quality samples
- **Quality Validators**: Ensure audio meets training requirements
- **Test Scripts**: Validate preprocessing and training workflows

## Preprocessing Approaches

### 1. Basic Audio Preprocessing

**Script**: `src/pipeline/stages/audio_preprocessor.py`
**Config**: `config/audio/preprocessing.yaml`
**Test Script**: `test_manual_refs_basic.py`

**Features**:
- Light denoising (spectral gating)
- Volume normalization (peak/RMS/LUFS)
- Silence trimming
- Resampling to target rate
- Basic quality validation

**Best For**: Already good quality audio that needs minimal processing

**Usage Example**:
```python
from src.pipeline.stages.audio_preprocessor import AudioPreprocessor

preprocessor = AudioPreprocessor()
results = await preprocessor.process_directory("input/", "output/")
```

### 2. Enhanced Audio Preprocessing

**Script**: `src/pipeline/stages/enhanced_audio_preprocessor.py`
**Config**: `config/audio/enhanced_preprocessing.yaml`
**Test Script**: `test_manual_refs_enhanced.py`

**Features**:
- Vocal fry detection and removal
- Resemble-Enhance voice enhancement
- Demucs vocal separation (when available)
- Advanced spectral processing
- Duration preservation
- Higher quality thresholds

**Best For**: Poor quality audio with vocal fry, background noise, or clarity issues

**Usage Example**:
```python
from src.pipeline.stages.enhanced_audio_preprocessor import EnhancedAudioPreprocessor

preprocessor = EnhancedAudioPreprocessor()
results = await preprocessor.process_directory("input/", "output/")
```

## Training Scripts

### Multi-Reference Trainer

**Script**: `multi_reference_trainer.py`

**Purpose**: Train TTS models on curated reference samples with parameter optimization

**Features**:
- Multi-reference training (combine multiple audio samples)
- Parameter grid search (temperature, repetition penalty, length penalty)
- Organized output with subfolders
- Comprehensive logging and metrics
- Support for both basic and enhanced preprocessed audio

**Command Line Usage**:
```bash
# Train on manually curated references
python multi_reference_trainer.py --references temp/manual_refs_basic/*.wav

# Train on enhanced processed audio
python multi_reference_trainer.py --references temp/manual_refs_enhanced/*.wav

# Train on specific files
python multi_reference_trainer.py --references \
  "path/to/sample1.wav" \
  "path/to/sample2.wav" \
  "path/to/sample3.wav"
```

**Output Structure**:
```
artifacts/character_voices/meowth_multi_test/
├── combined_reference_audio.wav
├── training_log.txt
├── temp_0.5_rep_1.0_len_1.0/
│   ├── synthesis_1.wav
│   ├── synthesis_2.wav
│   └── synthesis_3.wav
├── temp_0.7_rep_1.1_len_1.0/
│   └── ...
└── temp_0.9_rep_1.2_len_1.0/
    └── ...
```

### Manual Character Training (Legacy)

**Scripts**: 
- `manual_character_training.py`
- `manual_character_training_fixed.py`

**Purpose**: Original training scripts for individual character voices

**Status**: Superseded by multi-reference trainer

## Test Scripts

### 1. Basic Preprocessing Test

**Script**: `test_manual_refs_basic.py`

**Purpose**: Test basic preprocessing on manual reference files

**Features**:
- Loads references from `manual_refs.txt`
- Applies basic preprocessing
- Clean temp directories (no contamination)
- Validation scoring
- Ready-to-use training commands

**Usage**:
```bash
python test_manual_refs_basic.py
```

### 2. Enhanced Preprocessing Test

**Script**: `test_manual_refs_enhanced.py`

**Purpose**: Test enhanced preprocessing on manual reference files

**Features**:
- Same as basic test but with enhanced processing
- Vocal fry detection and removal
- Resemble-Enhance voice enhancement
- Duration preservation
- Quality improvements

**Usage**:
```bash
python test_manual_refs_enhanced.py
```

### 3. General Enhanced Preprocessing Test

**Script**: `test_enhanced_preprocessing.py`

**Purpose**: Test enhanced preprocessing on sample Meowth files

**Features**:
- Tests with predefined Meowth samples
- Dependency checking
- Performance metrics
- Fallback testing

**Usage**:
```bash
python test_enhanced_preprocessing.py
```

## Workflow Examples

### Workflow 1: Manual Reference Training (Recommended)

1. **Curate References**: Add high-quality samples to `manual_refs.txt`
   ```
   meowth:resources/validation_samples_v4/meowth/Pokemon S01E17 Island of Giant Pokemon_SPEAKER_15_006.wav
   meowth:resources/validation_samples_v4/meowth/Pokemon S01E17 Island of Giant Pokemon_SPEAKER_15_007.wav
   meowth:resources/validation_samples_v4/meowth/Pokemon S01E79 Fourth Round Rumble_SPEAKER_20_009.wav
   ```

2. **🎵 CRITICAL: Remove Background Music (REQUIRED for TV/Anime)**:
   ```bash
   python main.py remove-background-music --install  # First time only
   python main.py remove-background-music             # Process validation samples
   ```
   This step is **essential** for Pokemon episodes and other TV shows with constant background music.

3. **Test Basic Preprocessing**:
   ```bash
   python test_manual_refs_basic.py
   ```

4. **Train with Multi-Reference Trainer**:
   ```bash
   python multi_reference_trainer.py --references temp/manual_refs_basic/*.wav
   ```

5. **Evaluate Results**: Check `artifacts/character_voices/meowth_multi_test/` for outputs

### Workflow 2: Enhanced Processing (For Poor Quality Audio)

1. **🎵 CRITICAL: Remove Background Music First**:
   ```bash
   python main.py remove-background-music --install  # First time only
   python main.py remove-background-music             # Always run this first
   ```

2. **Test Enhanced Preprocessing**:
   ```bash
   python test_manual_refs_enhanced.py
   ```

3. **Train with Enhanced Audio**:
   ```bash
   python multi_reference_trainer.py --references temp/manual_refs_enhanced/*.wav
   ```

4. **Compare Results**: Compare basic vs enhanced preprocessing results

### Workflow 3: Main CLI Pipeline

1. **Individual Stages** (Recommended for control):
   ```bash
   python main.py extract-audio --input resources/videos/
   python main.py transcribe --input resources/audio/
   python main.py segment-speakers --audio resources/audio/ --transcripts resources/transcripts/
   
   # 🎵 CRITICAL: Remove background music before training
   python main.py remove-background-music --install  # First time only
   python main.py remove-background-music             # Process validation samples
   
   python main.py train --model xtts_v2 --dataset manual_refs.txt
   ```

2. **Full Pipeline** (Note: Does NOT include background music removal):
   ```bash
   # ⚠️ WARNING: This skips background music removal step
   python main.py run-pipeline --input resources/videos/ --output artifacts/models/
   ```

## Configuration Files

### Basic Preprocessing Config

**File**: `config/audio/preprocessing.yaml`

**Key Settings**:
```yaml
preprocessing:
  trim_silence:
    enabled: true
    threshold_db: -40
  denoise:
    enabled: true
    method: "spectral_gating"
  normalize:
    enabled: true
    method: "lufs"
    target_lufs: -23
  resample:
    target_rate: 24000
```

### Enhanced Preprocessing Config

**File**: `config/audio/enhanced_preprocessing.yaml`

**Key Settings**:
```yaml
preprocessing:
  vocal_fry_removal:
    enabled: true
    aggressive: 0.3
  enhance:
    enabled: true
    method: "resemble_enhance"
  denoise:
    enabled: true
    method: "demucs"
    fallback_method: "spectral_gating"
```

## Manual References File

**File**: `manual_refs.txt`

**Format**:
```
character:path/to/audio/file.wav
```

**Example**:
```
meowth:resources/validation_samples_v4/meowth/Pokemon S01E17 Island of Giant Pokemon_SPEAKER_15_006.wav
meowth:resources/validation_samples_v4/meowth/Pokemon S01E17 Island of Giant Pokemon_SPEAKER_15_007.wav
meowth:resources/validation_samples_v4/meowth/Pokemon S01E79 Fourth Round Rumble_SPEAKER_20_009.wav
```

## Troubleshooting

### Common Issues

1. **Raspy/Low Quality Voices**
   - **Cause**: Over-processing with enhanced preprocessing
   - **Solution**: Use basic preprocessing for already good quality audio
   - **Command**: `python test_manual_refs_basic.py`

2. **Duration Inflation**
   - **Cause**: Resemble-Enhance changing audio length
   - **Solution**: Duration preservation now implemented
   - **Fix**: Enhanced preprocessor now truncates/pads to original length

3. **Contaminated Test Results**
   - **Cause**: Temp files from previous runs
   - **Solution**: All test scripts now clean temp directories
   - **Evidence**: Look for "🧹 Cleaned temp directories" message

4. **Low SNR Validation Scores**
   - **Cause**: Overly strict validation thresholds
   - **Solution**: Adjusted thresholds in enhanced config
   - **Setting**: `snr_threshold: 8` (reduced from 15)

5. **Vocal Fry Detection Errors**
   - **Cause**: Array dimension mismatches in spectral processing
   - **Solution**: Fixed FFT parameters and added dimension checks
   - **Fix**: Consistent `n_fft=2048` across all spectral operations

### Dependency Issues

1. **Demucs Not Available**
   - **Status**: Currently disabled (complex setup required)
   - **Fallback**: Spectral denoising automatically used
   - **Message**: "Demucs setup requires additional configuration"

2. **Resemble-Enhance Errors**
   - **Common**: "Expected 1D waveform, got 2D"
   - **Solution**: Audio flattening implemented
   - **Prevention**: Proper mono conversion before processing

## Best Practices

### For High-Quality Source Audio

1. **Use Basic Preprocessing**: Don't over-process already good audio
2. **Manual Curation**: Hand-pick the best samples in `manual_refs.txt`
3. **Test First**: Always run `test_manual_refs_basic.py` before training
4. **Clean Runs**: Rely on automatic temp directory cleaning

### For Poor-Quality Source Audio

1. **Use Enhanced Preprocessing**: Apply vocal fry removal and enhancement
2. **Check Dependencies**: Ensure Resemble-Enhance is available
3. **Validate Results**: Check quality scores after processing
4. **Compare Approaches**: Test both basic and enhanced to compare

### General Training Tips

1. **Multiple References**: Use 3-5 high-quality samples per character
2. **Parameter Testing**: Let multi-reference trainer test different parameters
3. **Organized Output**: Use the structured output folders for comparison
4. **Quality First**: Better to have fewer, higher-quality samples

### File Organization

```
tts-trainer/
├── manual_refs.txt              # Curated reference files
├── multi_reference_trainer.py   # Main training script
├── test_manual_refs_basic.py    # Basic preprocessing test
├── test_manual_refs_enhanced.py # Enhanced preprocessing test
├── temp/                        # Temporary processing (auto-cleaned)
│   ├── manual_refs_basic/       # Basic processed references
│   └── manual_refs_enhanced/    # Enhanced processed references
└── artifacts/                   # Training outputs
    └── character_voices/
        └── meowth_multi_test/   # Generated voices
```

## Performance Notes

### Processing Times

- **Basic Preprocessing**: ~1-2 seconds per file
- **Enhanced Preprocessing**: ~15-20 seconds per file (due to Resemble-Enhance)
- **Multi-Reference Training**: ~5-10 minutes (varies by parameter combinations)

### Resource Usage

- **GPU Required**: For Resemble-Enhance and TTS training
- **Memory**: 8GB+ recommended for enhanced processing
- **Storage**: ~1GB per character for full training pipeline

## Version History

### Current Version Features

- ✅ Clean temp directories (no contamination)
- ✅ Duration preservation in enhanced processing
- ✅ Fixed vocal fry detection array mismatches
- ✅ Improved SNR calculation for enhanced audio
- ✅ Manual reference workflow
- ✅ Multi-reference trainer with parameter optimization
- ✅ Both basic and enhanced preprocessing options

### Known Limitations

- **CRITICAL**: Voice training still produces raspy/strange voices despite all optimization efforts
- **UNRESOLVED**: Meowth character voices remain unsatisfactory even with manual reference curation
- Demucs integration disabled (complex setup)
- Resemble-Enhance requires significant processing time
- Enhanced processing may over-process good quality audio
- Parameter optimization is computationally intensive
- **ONGOING**: Core TTS voice quality issues not yet solved by preprocessing improvements

## Current Status and Ongoing Issues

### Unresolved Voice Quality Problems

Despite implementing all documented fixes and optimizations:

- **Meowth voices still sound raspy/strange** even with:
  - ✅ Basic preprocessing (no over-processing)
  - ✅ Manually curated high-quality references
  - ✅ Clean temp directories (no contamination)
  - ✅ Fixed technical processing issues
  - ✅ Parameter optimization

- **Root cause remains unknown** and may involve:
  - TTS model architecture limitations
  - Training parameter incompatibility
  - Fundamental voice cloning approach issues
  - Reference audio characteristics not captured by current quality metrics

### Research Needed

The voice quality issue suggests deeper problems requiring investigation into:

1. **Alternative TTS Models**: VITS, other XTTS variants, or different architectures entirely
2. **Training Methodology**: Different approaches to fine-tuning on character voices
3. **Reference Selection**: What makes truly effective reference audio beyond technical quality
4. **Model Parameters**: Systematic exploration of learning rates, epochs, loss functions

## Future Improvements

1. **PRIORITY: Resolve Core Voice Quality Issues** - Investigate alternative TTS approaches
2. **Automatic Quality Assessment**: Choose basic vs enhanced based on input quality
3. **Streaming Processing**: Real-time audio enhancement for Discord bot
4. **Character-Specific Presets**: Optimized settings per character type
5. **Batch Processing**: Process entire character datasets efficiently
6. **Quality Metrics**: More sophisticated audio quality scoring
7. **Model Architecture Research**: Test fundamentally different voice cloning approaches 