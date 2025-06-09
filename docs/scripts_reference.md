# Voice Training Scripts Quick Reference

**⚠️ CURRENT STATUS**: Voice training is still producing raspy/strange voices despite all optimizations. These scripts work technically but core voice quality issues remain unresolved.

## Main Scripts

### Multi-Reference Trainer
**File**: `multi_reference_trainer.py`
**Purpose**: Train TTS models on curated audio samples with parameter optimization

```bash
# Basic usage
python multi_reference_trainer.py --references temp/manual_refs_basic/*.wav

# Enhanced preprocessing
python multi_reference_trainer.py --references temp/manual_refs_enhanced/*.wav

# Specific files
python multi_reference_trainer.py --references \
  "resources/validation_samples_v4/meowth/file1.wav" \
  "resources/validation_samples_v4/meowth/file2.wav"
```

## Test/Validation Scripts

### Basic Preprocessing Test
**File**: `test_manual_refs_basic.py`
**Purpose**: Test basic preprocessing on manual references

```bash
python test_manual_refs_basic.py
```

**Output**: Processes files from `manual_refs.txt` → `temp/manual_refs_basic/`

### Enhanced Preprocessing Test
**File**: `test_manual_refs_enhanced.py`
**Purpose**: Test enhanced preprocessing on manual references

```bash
python test_manual_refs_enhanced.py
```

**Output**: Processes files from `manual_refs.txt` → `temp/manual_refs_enhanced/`

### General Enhanced Test
**File**: `test_enhanced_preprocessing.py`
**Purpose**: Test enhanced preprocessing on sample files

```bash
python test_enhanced_preprocessing.py
```

**Output**: Tests predefined Meowth samples

## Configuration Files

### Manual References
**File**: `manual_refs.txt`
**Format**: `character:path/to/file.wav`

```
meowth:resources/validation_samples_v4/meowth/Pokemon S01E17 Island of Giant Pokemon_SPEAKER_15_006.wav
meowth:resources/validation_samples_v4/meowth/Pokemon S01E17 Island of Giant Pokemon_SPEAKER_15_007.wav
meowth:resources/validation_samples_v4/meowth/Pokemon S01E79 Fourth Round Rumble_SPEAKER_20_009.wav
```

### Audio Preprocessing Configs
- `config/audio/preprocessing.yaml` - Basic preprocessing settings
- `config/audio/enhanced_preprocessing.yaml` - Enhanced preprocessing settings

## Legacy Scripts (Deprecated)

### Manual Character Training
**Files**: 
- `manual_character_training.py`
- `manual_character_training_fixed.py`

**Status**: Superseded by `multi_reference_trainer.py`

## Output Locations

### Temporary Processing
- `temp/manual_refs_basic/` - Basic processed references
- `temp/manual_refs_enhanced/` - Enhanced processed references

### Training Outputs
```
artifacts/character_voices/meowth_multi_test/
├── combined_reference_audio.wav
├── training_log.txt
├── temp_0.5_rep_1.0_len_1.0/
│   ├── synthesis_1.wav
│   ├── synthesis_2.wav
│   └── synthesis_3.wav
└── [other parameter combinations]/
```

## Typical Workflow

1. **Setup**: Add references to `manual_refs.txt`
2. **Test**: Run `python test_manual_refs_basic.py`
3. **Train**: Run `python multi_reference_trainer.py --references temp/manual_refs_basic/*.wav`
4. **Evaluate**: Listen to results in `artifacts/character_voices/meowth_multi_test/`

## Key Features

- ✅ Automatic temp directory cleaning (no contamination)
- ✅ Duration preservation in enhanced processing
- ✅ Parameter grid search for optimal voice settings
- ✅ Organized output structure for easy comparison
- ✅ Comprehensive logging and validation 