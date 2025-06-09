# Voice Training Troubleshooting Guide

## Common Issues and Solutions

### 1. Raspy/Unnatural Voices (CRITICAL UNRESOLVED ISSUE)

**Symptoms**:
- Generated voices sound raspy, robotic, or unnatural
- Characters don't sound like their source material
- Voice quality is worse than expected
- **ONGOING**: Meowth voices remain raspy/strange despite all attempts

**Root Cause**: Likely over-processing with enhanced audio preprocessing, but **UNRESOLVED** - basic preprocessing also produces poor results

**Attempted Solutions**:
```bash
# Use basic preprocessing instead of enhanced
python test_manual_refs_basic.py
python multi_reference_trainer.py --references temp/manual_refs_basic/*.wav
```

**Why This Partially Works**: Enhanced preprocessing (vocal fry removal, Resemble-Enhance) can over-process already good quality audio, introducing artifacts.

**⚠️ CURRENT STATUS**: Even with basic preprocessing and manually curated high-quality references, voice training still produces unsatisfactory raspy/strange voices for characters like Meowth. This suggests the issue may be deeper in the TTS model architecture, training parameters, or reference audio selection criteria.

**Best Practice**: Only use enhanced preprocessing for genuinely poor quality source audio, but be aware that voice quality issues persist regardless of preprocessing method.

---

### 2. Duration Inflation

**Symptoms**:
- Audio files becoming much longer after processing
- 3-second files becoming 13+ seconds
- Training taking much longer than expected

**Root Cause**: Resemble-Enhance changing audio duration without preservation

**Solution**: Fixed in enhanced preprocessor - duration is now preserved automatically

**Technical Fix**:
```python
# Enhanced preprocessor now includes:
original_length = len(audio)
# ... processing ...
if len(enhanced_audio) != original_length:
    if len(enhanced_audio) > original_length:
        enhanced_audio = enhanced_audio[:original_length]
    else:
        enhanced_audio = np.pad(enhanced_audio, (0, original_length - len(enhanced_audio)))
```

---

### 3. Contaminated Test Results

**Symptoms**:
- Test results inconsistent between runs
- Previous processing affecting new tests
- Unexpected files in temp directories

**Root Cause**: Temp directories not cleaned between test runs

**Solution**: All test scripts now automatically clean temp directories

**Evidence**: Look for this message:
```
🧹 Cleaned temp directories before processing
```

**Scripts Fixed**:
- `test_enhanced_preprocessing.py`
- `test_manual_refs_enhanced.py` 
- `test_manual_refs_basic.py`

---

### 4. Vocal Fry Detection Array Errors

**Symptoms**:
```
IndexError: boolean index did not match indexed array along axis 0; 
size of axis is 1025 but size of corresponding boolean axis is 513
```

**Root Cause**: FFT parameter mismatches in spectral processing

**Solution**: Fixed with consistent FFT parameters across all operations

**Technical Fix**:
```python
# All spectral operations now use consistent parameters:
n_fft = 2048
hop_length = 512
# This ensures frequency bins match across operations
```

---

### 5. Low SNR Validation Scores

**Symptoms**:
- Quality validation failing with low SNR scores (~0.4dB)
- Enhanced audio being rejected despite sounding better

**Root Cause**: Overly strict validation thresholds

**Solution**: Adjusted thresholds in enhanced preprocessing config

**Settings Changed**:
```yaml
# Old threshold
snr_threshold: 15  # Too strict

# New threshold  
snr_threshold: 8   # More realistic
```

---

### 6. Demucs Integration Issues

**Symptoms**:
- "Demucs detected but failed to initialize"
- Demucs import errors or setup problems

**Current Status**: Demucs integration disabled (complex setup)

**Fallback**: Automatic fallback to spectral denoising

**Message You'll See**:
```
⚠️ Demucs setup requires additional configuration, using fallback denoising
```

**Action Required**: None - fallback works fine for most cases

---

### 7. Resemble-Enhance Dimension Errors

**Symptoms**:
```
RuntimeError: Expected 1D waveform, got 2D with shape [channels, samples]
```

**Root Cause**: Audio not properly flattened to 1D before enhancement

**Solution**: Fixed with proper audio flattening

**Technical Fix**:
```python
# Audio is now properly flattened before enhancement:
if audio.ndim > 1:
    audio = audio.flatten()
```

---

### 8. Manual References Path Issues

**Symptoms**:
- Files not found when processing manual references
- Mixed path separators (Windows/Linux)

**Root Cause**: Inconsistent path formats in `manual_refs.txt`

**Solution**: Use forward slashes consistently:

**Correct Format**:
```
meowth:resources/validation_samples_v4/meowth/Pokemon S01E17 Island of Giant Pokemon_SPEAKER_15_006.wav
```

**Incorrect Format**:
```
meowth:resources\validation_samples_v4\meowth\Pokemon S01E17 Island of Giant Pokemon_SPEAKER_15_006.wav
```

---

## Diagnostic Commands

### Check Audio Quality
```bash
# Test basic preprocessing
python test_manual_refs_basic.py

# Test enhanced preprocessing  
python test_manual_refs_enhanced.py

# Compare outputs and quality scores
```

### Verify Dependencies
```bash
# Check if enhanced processing dependencies are available
python -c "
try:
    from resemble_enhance.enhancer.inference import denoise, enhance
    print('✅ Resemble-Enhance available')
except:
    print('❌ Resemble-Enhance not available')

try:
    import demucs
    print('✅ Demucs available') 
except:
    print('❌ Demucs not available')
"
```

### Check File Paths
```bash
# Verify manual references exist
python -c "
with open('manual_refs.txt', 'r') as f:
    for line in f:
        if ':' in line:
            char, path = line.strip().split(':', 1)
            import os
            if os.path.exists(path):
                print(f'✅ {char}: {path}')
            else:
                print(f'❌ {char}: {path} NOT FOUND')
"
```

## Performance Troubleshooting

### Slow Enhanced Processing
**Cause**: Resemble-Enhance is compute-intensive
**Solution**: 
- Use basic preprocessing for good quality audio
- Ensure GPU is available for enhanced processing
- Process files in smaller batches

### High Memory Usage
**Cause**: Large audio files or batch processing
**Solution**:
- Process files individually rather than in batches
- Monitor GPU memory usage
- Restart Python session between large processing runs

### Training Taking Too Long
**Cause**: Too many parameter combinations in multi-reference trainer
**Solution**: Reduce parameter grid in `multi_reference_trainer.py`:
```python
# Reduce combinations for faster testing
temperature_values = [0.7, 0.9]  # Instead of [0.5, 0.7, 0.9]
repetition_penalty_values = [1.0, 1.1]  # Instead of [1.0, 1.1, 1.2]
```

## Best Practices for Avoiding Issues

### 1. Always Test First
```bash
# Always run test before training
python test_manual_refs_basic.py
# Then train only if test succeeds
python multi_reference_trainer.py --references temp/manual_refs_basic/*.wav
```

### 2. Use Basic for Good Audio
- If source audio is already good quality, use basic preprocessing
- Enhanced preprocessing is for poor quality audio only

### 3. Monitor Temp Directories
- Scripts automatically clean temp directories
- But verify they're clean if you see unexpected behavior

### 4. Check Quality Scores
- Basic preprocessing: Should see reasonable quality scores (6-8/10)
- Enhanced preprocessing: Should see improvement in quality scores
- If quality gets worse, use basic preprocessing instead

### 5. Validate Results
- Always listen to a few samples before training
- Compare processed audio with originals
- If processed audio sounds worse, switch preprocessing methods

## When to Use Each Preprocessing Method

### Use Basic Preprocessing When:
- ✅ Source audio is already good quality
- ✅ Audio is clear with minimal background noise
- ✅ You want to preserve natural characteristics
- ✅ Enhanced preprocessing makes audio sound worse

### Use Enhanced Preprocessing When:
- ✅ Source audio has vocal fry or raspiness
- ✅ Audio has background noise or interference
- ✅ Audio quality is genuinely poor
- ✅ Basic preprocessing doesn't provide sufficient quality

## Current Limitations and Known Issues

### Unresolved Core Issues

**Voice Quality Problems**:
- Despite all documented fixes and optimizations, TTS training still produces raspy/strange voices
- Meowth character voices remain unsatisfactory regardless of preprocessing method
- Issue appears to be fundamental to the TTS model or training approach, not just preprocessing

**Possible Root Causes** (Unverified):
- TTS model architecture may not be suitable for character voice cloning
- Training parameters may need significant adjustment
- Reference audio may need different selection criteria
- XTTS model may have inherent limitations for certain voice types

### Research Directions Needed

1. **Model Architecture Investigation**: Test different TTS models (VITS, other XTTS variants)
2. **Training Parameter Analysis**: Systematic testing of learning rates, batch sizes, epochs
3. **Reference Audio Analysis**: Investigate what makes good reference audio beyond basic quality metrics
4. **Alternative Approaches**: Consider different voice cloning methodologies entirely

## Emergency Fixes

### If Everything Breaks:
1. **Clean Everything**:
   ```bash
   # Remove temp directories
   rm -rf temp/
   mkdir temp
   ```

2. **Start Simple**:
   ```bash
   # Test with basic preprocessing only
   python test_manual_refs_basic.py
   ```

3. **Use Known Good Files**:
   - Verify your `manual_refs.txt` files exist and sound good
   - Start with just one file for testing

4. **Check Dependencies**:
   ```bash
   # Reinstall if needed
   pip install --upgrade tts
   pip install resemble-enhance  # Only if using enhanced processing
   ```

5. **Accept Current Limitations**:
   - Current system can process audio and run training
   - Voice quality issues are unresolved and may require fundamental changes to approach 