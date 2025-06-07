# Speaker Diarization and Segmentation Setup

This document provides setup instructions for the newly implemented speaker diarization and segmentation pipeline.

## 📦 **Required Dependencies**

The speaker pipeline requires additional dependencies that may need manual installation:

### **1. Install pyannote.audio**
```bash
# Install pyannote.audio for speaker diarization
pip install pyannote.audio

# You may need to accept the terms of use for the pretrained models
# Visit: https://huggingface.co/pyannote/speaker-diarization-3.1
```

### **2. Verify Installation**
```bash
# Test that all dependencies are available
python -c "from faster_whisper import WhisperModel; print('✅ Faster-Whisper available')"
python -c "from pyannote.audio import Pipeline; print('✅ pyannote.audio available')"
python -c "import librosa; print('✅ librosa available')"
python -c "import soundfile; print('✅ soundfile available')"
```

## 🎯 **Correct Workflow for Pokemon Episodes**

The speaker pipeline implements the correct workflow for character-specific TTS training:

### **Step 1: Extract Audio from Videos**
```bash
python main.py extract-audio --input resources/videos/ --output resources/audio/
```

### **Step 2: Transcribe with Speaker Diarization** 
```bash
# This is the key step - it identifies WHO is speaking WHEN
python main.py transcribe --input resources/audio/ --output resources/transcripts/ --speaker-diarization
```

### **Step 3: Analyze Speakers**
```bash
# Analyze who was detected and their speaking patterns
python main.py analyze-speakers --transcripts resources/transcripts/ --output speaker_analysis.json
```

### **Step 4: Configure Speaker Mapping**
Edit `config/speaker_mapping.yaml` to map detected speakers to Pokemon characters:
```yaml
mapping:
  manual_mappings:
    "SPEAKER_00": "ash"      # Map detected speaker to Ash
    "SPEAKER_01": "misty"    # Map detected speaker to Misty
    "SPEAKER_02": "brock"    # Map detected speaker to Brock
    # etc.
```

### **Step 5: Extract Character Segments**
```bash
# Extract character-specific audio clips
python main.py segment-speakers --audio resources/audio/ --transcripts resources/transcripts/ --output resources/segments/
```

### **Step 6: Preprocess Character Clips**
```bash
# Now preprocess the SHORT character clips (not full episodes)
python main.py preprocess-audio --input resources/segments/ash/ --output resources/processed/ash/
python main.py preprocess-audio --input resources/segments/misty/ --output resources/processed/misty/
# etc. for each character
```

## 🧪 **Testing the Pipeline**

### **Test with a Single Episode**
```bash
# 1. Create test directory with one episode
mkdir -p test_episode
cp "resources/audio/Pokemon S01E43 March of the Exeggutor Squad.wav" test_episode/

# 2. Transcribe with speaker diarization
python main.py transcribe --input test_episode/ --output test_transcripts/ --speaker-diarization

# 3. Analyze detected speakers
python main.py analyze-speakers --transcripts test_transcripts/

# 4. Extract segments (after configuring speaker mapping)
python main.py segment-speakers --audio test_episode/ --transcripts test_transcripts/ --output test_segments/
```

## 📊 **Expected Output Structure**

After running the full pipeline, you'll have:

```
resources/
├── transcripts/
│   ├── transcripts/          # Timestamped transcripts with speaker labels
│   ├── speakers/             # Speaker timeline data  
│   └── segments/             # (created by transcriber if enabled)
├── segments/
│   ├── ash/                  # Ash's audio clips
│   │   ├── Pokemon_S01E43_ash_001.wav
│   │   ├── Pokemon_S01E43_ash_002.wav
│   │   └── ...
│   ├── misty/                # Misty's audio clips
│   └── brock/                # Brock's audio clips
└── processed/
    ├── ash/                  # Preprocessed Ash clips for training
    ├── misty/                # Preprocessed Misty clips for training
    └── brock/                # Preprocessed Brock clips for training
```

## 🎭 **Character Mapping Process**

1. **Run speaker analysis** to see detected speakers and their characteristics
2. **Listen to sample audio** to identify which speaker is which character
3. **Update speaker mapping config** with the correct character assignments
4. **Re-run segmentation** to extract properly labeled character clips

## ⚠️ **Important Notes**

- **Speaker diarization requires significant compute** - expect longer processing times
- **Manual mapping is required** - the system can't automatically know that "SPEAKER_00" is Ash
- **Quality filtering is built-in** - short segments and poor quality audio are automatically filtered out
- **This creates the SHORT clips** that the audio preprocessor expects (2-15 seconds per character)

## 🚀 **Next Steps After Segmentation**

Once you have character-specific segments:

1. **Preprocess each character's clips** separately
2. **Build character-specific datasets** 
3. **Train character-specific TTS models**
4. **Test character voices in Discord bot**

This pipeline finally enables the Pokemon character TTS training that the project was designed for! 