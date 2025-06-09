# Pokemon Voice Clustering Workflow
## Complete Guide: From Audio to Character-Specific TTS Training Data

### Overview
This workflow extracts character-specific voice segments from Pokemon episodes for TTS training. It processes raw audio files through speaker diarization, voice clustering, and content-aware character assignment.

### Prerequisites
- Python 3.12+ with virtual environment
- GPU support for faster processing
- FFmpeg for audio processing
- At least 16GB RAM recommended for full dataset

---

## Phase 1: Initial Setup

### 1.1 Environment Setup
```bash
# Create and activate virtual environment
cd ~/code/tts-trainer
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify GPU setup
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 1.2 Directory Structure
```
resources/
├── videos/           # Raw episode files
├── audio/           # Extracted audio (full episodes)
├── segments/        # Speaker-segmented clips
├── voice_clusters/  # Clustering results
└── validation_samples_v3/  # Final character datasets
```

---

## Phase 2: Audio Processing Pipeline

### 2.1 Extract Audio from Videos
```bash
# Extract audio from all video files
python main.py extract-audio --input resources/videos/ --output resources/audio/

# Expected output: One .wav file per episode in resources/audio/
```

### 2.2 Transcribe Audio with Speaker Diarization
```bash
# Generate transcripts with speaker separation
python main.py transcribe --input resources/audio/ --output resources/transcripts/

# Expected output: JSON files with speaker-labeled transcripts in resources/transcripts/transcripts/
```

### 2.3 Segment Audio by Speaker
```bash
# Extract individual speaker segments
python main.py segment-speakers --audio resources/audio/ --transcripts resources/transcripts/transcripts/ --output resources/segments/

# Expected output: 
# - resources/segments/SPEAKER_XX/ directories with audio clips
# - Each clip has corresponding .json metadata file
```

---

## Phase 3: Voice Clustering & Character Assignment

### 3.1 Configuration
Key configuration file: `config/audio/voice_clustering.yaml`
```yaml
clustering:
  eps: 0.15                    # Aggressive merging for similar voices
  min_samples: 15              # High confidence requirement
  
quality:
  min_cluster_duration: 180.0  # 3+ minutes required
  min_episodes_per_cluster: 4  # Must appear in multiple episodes
  min_segments_per_cluster: 50 # Substantial dialogue required
```

### 3.2 Voice Clustering with Content Analysis
```bash
# Perform intelligent voice clustering
python main.py cluster-voices --segments resources/segments --output resources/voice_clusters

# Expected output:
# - resources/voice_clusters/voice_clustering_results.json
# - Character assignments based on dialogue content analysis
# - Mixed cluster detection and warnings
```

### 3.3 Generate Training Datasets
```bash
# Create character-specific validation samples
python main.py create-validation-samples --clustering-results resources/voice_clusters/voice_clustering_results.json --output resources/validation_samples_v3

# Expected output:
# - resources/validation_samples_v3/CHARACTER_NAME/ directories
# - Audio samples + full transcript for each character
# - Ready for TTS training
```

---

## Phase 4: Quality Validation

### 4.1 Character Distribution Check
```bash
# Check final character line counts
for dir in resources/validation_samples_v3/*/; do
    char=$(basename "$dir")
    if [ -f "$dir/full_transcript.txt" ]; then
        lines=$(grep -c "SPEAKER_" "$dir/full_transcript.txt" 2>/dev/null || echo "0")
        echo "$char: $lines lines"
    fi
done | sort -k2 -nr
```

### 4.2 Content Quality Verification
```bash
# Verify character-specific content
echo "=== ASH CONTENT ==="
grep "\[" resources/validation_samples_v3/*/ash/full_transcript.txt | head -3

echo "=== MISTY CONTENT ==="
grep -i "bike\|starmie\|water" resources/validation_samples_v3/*/misty/full_transcript.txt | head -3

echo "=== MIXED CLUSTERS DETECTED ==="
grep -i "MIXED CLUSTER" resources/voice_clusters/voice_clustering_results.json
```

---

## Phase 5: Full Dataset Processing

### 5.1 Scale to Complete Audio Collection
```bash
# Process ALL episodes (not just audio_small)
# WARNING: This will process the entire Pokemon episode collection

# 1. Extract audio from all videos
python main.py extract-audio --input resources/videos/ --output resources/audio/

# 2. Transcribe all episodes (very time-intensive)
python main.py transcribe --input resources/audio/ --output resources/transcripts/

# 3. Segment all speakers
python main.py segment-speakers --audio resources/audio/ --transcripts resources/transcripts/ --output resources/segments/

# 4. Cluster voices across all episodes
python main.py cluster-voices --segments resources/segments --output resources/voice_clusters

# 5. Generate final character datasets
python main.py create-validation-samples --clustering-results resources/voice_clusters/voice_clustering_results.json --output resources/character_datasets_final
```

### 5.2 Expected Full Dataset Results
- **Processing Time**: 8-12 hours for ~75 episodes
- **Ash**: 2000-5000+ dialogue lines
- **Main Characters**: 500-2000+ lines each
- **Total Speakers**: 50-100+ unique clusters
- **Storage**: 20-50GB+ total

---

## Key Algorithm Features

### Content-Aware Character Assignment
- **Dialogue Pattern Recognition**: Identifies character-specific speech patterns
- **Theme Song Filtering**: Separates singing from character dialogue
- **Mixed Cluster Detection**: Identifies when multiple characters are grouped together
- **Quality Scoring**: Prioritizes high-confidence character assignments

### Character Patterns Used
```python
character_patterns = {
    'ash': ['pikachu', 'gotta catch', 'i choose you', 'pokemon master'],
    'misty': ['water pokemon', 'starmie', 'psyduck', 'bike', 'ash ketchum'],
    'brock': ['rock pokemon', 'onix', 'geodude', 'breeding', 'brock'],
    'jessie': ['team rocket', 'prepare for trouble', 'arbok'],
    'james': ['team rocket', 'make it double', 'ekans', 'weezing'],
    'meowth': ['meowth', 'that.*right', 'team rocket', 'boss']
}
```

---

## Troubleshooting

### Common Issues
1. **Memory Issues**: Reduce batch size in config, process smaller chunks
2. **Mixed Clusters**: Review recommendations, manual verification may be needed
3. **Low Character Counts**: Adjust quality thresholds in voice_clustering.yaml
4. **Theme Song Contamination**: Patterns may need refinement for specific episodes

### Quality Validation
- Each character should have 50+ dialogue segments minimum
- Check for authentic character-specific content
- Verify cross-episode consistency
- Review mixed cluster warnings

---

## Output Structure
```
resources/validation_samples_v3/
├── 12_ash/
│   ├── 01_sample.wav
│   ├── 02_sample.wav
│   ├── 03_sample.wav
│   ├── full_transcript.txt    # All character dialogue
│   └── cluster_info.txt       # Technical details
├── 11_misty/
│   └── [same structure]
└── [additional characters...]
```

---

## Success Metrics (5-Episode Test)
- **Ash**: 493 pure dialogue lines ✅
- **Misty**: 117 character-specific lines ✅  
- **Brock**: 81 rock Pokemon trainer lines ✅
- **Team Rocket**: 93+53+41 lines ✅
- **Theme Song Separation**: 100% effective ✅
- **Mixed Cluster Detection**: 2 detected and handled ✅

This workflow produces high-quality, character-specific voice datasets ready for TTS model training. 