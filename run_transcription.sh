#!/bin/bash
# Set cuDNN library path for the virtual environment
export LD_LIBRARY_PATH="$PWD/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:$PWD/venv/lib/python3.12/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH"

# Run the transcription with proper library paths
python main.py transcribe --input test_clips/ --speaker-diarization --diarization-config config/audio/speaker_diarization.yaml
