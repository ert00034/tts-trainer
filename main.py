#!/usr/bin/env python3
"""
TTS Trainer - Pipeline-Orchestrated Architecture
Main CLI entry point for all operations
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pipeline.orchestrator import PipelineOrchestrator
from models.base_trainer import ModelRegistry
from utils.logging_utils import setup_logging
from utils.file_utils import validate_input_path, create_output_dir


def setup_parser() -> argparse.ArgumentParser:
    """Setup the main argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        description="TTS Trainer - Convert videos to TTS training data and train models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline from videos to trained model
  python main.py run-pipeline --input resources/videos/ --output artifacts/models/
  
  # Extract audio from videos only
  python main.py extract-audio --input resources/videos/ --output resources/audio/
  
  # Train a specific model
  python main.py train --model xtts_v2 --dataset resources/datasets/my_voice/
  
  # Launch Discord bot
  python main.py discord-bot --token YOUR_DISCORD_TOKEN
  
  # Test inference
  python main.py inference --model artifacts/models/my_model.pth --text "Hello world!"
        """)
    
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--config", type=str, help="Path to custom config file")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Run Pipeline Command
    pipeline_parser = subparsers.add_parser("run-pipeline", help="Run the full video-to-model pipeline")
    pipeline_parser.add_argument("--input", required=True, help="Input directory containing video files")
    pipeline_parser.add_argument("--output", required=True, help="Output directory for trained models")
    pipeline_parser.add_argument("--model", default="xtts_v2", choices=["xtts_v2", "vits", "tortoise"], 
                                 help="Model type to train")
    pipeline_parser.add_argument("--skip-training", action="store_true", 
                                 help="Skip training, only prepare dataset")
    
    # Extract Audio Command
    audio_parser = subparsers.add_parser("extract-audio", help="Extract audio from video files")
    audio_parser.add_argument("--input", required=True, help="Input directory containing video files")
    audio_parser.add_argument("--output", default="resources/audio/", help="Output directory for audio files")
    audio_parser.add_argument("--format", default="wav", choices=["wav", "mp3", "flac"], help="Output audio format")
    
    # Transcribe Command
    transcribe_parser = subparsers.add_parser("transcribe", help="Generate transcripts from audio files")
    transcribe_parser.add_argument("--input", required=True, help="Input directory containing audio files")
    transcribe_parser.add_argument("--output", default="resources/transcripts/", help="Output directory for transcripts")
    transcribe_parser.add_argument("--model", default="large-v3", help="Whisper model size")
    transcribe_parser.add_argument("--speaker-diarization", action="store_true", help="Enable speaker separation")
    
    # Train Command
    train_parser = subparsers.add_parser("train", help="Train a TTS model")
    train_parser.add_argument("--model", required=True, choices=["xtts_v2", "vits", "tortoise"],
                              help="Model type to train")
    train_parser.add_argument("--dataset", required=True, help="Path to training dataset")
    train_parser.add_argument("--output", default="artifacts/models/", help="Output directory for trained model")
    train_parser.add_argument("--epochs", type=int, help="Number of training epochs (overrides config)")
    train_parser.add_argument("--batch-size", type=int, help="Batch size (overrides config)")
    train_parser.add_argument("--resume", help="Path to checkpoint to resume from")
    
    # Inference Command
    inference_parser = subparsers.add_parser("inference", help="Test model inference")
    inference_parser.add_argument("--model", required=True, help="Path to trained model or model name")
    inference_parser.add_argument("--text", required=True, help="Text to synthesize")
    inference_parser.add_argument("--reference", help="Path to reference audio file for voice cloning")
    inference_parser.add_argument("--output", default="inference_output.wav", help="Output audio file")
    inference_parser.add_argument("--streaming", action="store_true", help="Enable streaming inference")
    
    # Discord Bot Command
    bot_parser = subparsers.add_parser("discord-bot", help="Launch Discord bot")
    bot_parser.add_argument("--token", required=True, help="Discord bot token")
    bot_parser.add_argument("--model", default="xtts_v2", help="Model to use for TTS")
    bot_parser.add_argument("--voice-clone", help="Path to reference audio for voice cloning")
    
    # Validate Command
    validate_parser = subparsers.add_parser("validate", help="Validate audio/dataset quality")
    validate_parser.add_argument("--input", required=True, help="Input directory to validate")
    validate_parser.add_argument("--type", choices=["audio", "dataset", "transcripts"], 
                                 default="audio", help="Type of validation to perform")
    
    return parser


async def run_pipeline(args):
    """Run the full video-to-model pipeline."""
    print(f"🚀 Starting full pipeline: {args.input} → {args.output}")
    
    orchestrator = PipelineOrchestrator(
        model_type=args.model,
        skip_training=args.skip_training,
        config_path=args.config
    )
    
    result = await orchestrator.run_full_pipeline(
        input_path=args.input,
        output_path=args.output
    )
    
    if result.success:
        print(f"✅ Pipeline completed successfully!")
        print(f"📁 Model saved to: {result.model_path}")
        print(f"📊 Training metrics: {result.metrics}")
    else:
        print(f"❌ Pipeline failed: {result.error}")
        return 1
    
    return 0


async def extract_audio(args):
    """Extract audio from video files."""
    print(f"🎵 Extracting audio: {args.input} → {args.output}")
    
    from pipeline.stages.video_processor import VideoProcessor
    
    processor = VideoProcessor(output_format=args.format)
    result = await processor.process_directory(args.input, args.output)
    
    print(f"✅ Extracted {result.files_processed} audio files")
    return 0


async def transcribe_audio(args):
    """Generate transcripts from audio files."""
    print(f"📝 Transcribing audio: {args.input} → {args.output}")
    
    from pipeline.stages.transcriber import Transcriber
    
    transcriber = Transcriber(
        model_size=args.model,
        speaker_diarization=args.speaker_diarization
    )
    result = await transcriber.process_directory(args.input, args.output)
    
    print(f"✅ Transcribed {result.files_processed} audio files")
    return 0


async def train_model(args):
    """Train a TTS model."""
    print(f"🏋️ Training {args.model} model with dataset: {args.dataset}")
    
    model_trainer = ModelRegistry.get_trainer(args.model)
    
    # Load config and override with CLI args if provided
    config = model_trainer.load_config(args.config)
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    
    result = await model_trainer.train(
        dataset_path=args.dataset,
        output_path=args.output,
        config=config,
        resume_from=args.resume
    )
    
    if result.success:
        print(f"✅ Training completed!")
        print(f"📁 Model saved to: {result.model_path}")
        print(f"📊 Final metrics: {result.final_metrics}")
    else:
        print(f"❌ Training failed: {result.error}")
        return 1
    
    return 0


async def run_inference(args):
    """Test model inference."""
    print(f"🎤 Running inference with model: {args.model}")
    
    model_type = ModelRegistry.detect_model_type(args.model)
    model = ModelRegistry.load_model(model_type, args.model)
    
    result = await model.synthesize(
        text=args.text,
        reference_audio=args.reference,
        output_path=args.output,
        streaming=args.streaming
    )
    
    if result.success:
        print(f"✅ Audio generated: {args.output}")
        print(f"⏱️ Generation time: {result.generation_time:.2f}s")
    else:
        print(f"❌ Inference failed: {result.error}")
        return 1
    
    return 0


async def launch_discord_bot(args):
    """Launch the Discord bot."""
    print(f"🤖 Launching Discord bot with {args.model} model")
    
    from discord_bot.bot import TTSBot
    
    bot = TTSBot(
        model_type=args.model,
        voice_clone_path=args.voice_clone
    )
    
    try:
        await bot.run(args.token)
    except KeyboardInterrupt:
        print("\n👋 Discord bot stopped")
    except Exception as e:
        print(f"❌ Bot error: {e}")
        return 1
    
    return 0


async def validate_data(args):
    """Validate audio/dataset quality."""
    print(f"🔍 Validating {args.type}: {args.input}")
    
    if args.type == "audio":
        from pipeline.validators.audio_quality import AudioQualityValidator
        validator = AudioQualityValidator()
    elif args.type == "dataset":
        from pipeline.validators.dataset_validator import DatasetValidator
        validator = DatasetValidator()
    else:
        from pipeline.validators.transcript_alignment import TranscriptAlignmentValidator
        validator = TranscriptAlignmentValidator()
    
    result = await validator.validate_directory(args.input)
    
    print(f"✅ Validation completed:")
    print(f"   📊 Quality score: {result.overall_score:.2f}/10")
    print(f"   ✅ Passed: {result.passed_files}")
    print(f"   ❌ Failed: {result.failed_files}")
    
    if result.issues:
        print("\n⚠️ Issues found:")
        for issue in result.issues:
            print(f"   - {issue}")
    
    return 0 if result.overall_score >= 7.0 else 1


async def main():
    """Main entry point."""
    parser = setup_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Setup logging
    setup_logging(verbose=args.verbose)
    
    # Route to appropriate handler
    handlers = {
        "run-pipeline": run_pipeline,
        "extract-audio": extract_audio,
        "transcribe": transcribe_audio,
        "train": train_model,
        "inference": run_inference,
        "discord-bot": launch_discord_bot,
        "validate": validate_data,
    }
    
    handler = handlers.get(args.command)
    if not handler:
        print(f"❌ Unknown command: {args.command}")
        return 1
    
    try:
        return await handler(args)
    except KeyboardInterrupt:
        print("\n👋 Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 