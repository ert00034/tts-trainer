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
    
    # Preprocess Audio Command
    preprocess_parser = subparsers.add_parser("preprocess-audio", help="Preprocess audio files for TTS training")
    preprocess_parser.add_argument("--input", required=True, help="Input directory containing audio files")
    preprocess_parser.add_argument("--output", default="resources/audio/processed/", help="Output directory for processed audio")
    preprocess_parser.add_argument("--config", help="Path to audio preprocessing config file")
    preprocess_parser.add_argument("--validate-only", action="store_true", help="Only validate quality, don't process")
    
    # Transcribe Command
    transcribe_parser = subparsers.add_parser("transcribe", help="Generate transcripts from audio files")
    transcribe_parser.add_argument("--input", required=True, help="Input directory containing audio files")
    transcribe_parser.add_argument("--output", default="resources/transcripts/", help="Output directory for transcripts")
    transcribe_parser.add_argument("--model", default="large-v3", help="Whisper model size")
    transcribe_parser.add_argument("--speaker-diarization", action="store_true", help="Enable speaker separation")
    transcribe_parser.add_argument("--device", default="auto", help="Device for processing (auto, cpu, cuda)")
    transcribe_parser.add_argument("--diarization-config", help="Path to speaker diarization config file")
    
    # Speaker Segmentation Command
    segment_parser = subparsers.add_parser("segment-speakers", help="Extract character-specific audio clips")
    segment_parser.add_argument("--audio", required=True, help="Input directory containing original audio files")
    segment_parser.add_argument("--transcripts", required=True, help="Directory containing transcription results")
    segment_parser.add_argument("--output", default="resources/segments/", help="Output directory for character segments")
    segment_parser.add_argument("--config", help="Path to speaker mapping config file")
    segment_parser.add_argument("--analyze-only", action="store_true", help="Only analyze speakers, don't extract segments")
    
    # Speaker Analysis Command
    analyze_parser = subparsers.add_parser("analyze-speakers", help="Analyze speaker patterns in transcripts")
    analyze_parser.add_argument("--transcripts", required=True, help="Directory containing transcription results")
    analyze_parser.add_argument("--output", help="Output file for analysis report")
    
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
    
    from pipeline.stages.audio_extractor import AudioExtractor
    
    extractor = AudioExtractor(output_format=args.format)
    result = await extractor.extract_from_directory(args.input, args.output)
    
    if result['success']:
        print(f"✅ Extracted {result['files_successful']} audio files successfully")
        if result['files_failed'] > 0:
            print(f"⚠️ Failed to extract {result['files_failed']} files")
            if result.get('errors'):
                print("Errors:")
                for error in result['errors'][:5]:  # Show first 5 errors
                    print(f"   - {error}")
    else:
        print(f"❌ Audio extraction failed: {result.get('error', 'Unknown error')}")
        return 1
    
    return 0


async def preprocess_audio(args):
    """Preprocess audio files for TTS training."""
    print(f"🔧 Preprocessing audio: {args.input} → {args.output}")
    
    from pipeline.stages.audio_preprocessor import AudioPreprocessor
    from pipeline.validators.audio_quality import AudioQualityValidator
    
    if args.validate_only:
        # Only validate audio quality
        validator = AudioQualityValidator(args.config) if args.config else AudioQualityValidator()
        result = await validator.validate_directory(args.input)
        
        print(f"📊 Audio Quality Report:")
        print(f"   Overall Score: {result.overall_score:.1f}/10")
        print(f"   Passed Files: {result.passed_files}")
        print(f"   Failed Files: {result.failed_files}")
        
        if result.issues:
            print(f"   Issues found:")
            for issue in result.issues[:10]:  # Show first 10 issues
                print(f"      - {issue}")
        
        if result.overall_score >= 7.0:
            print("✅ Audio quality is acceptable for TTS training")
        else:
            print("⚠️ Audio quality may need improvement")
            
        return 0 if result.overall_score >= 5.0 else 1
    
    else:
        # Process audio files
        preprocessor = AudioPreprocessor(args.config) if args.config else AudioPreprocessor()
        result = await preprocessor.process_directory(args.input, args.output)
        
        if result['success']:
            print(f"✅ Processed {result['files_processed']} audio files successfully")
            if result.get('files_failed', 0) > 0:
                print(f"⚠️ Failed to process {result['files_failed']} files")
                if result.get('errors'):
                    print("Errors:")
                    for error in result['errors'][:5]:  # Show first 5 errors
                        print(f"   - {error}")
            
            validation_score = result.get('validation_score', 0)
            print(f"📊 Processed audio quality score: {validation_score:.1f}/10")
            
            if validation_score >= 7.0:
                print("✅ Processed audio quality is good for TTS training")
            else:
                print("⚠️ Some processed audio may need attention")
                
        else:
            print(f"❌ Audio preprocessing failed: {result.get('error', 'Unknown error')}")
            return 1
        
        return 0


async def transcribe_audio(args):
    """Generate transcripts from audio files."""
    print(f"📝 Transcribing audio: {args.input} → {args.output}")
    
    from pipeline.stages.transcriber import Transcriber
    
    transcriber = Transcriber(
        model_size=args.model,
        speaker_diarization=args.speaker_diarization,
        device=args.device,
        diarization_config=args.diarization_config
    )
    result = await transcriber.process_directory(args.input, args.output)
    
    if result['success']:
        print(f"✅ Processed {result['files_processed']} audio files successfully")
        if result.get('files_failed', 0) > 0:
            print(f"⚠️ Failed to process {result['files_failed']} files")
        
        if args.speaker_diarization:
            speakers = result.get('unique_speakers', [])
            total_segments = result.get('total_segments', 0)
            print(f"🎭 Found {len(speakers)} unique speakers: {speakers}")
            print(f"📊 Extracted {total_segments} speaker segments")
            print(f"💡 Next: Use 'segment-speakers' to extract character-specific clips")
        
        if result.get('errors'):
            print("Errors:")
            for error in result['errors'][:5]:
                print(f"   - {error}")
    else:
        print(f"❌ Transcription failed: {result.get('error', 'Unknown error')}")
        return 1
    
    return 0


async def segment_speakers(args):
    """Extract character-specific audio clips from transcribed episodes."""
    print(f"✂️ Segmenting speakers: {args.audio} + {args.transcripts} → {args.output}")
    
    from pipeline.stages.speaker_segmenter import SpeakerSegmenter
    
    segmenter = SpeakerSegmenter(args.config) if args.config else SpeakerSegmenter()
    
    if args.analyze_only:
        # Only analyze speaker patterns
        result = await segmenter.analyze_speakers(args.transcripts)
        
        if result.get('success'):
            speakers = result.get('speaker_analysis', {})
            print(f"📊 Speaker Analysis Results:")
            print(f"   Found {result.get('speakers_found', 0)} unique speakers")
            
            for speaker_id, data in speakers.items():
                segments = data.get('total_segments', 0)
                duration = data.get('total_duration', 0)
                episodes = len(data.get('episodes', []))
                
                print(f"\n   Speaker: {speaker_id}")
                print(f"      Segments: {segments}")
                print(f"      Duration: {duration:.1f}s ({duration/60:.1f}min)")
                print(f"      Episodes: {episodes}")
                
                sample_texts = data.get('sample_texts', [])
                if sample_texts:
                    print(f"      Sample text: \"{sample_texts[0][:50]}...\"")
            
            print(f"\n💡 Use speaker mapping config to assign character names")
        else:
            print(f"❌ Speaker analysis failed: {result.get('error', 'Unknown error')}")
            return 1
    
    else:
        # Extract character segments
        result = await segmenter.process_directory(args.audio, args.transcripts, args.output)
        
        if result['success']:
            print(f"✅ Processed {result['episodes_processed']} episodes successfully")
            print(f"📊 Extracted {result['total_segments']} character segments")
            
            character_stats = result.get('character_stats', {})
            if character_stats:
                print(f"\n🎭 Character breakdown:")
                for character, stats in character_stats.items():
                    segments = stats.get('segments', 0)
                    duration = stats.get('duration', 0)
                    print(f"   {character}: {segments} segments ({duration/60:.1f} min)")
            
            if result.get('episodes_failed', 0) > 0:
                print(f"⚠️ Failed to process {result['episodes_failed']} episodes")
            
            print(f"💡 Next: Use 'preprocess-audio' on character segments for TTS training")
        else:
            print(f"❌ Speaker segmentation failed: {result.get('error', 'Unknown error')}")
            return 1
    
    return 0


async def analyze_speakers(args):
    """Analyze speaker patterns in transcripts."""
    print(f"🔍 Analyzing speakers in: {args.transcripts}")
    
    from pipeline.stages.speaker_segmenter import SpeakerSegmenter
    
    segmenter = SpeakerSegmenter()
    result = await segmenter.analyze_speakers(args.transcripts)
    
    if result.get('success'):
        speakers = result.get('speaker_analysis', {})
        print(f"📊 Found {result.get('speakers_found', 0)} unique speakers")
        
        # Sort speakers by total duration
        sorted_speakers = sorted(
            speakers.items(), 
            key=lambda x: x[1].get('total_duration', 0), 
            reverse=True
        )
        
        for speaker_id, data in sorted_speakers:
            segments = data.get('total_segments', 0)
            duration = data.get('total_duration', 0)
            episodes = len(data.get('episodes', []))
            
            print(f"\n🎭 Speaker: {speaker_id}")
            print(f"    Segments: {segments}")
            print(f"    Duration: {duration:.1f}s ({duration/60:.1f}min)")
            print(f"    Episodes: {episodes}")
            
            sample_texts = data.get('sample_texts', [])
            if sample_texts:
                print(f"    Sample quotes:")
                for i, text in enumerate(sample_texts[:3]):
                    print(f"      {i+1}. \"{text[:60]}...\"")
        
        # Save detailed report if output specified
        if args.output:
            import json
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\n📄 Detailed report saved to: {args.output}")
        
        print(f"\n💡 Use this analysis to update speaker mapping config")
    else:
        print(f"❌ Speaker analysis failed: {result.get('error', 'Unknown error')}")
        return 1
    
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
        "preprocess-audio": preprocess_audio,
        "transcribe": transcribe_audio,
        "segment-speakers": segment_speakers,
        "analyze-speakers": analyze_speakers,
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