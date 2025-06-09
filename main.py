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
import click
import json
import shutil

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.pipeline.orchestrator import PipelineOrchestrator
from src.models.base_trainer import ModelRegistry
from src.utils.logging_utils import setup_logging
from src.utils.file_utils import validate_input_path, create_output_dir


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
    
    # Process Segments Command
    process_segments_parser = subparsers.add_parser("process-segments", help="Improve quality of audio segments for TTS training")
    process_segments_parser.add_argument("--input", required=True, help="Input directory containing speaker segments")
    process_segments_parser.add_argument("--output", default="resources/segments_processed/", help="Output directory for processed segments")
    process_segments_parser.add_argument("--config", help="Path to segment processing config file")
    
    # Voice Clustering Command
    clustering_parser = subparsers.add_parser("cluster-voices", help="Group similar voices across episodes for consistent speaker IDs")
    clustering_parser.add_argument("--segments", required=True, help="Input directory containing speaker segments")
    clustering_parser.add_argument("--output", default="resources/voice_clusters/", help="Output directory for clustering results")
    clustering_parser.add_argument("--config", help="Path to voice clustering config file")
    clustering_parser.add_argument("--apply", help="Apply existing clustering results to reorganize segments by character")
    
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
    inference_parser.add_argument("--character", help="Character name for voice cloning (when using character profiles)")
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
    
    # Create Validation Samples Command
    validation_samples_parser = subparsers.add_parser("create-validation-samples", 
                                                       help="Create validation samples for manual review of clustering quality")
    validation_samples_parser.add_argument("--clustering-results", required=True, 
                                          help="Path to voice clustering results JSON file")
    validation_samples_parser.add_argument("--output", required=True, 
                                          help="Output directory for validation samples")
    validation_samples_parser.add_argument("--samples-per-cluster", type=int, default=3, 
                                          help="Number of samples per cluster to copy")
    
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
    
    from src.pipeline.stages.audio_extractor import AudioExtractor
    
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
    
    from src.pipeline.stages.audio_preprocessor import AudioPreprocessor
    from src.pipeline.validators.audio_quality import AudioQualityValidator
    
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
    
    from src.pipeline.stages.transcriber import Transcriber
    
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
    
    from src.pipeline.stages.speaker_segmenter import SpeakerSegmenter
    
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
    
    from src.pipeline.stages.speaker_segmenter import SpeakerSegmenter
    
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
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\n📄 Detailed report saved to: {args.output}")
        
        print(f"\n💡 Use this analysis to update speaker mapping config")
    else:
        print(f"❌ Speaker analysis failed: {result.get('error', 'Unknown error')}")
        return 1
    
    return 0


async def process_segments(args):
    """Process audio segments to improve quality for TTS training."""
    print(f"🔧 Processing audio segments: {args.input} → {args.output}")
    
    from src.pipeline.stages.audio_segment_processor import AudioSegmentProcessor
    
    processor = AudioSegmentProcessor(args.config) if args.config else AudioSegmentProcessor()
    result = await processor.process_segments(args.input, args.output)
    
    if result['success']:
        print(f"✅ Processing completed!")
        print(f"   📊 Speakers processed: {result['speakers_processed']}")
        print(f"   🎯 Segments accepted: {result['segments_accepted']}/{result['segments_input']}")
        print(f"   📉 Rejection rate: {result['rejection_rate']:.1%}")
        print(f"   ⏱️ Total duration: {result['total_duration_hours']:.1f} hours")
        
        if result.get('speakers_failed', 0) > 0:
            print(f"   ⚠️ Failed speakers: {result['speakers_failed']}")
        
        # Show common rejection reasons
        if result['rejection_rate'] > 0.1:  # If >10% rejected
            print(f"\n💡 Consider adjusting quality thresholds based on your data")
    else:
        print(f"❌ Segment processing failed: {result.get('error', 'Unknown error')}")
        return 1
    
    return 0


async def cluster_voices(args):
    """Perform cross-episode voice clustering or apply existing clustering results."""
    from src.pipeline.stages.voice_clustering import VoiceClusteringSystem
    
    if args.apply:
        # Apply existing clustering results to reorganize segments
        print(f"🔄 Applying clustering results: {args.apply}")
        print(f"📁 Reorganizing segments: {args.segments} → {args.output}")
        
        clustering_system = VoiceClusteringSystem()
        result = await clustering_system.apply_clustering_to_segments(
            clustering_results_path=args.apply,
            segments_dir=args.segments,
            output_dir=args.output
        )
        
        if result['success']:
            print(f"✅ Segments reorganized successfully!")
            print(f"📊 Statistics:")
            print(f"   Segments reorganized: {result['segments_reorganized']}")
            print(f"   Characters created: {result['characters_created']}")
            print(f"   Failed segments: {result['segments_failed']}")
            print(f"📁 Character segments saved to: {result['output_path']}")
        else:
            print(f"❌ Failed to apply clustering: {result.get('error', 'Unknown error')}")
            return 1
    else:
        # Perform new voice clustering analysis
        print(f"🎯 Analyzing voices for clustering: {args.segments}")
        print(f"📁 Clustering results will be saved to: {args.output}")
        
        # Load config if provided
        config = None
        if args.config:
            import yaml
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
        
        clustering_system = VoiceClusteringSystem(config)
        result = await clustering_system.cluster_voices(args.segments, args.output)
        
        if result['success']:
            print(f"✅ Voice clustering completed successfully!")
            print(f"📊 Clustering Results:")
            print(f"   Total voice embeddings: {result['total_embeddings']}")
            print(f"   Clusters found: {result['clusters_found']}")
            print(f"📁 Results saved to: {result['output_path']}")
            
            # Show cluster assignments
            if 'cluster_assignments' in result:
                print(f"🎭 Character Assignments:")
                for cluster_id, character in result['cluster_assignments'].items():
                    print(f"   Cluster {cluster_id} → {character}")
            
            # Show recommendations if available
            if 'recommendations' in result:
                print(f"💡 Recommendations:")
                for rec in result['recommendations'][:3]:  # Show first 3
                    print(f"   • {rec}")
            
            print(f"\n🔄 To apply these results, run:")
            print(f"   python main.py cluster-voices --segments {args.segments} --apply {args.output}/voice_clustering_results.json --output resources/characters/")
        else:
            print(f"❌ Voice clustering failed: {result.get('error', 'Unknown error')}")
            return 1
    
    return 0


async def train_model(args):
    """Train/Setup a TTS model with character voice profiles."""
    print(f"🏋️ Training {args.model} model with dataset: {args.dataset}")
    
    try:
        from src.models.base_trainer import ModelRegistry
        
        model_trainer = ModelRegistry.get_trainer(args.model)
        
        # Load config and override with CLI args if provided
        config = model_trainer.load_config(args.config)
        if hasattr(config, 'training') and args.epochs:
            config['training']['epochs'] = args.epochs
        if hasattr(config, 'training') and args.batch_size:
            config['training']['batch_size'] = args.batch_size
        
        # For XTTS v2, this creates character voice profiles
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
            
            # Show character-specific results
            character_metrics = result.final_metrics.get('character_metrics', {})
            if character_metrics:
                print(f"\n🎭 Character Voice Profiles Created:")
                for character, metrics in character_metrics.items():
                    speed = metrics.get('synthesis_speed', 0)
                    print(f"   {character}: ⏱️ {speed:.2f}s synthesis test")
                
                print(f"\n💡 Test voices with:")
                for character in character_metrics.keys():
                    print(f"   python main.py inference --model {args.output}/character_voice_profiles.json --character {character} --text \"Hello, I'm {character}\"")
        else:
            error_msg = getattr(result, 'error', 'Unknown error')
            print(f"❌ Training failed: {error_msg}")
            return 1
        
    except Exception as e:
        print(f"❌ Training failed with exception: {e}")
        return 1
    
    return 0


async def run_inference(args):
    """Test model inference with character voice cloning."""
    print(f"🎤 Running inference with model: {args.model}")
    
    from src.models.base_trainer import ModelRegistry
    
    try:
        model_type = ModelRegistry.detect_model_type(args.model)
        model = ModelRegistry.load_model(model_type, args.model)
        
        # Load character voice profiles if available
        if hasattr(model, 'load_character_voices') and args.model.endswith('character_voice_profiles.json'):
            model.load_character_voices(args.model)
            available_characters = model.list_available_characters()
            
            if args.character and args.character not in available_characters:
                print(f"❌ Character '{args.character}' not found. Available: {available_characters}")
                return 1
            elif not args.character and not args.reference:
                print(f"Available characters: {available_characters}")
                print(f"Use --character <name> or --reference <audio_file>")
                return 1
        
        # Display inference info
        if args.character:
            print(f"🎭 Using character voice: {args.character}")
        elif args.reference:
            print(f"🎵 Using reference audio: {args.reference}")
        
        result = await model.synthesize(
            text=args.text,
            reference_audio=args.reference,
            character=args.character,
            output_path=args.output,
            streaming=args.streaming
        )
        
        if result.success:
            print(f"✅ Audio generated: {args.output}")
            print(f"⏱️ Generation time: {result.generation_time:.2f}s")
            
            # Provide playback suggestion
            import platform
            if platform.system() == "Windows":
                print(f"🔊 Play with: start {args.output}")
            elif platform.system() == "Darwin":  # macOS
                print(f"🔊 Play with: open {args.output}")
            else:  # Linux
                print(f"🔊 Play with: xdg-open {args.output}")
        else:
            error_msg = getattr(result, 'error', 'Unknown error')
            print(f"❌ Inference failed: {error_msg}")
            return 1
            
    except Exception as e:
        print(f"❌ Inference failed with exception: {e}")
        return 1
    
    return 0


async def launch_discord_bot(args):
    """Launch the Discord bot."""
    print(f"🤖 Launching Discord bot with {args.model} model")
    
    from src.discord_bot.bot import TTSBot
    
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
        from src.pipeline.validators.audio_quality import AudioQualityValidator
        validator = AudioQualityValidator()
    elif args.type == "dataset":
        from src.pipeline.validators.dataset_validator import DatasetValidator
        validator = DatasetValidator()
    else:
        from src.pipeline.validators.transcript_alignment import TranscriptAlignmentValidator
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


async def create_validation_samples(args):
    """Create validation samples for manual review of clustering quality."""
    # Load clustering results
    with open(args.clustering_results, 'r') as f:
        data = json.load(f)
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear existing validation samples
    for item in output_dir.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
    
    print(f"🎧 Creating validation samples in: {output_dir}")
    
    # Load all transcript files for full transcript generation
    transcripts_dir = Path("resources/transcripts/transcripts")
    all_transcripts = {}
    
    if transcripts_dir.exists():
        print("📝 Loading transcript files for full transcript generation...")
        for transcript_file in transcripts_dir.glob("*.json"):
            try:
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    transcript_data = json.load(f)
                    all_transcripts[transcript_file.stem] = transcript_data
            except Exception as e:
                print(f"  ⚠️  Failed to load {transcript_file}: {e}")
    
    # Process each cluster
    for cluster in data['clusters']:
        cluster_id = cluster['cluster_id']
        character = cluster.get('assigned_character', f'cluster_{cluster_id}')
        duration = cluster['total_duration'] / 60
        episodes = len(cluster['episodes'])
        
        # Create character directory
        char_dir = output_dir / f"{cluster_id:02d}_{character}"
        char_dir.mkdir(exist_ok=True)
        
        # Copy representative samples
        samples = cluster.get('representative_samples', [])[:args.samples_per_cluster]
        
        print(f"📁 {character}: {duration:.1f}min, {episodes} episodes, copying {len(samples)} samples")
        
        for i, sample_path in enumerate(samples):
            if Path(sample_path).exists():
                # Create descriptive filename
                original_name = Path(sample_path).name
                new_name = f"{i+1:02d}_{original_name}"
                dest_path = char_dir / new_name
                
                try:
                    shutil.copy2(sample_path, dest_path)
                except Exception as e:
                    print(f"  ⚠️  Failed to copy {sample_path}: {e}")
            else:
                print(f"  ⚠️  Sample not found: {sample_path}")
        
        # Generate full transcript for this cluster
        cluster_transcript_file = char_dir / "full_transcript.txt"
        _generate_cluster_transcript(cluster, data['speaker_mapping'], all_transcripts, cluster_transcript_file)
        
        # Create a summary file for this cluster
        summary_file = char_dir / "cluster_info.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Cluster ID: {cluster_id}\n")
            f.write(f"Character: {character}\n")
            f.write(f"Duration: {duration:.1f} minutes\n")
            f.write(f"Episodes: {episodes}\n")
            f.write(f"Segments: {cluster['segment_count']}\n")
            f.write(f"Quality Score: {cluster.get('quality_score', 'N/A')}\n")
            f.write(f"Original Speakers: {', '.join(cluster.get('original_speakers', []))}\n")
            f.write(f"\nFirst few episodes:\n")
            for ep in cluster['episodes'][:5]:
                f.write(f"  - {ep}\n")
            f.write(f"\nFiles in this directory:\n")
            f.write(f"  - full_transcript.txt: Complete transcript of all dialogue\n")
            f.write(f"  - 01_*.wav to 0{len(samples)}_*.wav: Representative audio samples\n")
    
    print(f"\n✅ Validation samples created in: {output_dir}")
    print("📝 Each cluster has a 'cluster_info.txt' file with details")
    print("📄 Each cluster has a 'full_transcript.txt' with all dialogue")
    print("🎵 Audio files are numbered for easy sequential listening")


def _generate_cluster_transcript(cluster, speaker_mapping, all_transcripts, output_file):
    """Generate a full transcript file for a cluster."""
    try:
        cluster_id = cluster['cluster_id']
        original_speakers = set(cluster.get('original_speakers', []))
        
        # Collect all dialogue from this cluster
        all_dialogue = []
        
        # Iterate through episodes and find segments from speakers in this cluster
        for episode_key, episode_speakers in speaker_mapping.items():
            # Extract clean episode name (remove _SPEAKER suffix if present)
            episode_name = episode_key.replace('_SPEAKER', '')
            
            # Check if we have transcript data for this episode
            if episode_name not in all_transcripts:
                continue
            
            transcript_data = all_transcripts[episode_name]
            
            # Find speakers in this episode that belong to our cluster
            cluster_speakers_in_episode = set()
            for speaker_id, mapped_cluster_id in episode_speakers.items():
                if mapped_cluster_id == cluster_id:
                    cluster_speakers_in_episode.add(speaker_id)
            
            if not cluster_speakers_in_episode:
                continue
            
            # Extract dialogue from these speakers
            episode_dialogue = []
            for segment in transcript_data.get('segments', []):
                speaker = segment.get('speaker')
                if speaker in cluster_speakers_in_episode:
                    text = segment.get('text', '').strip()
                    start_time = segment.get('start', 0)
                    if text:
                        episode_dialogue.append({
                            'episode': episode_name,
                            'speaker': speaker,
                            'time': start_time,
                            'text': text
                        })
            
            # Sort by time and add to overall collection
            episode_dialogue.sort(key=lambda x: x['time'])
            all_dialogue.extend(episode_dialogue)
        
        # Write the transcript file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"FULL TRANSCRIPT - Cluster {cluster_id}\n")
            f.write(f"Character: {cluster.get('assigned_character', 'unassigned')}\n")
            f.write(f"Total Duration: {cluster['total_duration'] / 60:.1f} minutes\n")
            f.write(f"Original Speakers: {', '.join(original_speakers)}\n")
            f.write(f"Episodes: {len(cluster['episodes'])}\n")
            f.write("=" * 60 + "\n\n")
            
            if not all_dialogue:
                f.write("No dialogue found for this cluster.\n")
                return
            
            # Group by episode for better readability
            current_episode = None
            for dialogue in all_dialogue:
                if dialogue['episode'] != current_episode:
                    current_episode = dialogue['episode']
                    f.write(f"\n--- {current_episode} ---\n\n")
                
                # Format: [MM:SS] SPEAKER_XX: "dialogue"
                minutes = int(dialogue['time'] // 60)
                seconds = int(dialogue['time'] % 60)
                f.write(f"[{minutes:02d}:{seconds:02d}] {dialogue['speaker']}: \"{dialogue['text']}\"\n")
            
            # Add summary statistics
            f.write(f"\n" + "=" * 60 + "\n")
            f.write(f"SUMMARY:\n")
            f.write(f"Total lines: {len(all_dialogue)}\n")
            
            # Count by episode
            episode_counts = {}
            for dialogue in all_dialogue:
                episode_counts[dialogue['episode']] = episode_counts.get(dialogue['episode'], 0) + 1
            
            f.write(f"Lines by episode:\n")
            for episode, count in sorted(episode_counts.items()):
                f.write(f"  {episode}: {count} lines\n")
        
        print(f"  📄 Generated full transcript: {len(all_dialogue)} dialogue lines")
        
    except Exception as e:
        print(f"  ⚠️  Failed to generate transcript: {e}")
        # Create a minimal file indicating the error
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Error generating transcript for cluster {cluster['cluster_id']}: {e}\n")


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
        "process-segments": process_segments,
        "cluster-voices": cluster_voices,
        "train": train_model,
        "inference": run_inference,
        "discord-bot": launch_discord_bot,
        "validate": validate_data,
        "create-validation-samples": create_validation_samples,
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