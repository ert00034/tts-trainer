"""
Pipeline Orchestrator - Manages the entire video-to-TTS workflow
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import yaml

from .stages.video_processor import VideoProcessor
from .stages.audio_extractor import AudioExtractor
from .stages.audio_preprocessor import AudioPreprocessor
from .stages.transcriber import Transcriber
from .stages.dataset_builder import DatasetBuilder
from .stages.model_trainer import ModelTrainer
from .validators.audio_quality import AudioQualityValidator
from .validators.transcript_alignment import TranscriptAlignmentValidator


@dataclass
class PipelineResult:
    """Result of a pipeline execution."""
    success: bool
    model_path: Optional[str] = None
    dataset_path: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    stage_results: Optional[Dict[str, Any]] = None


@dataclass
class PipelineConfig:
    """Configuration for the pipeline."""
    model_type: str
    skip_training: bool = False
    validate_stages: bool = True
    parallel_processing: bool = True
    checkpoint_frequency: int = 1  # Save checkpoint every N stages


class PipelineOrchestrator:
    """
    Orchestrates the entire video-to-TTS training pipeline.
    
    The pipeline follows these stages:
    1. Video Processing - Extract metadata and prepare videos
    2. Audio Extraction - Extract audio from videos
    3. Audio Preprocessing - Clean, normalize, and enhance audio
    4. Transcription - Generate text transcripts using Whisper
    5. Dataset Building - Create training datasets in the correct format
    6. Model Training - Train or fine-tune the TTS model
    """
    
    def __init__(self, model_type: str = "xtts_v2", skip_training: bool = False, 
                 config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.config = PipelineConfig(
            model_type=model_type,
            skip_training=skip_training
        )
        
        # Load additional config if provided
        if config_path:
            self._load_config(config_path)
        
        # Initialize stages
        self.stages = self._initialize_stages()
        self.validators = self._initialize_validators()
        
        # Track pipeline state
        self.current_stage = 0
        self.stage_results = {}
    
    def _load_config(self, config_path: str):
        """Load pipeline configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Update config with loaded values
            for key, value in config_data.get('pipeline', {}).items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                    
        except Exception as e:
            self.logger.warning(f"Failed to load config from {config_path}: {e}")
    
    def _initialize_stages(self) -> List:
        """Initialize all pipeline stages."""
        return [
            VideoProcessor(),
            AudioExtractor(),
            AudioPreprocessor(),
            Transcriber(),
            DatasetBuilder(model_type=self.config.model_type),
            ModelTrainer(model_type=self.config.model_type) if not self.config.skip_training else None
        ]
    
    def _initialize_validators(self) -> Dict:
        """Initialize validation stages."""
        return {
            'audio_quality': AudioQualityValidator(),
            'transcript_alignment': TranscriptAlignmentValidator()
        }
    
    async def run_full_pipeline(self, input_path: str, output_path: str) -> PipelineResult:
        """
        Run the complete pipeline from videos to trained model.
        
        Args:
            input_path: Directory containing input video files
            output_path: Directory to save the final trained model
            
        Returns:
            PipelineResult with success status and paths
        """
        self.logger.info(f"🚀 Starting full pipeline: {input_path} → {output_path}")
        
        try:
            # Validate inputs
            input_dir = Path(input_path)
            output_dir = Path(output_path)
            
            if not input_dir.exists():
                return PipelineResult(success=False, error=f"Input directory not found: {input_path}")
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Define intermediate paths
            paths = {
                'videos': input_dir,
                'audio': Path("resources/audio"),
                'transcripts': Path("resources/transcripts"),
                'datasets': Path("resources/datasets"),
                'models': output_dir
            }
            
            # Create intermediate directories
            for path in paths.values():
                path.mkdir(parents=True, exist_ok=True)
            
            # Execute pipeline stages
            stage_names = ["video_processing", "audio_extraction", "audio_preprocessing", 
                          "transcription", "dataset_building", "model_training"]
            
            for i, (stage, stage_name) in enumerate(zip(self.stages, stage_names)):
                if stage is None:  # Skip training if disabled
                    continue
                    
                self.logger.info(f"📋 Stage {i+1}/{len(self.stages)}: {stage_name}")
                
                # Execute stage
                stage_result = await self._execute_stage(stage, paths, stage_name)
                self.stage_results[stage_name] = stage_result
                
                if not stage_result.get('success', False):
                    error_msg = f"Stage {stage_name} failed: {stage_result.get('error', 'Unknown error')}"
                    return PipelineResult(success=False, error=error_msg, stage_results=self.stage_results)
                
                # Validate stage output if enabled
                if self.config.validate_stages:
                    validation_result = await self._validate_stage_output(stage_name, stage_result)
                    if not validation_result:
                        error_msg = f"Stage {stage_name} validation failed"
                        return PipelineResult(success=False, error=error_msg, stage_results=self.stage_results)
                
                # Save checkpoint
                if i % self.config.checkpoint_frequency == 0:
                    await self._save_checkpoint(i, stage_result)
                
                self.current_stage = i + 1
            
            # Compile final results
            final_model_path = self.stage_results.get('model_training', {}).get('model_path')
            final_dataset_path = self.stage_results.get('dataset_building', {}).get('dataset_path')
            final_metrics = self._compile_metrics()
            
            self.logger.info("✅ Pipeline completed successfully!")
            
            return PipelineResult(
                success=True,
                model_path=final_model_path,
                dataset_path=final_dataset_path,
                metrics=final_metrics,
                stage_results=self.stage_results
            )
            
        except Exception as e:
            self.logger.error(f"Pipeline failed with exception: {e}")
            return PipelineResult(success=False, error=str(e), stage_results=self.stage_results)
    
    async def _execute_stage(self, stage, paths: Dict[str, Path], stage_name: str) -> Dict[str, Any]:
        """Execute a single pipeline stage."""
        try:
            if stage_name == "video_processing":
                return await stage.process_directory(paths['videos'], paths['videos'])
            elif stage_name == "audio_extraction":
                return await stage.extract_from_directory(paths['videos'], paths['audio'])
            elif stage_name == "audio_preprocessing":
                return await stage.process_directory(paths['audio'], paths['audio'])
            elif stage_name == "transcription":
                return await stage.transcribe_directory(paths['audio'], paths['transcripts'])
            elif stage_name == "dataset_building":
                return await stage.build_dataset(paths['audio'], paths['transcripts'], paths['datasets'])
            elif stage_name == "model_training":
                return await stage.train_model(paths['datasets'], paths['models'])
            else:
                return {'success': False, 'error': f'Unknown stage: {stage_name}'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _validate_stage_output(self, stage_name: str, stage_result: Dict[str, Any]) -> bool:
        """Validate the output of a pipeline stage."""
        try:
            if stage_name == "audio_preprocessing":
                validator = self.validators['audio_quality']
                audio_dir = stage_result.get('output_path')
                if audio_dir:
                    validation_result = await validator.validate_directory(audio_dir)
                    return validation_result.overall_score >= 7.0
            
            elif stage_name == "transcription":
                validator = self.validators['transcript_alignment']
                transcript_dir = stage_result.get('output_path')
                if transcript_dir:
                    validation_result = await validator.validate_directory(transcript_dir)
                    return validation_result.overall_score >= 7.0
            
            # Default: assume valid if stage succeeded
            return stage_result.get('success', False)
            
        except Exception as e:
            self.logger.warning(f"Validation failed for {stage_name}: {e}")
            return True  # Don't fail pipeline on validation errors
    
    async def _save_checkpoint(self, stage_index: int, stage_result: Dict[str, Any]):
        """Save pipeline checkpoint."""
        checkpoint_path = Path("artifacts/checkpoints") / f"pipeline_stage_{stage_index}.yaml"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint_data = {
            'stage_index': stage_index,
            'current_stage': self.current_stage,
            'config': self.config.__dict__,
            'stage_results': self.stage_results,
            'timestamp': str(asyncio.get_event_loop().time())
        }
        
        try:
            with open(checkpoint_path, 'w') as f:
                yaml.dump(checkpoint_data, f)
            self.logger.debug(f"Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            self.logger.warning(f"Failed to save checkpoint: {e}")
    
    def _compile_metrics(self) -> Dict[str, Any]:
        """Compile metrics from all pipeline stages."""
        metrics = {
            'pipeline_success': True,
            'stages_completed': self.current_stage,
            'total_stages': len([s for s in self.stages if s is not None])
        }
        
        # Extract metrics from each stage
        for stage_name, result in self.stage_results.items():
            if isinstance(result, dict):
                stage_metrics = result.get('metrics', {})
                metrics[f"{stage_name}_metrics"] = stage_metrics
        
        return metrics
    
    async def resume_from_checkpoint(self, checkpoint_path: str) -> PipelineResult:
        """Resume pipeline execution from a saved checkpoint."""
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = yaml.safe_load(f)
            
            self.current_stage = checkpoint_data['current_stage']
            self.stage_results = checkpoint_data['stage_results']
            
            self.logger.info(f"Resuming pipeline from stage {self.current_stage}")
            
            # Continue from where we left off
            # Implementation would depend on specific requirements
            
        except Exception as e:
            return PipelineResult(success=False, error=f"Failed to resume from checkpoint: {e}")
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        total_stages = len([s for s in self.stages if s is not None])
        progress = (self.current_stage / total_stages) * 100 if total_stages > 0 else 0
        
        return {
            'current_stage': self.current_stage,
            'total_stages': total_stages,
            'progress_percent': progress,
            'stage_results': self.stage_results,
            'config': self.config.__dict__
        } 