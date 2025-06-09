"""
Voice Clustering Pipeline Stage
Groups similar voices across episodes to solve speaker consistency problems
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
import json
import numpy as np
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import pickle
from collections import defaultdict
import re

# For voice embeddings and clustering
try:
    from speechbrain.pretrained import EncoderClassifier
    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    SPEECHBRAIN_AVAILABLE = False
    logging.warning("SpeechBrain not available - will use alternative embedding method")

from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import librosa
import soundfile as sf

logger = logging.getLogger(__name__)


@dataclass
class VoiceEmbedding:
    """Voice embedding with metadata."""
    speaker_id: str
    episode: str
    segment_file: str
    embedding: np.ndarray
    duration: float
    quality_score: float
    text: str


@dataclass
class VoiceCluster:
    """A cluster of similar voices."""
    cluster_id: int
    embeddings: List[VoiceEmbedding]
    centroid: np.ndarray
    total_duration: float
    episodes: Set[str]
    original_speakers: Set[str]
    quality_score: float
    representative_samples: List[str]


@dataclass
class ClusteringResult:
    """Result of voice clustering analysis."""
    clusters: List[VoiceCluster]
    speaker_mapping: Dict[str, Dict[str, int]]  # episode -> speaker_id -> cluster_id
    cluster_assignments: Dict[int, str]  # cluster_id -> suggested_character_name
    quality_metrics: Dict[str, float]
    recommendations: List[str]


def convert_numpy_types(obj):
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        # Convert both keys and values, handling numpy types in keys
        converted_dict = {}
        for k, v in obj.items():
            # Convert numpy keys to regular Python types
            if isinstance(k, np.integer):
                k = int(k)
            elif isinstance(k, np.floating):
                k = float(k)
            converted_dict[k] = convert_numpy_types(v)
        return converted_dict
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, set):
        return list(obj)
    else:
        return obj


class VoiceClusteringSystem:
    """Advanced voice clustering system for cross-episode speaker consistency."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize voice clustering system."""
        self.config = config or self._get_default_config()
        self.embedding_model = None
        self._initialize_embedding_model()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for voice clustering."""
        return {
            'embedding': {
                'model': 'speechbrain/spkrec-ecapa-voxceleb',  # Speaker verification model
                'device': 'cuda',
                'normalize': True,
                'max_audio_length': 10.0,  # Max seconds per segment for embedding
                'min_audio_length': 1.0,   # Min seconds per segment
            },
            'clustering': {
                'algorithm': 'dbscan',  # 'dbscan', 'hierarchical', 'kmeans'
                'eps': 0.3,             # DBSCAN epsilon (distance threshold)
                'min_samples': 3,       # DBSCAN minimum samples per cluster
                'n_clusters': None,     # Number of clusters (for hierarchical/kmeans)
                'linkage': 'ward',      # Linkage method for hierarchical clustering
                'distance_metric': 'cosine',
            },
            'quality': {
                'min_cluster_duration': 30.0,    # Minimum total duration per cluster (seconds)
                'min_episodes_per_cluster': 3,   # Minimum episodes per cluster
                'min_segments_per_cluster': 10,  # Minimum segments per cluster
                'purity_threshold': 0.7,         # Cluster purity threshold
            },
            'character_assignment': {
                'use_text_analysis': True,      # Use text content for character hints
                'pokemon_characters': [         # Expected Pokemon characters
                    'ash', 'misty', 'brock', 'jessie', 'james', 'meowth',
                    'narrator', 'pokédex', 'professor_oak'
                ],
                'min_confidence': 0.6,          # Minimum confidence for auto-assignment
            }
        }
    
    def _initialize_embedding_model(self):
        """Initialize the voice embedding model."""
        if not SPEECHBRAIN_AVAILABLE:
            logger.warning("SpeechBrain not available - using fallback embedding method")
            return
        
        try:
            model_name = self.config['embedding']['model']
            device = self.config['embedding']['device']
            
            logger.info(f"Loading voice embedding model: {model_name}")
            self.embedding_model = EncoderClassifier.from_hparams(
                source=model_name,
                run_opts={"device": device}
            )
            logger.info("Voice embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.embedding_model = None
    
    async def cluster_voices(self, segments_dir: str, output_dir: str) -> Dict[str, Any]:
        """
        Perform cross-episode voice clustering analysis.
        
        Args:
            segments_dir: Directory containing speaker segments
            output_dir: Directory to save clustering results
        """
        logger.info(f"Starting voice clustering analysis: {segments_dir} → {output_dir}")
        
        segments_path = Path(segments_dir)
        output_path = Path(output_dir)
        
        if not segments_path.exists():
            return {
                'success': False,
                'error': f"Segments directory not found: {segments_dir}",
                'clusters_found': 0
            }
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Extract voice embeddings from all segments
        logger.info("Step 1: Extracting voice embeddings...")
        embeddings = await self._extract_embeddings(segments_path)
        
        if len(embeddings) == 0:
            return {
                'success': False,
                'error': 'No valid voice embeddings extracted',
                'clusters_found': 0
            }
        
        logger.info(f"Extracted {len(embeddings)} voice embeddings")
        
        # Step 2: Perform clustering
        logger.info("Step 2: Performing voice clustering...")
        clustering_result = self._perform_clustering(embeddings)
        
        # Step 3: Analyze and validate clusters
        logger.info("Step 3: Analyzing cluster quality...")
        self._analyze_cluster_quality(clustering_result)
        
        # Step 4: Generate character assignments
        logger.info("Step 4: Generating character assignments...")
        self._assign_characters(clustering_result)
        
        # Step 5: Save results
        logger.info("Step 5: Saving clustering results...")
        await self._save_results(clustering_result, output_path)
        
        return {
            'success': True,
            'clusters_found': len(clustering_result.clusters),
            'total_embeddings': len(embeddings),
            'speaker_mapping': clustering_result.speaker_mapping,
            'cluster_assignments': clustering_result.cluster_assignments,
            'quality_metrics': clustering_result.quality_metrics,
            'output_path': str(output_path),
            'recommendations': clustering_result.recommendations
        }
    
    async def _extract_embeddings(self, segments_path: Path) -> List[VoiceEmbedding]:
        """Extract voice embeddings from all speaker segments."""
        embeddings = []
        
        # Find all speaker directories
        speaker_dirs = [d for d in segments_path.iterdir() if d.is_dir()]
        
        # Process in parallel with limited workers to avoid memory issues
        max_workers = 2
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            tasks = [
                asyncio.get_event_loop().run_in_executor(
                    executor, self._extract_speaker_embeddings, speaker_dir
                )
                for speaker_dir in speaker_dirs
            ]
            
            results = await asyncio.gather(*tasks)
            
            # Flatten results
            for speaker_embeddings in results:
                embeddings.extend(speaker_embeddings)
        
        return embeddings
    
    def _extract_speaker_embeddings(self, speaker_dir: Path) -> List[VoiceEmbedding]:
        """Extract embeddings for all segments of a specific speaker."""
        speaker_id = speaker_dir.name
        embeddings = []
        
        # Find all audio files
        audio_files = []
        for ext in ['.wav', '.mp3', '.flac']:
            audio_files.extend(speaker_dir.glob(f"*{ext}"))
        
        logger.debug(f"Processing {len(audio_files)} segments for speaker {speaker_id}")
        
        for audio_file in audio_files:
            try:
                # Load metadata - handle different naming patterns
                # Pattern 1: filename_processed.wav -> filename_processed.json
                # Pattern 2: filename.wav -> filename.wav.json
                metadata_file = audio_file.with_suffix('.json')
                if not metadata_file.exists():
                    metadata_file = audio_file.with_suffix(audio_file.suffix + '.json')
                    if not metadata_file.exists():
                        logger.debug(f"No metadata file found for {audio_file.name}")
                        continue
                
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # Extract episode name from filename
                episode = self._extract_episode_name(audio_file.name)
                
                # Load and process audio
                embedding = self._extract_single_embedding(audio_file, metadata)
                
                if embedding is not None:
                    voice_embedding = VoiceEmbedding(
                        speaker_id=speaker_id,
                        episode=episode,
                        segment_file=str(audio_file),
                        embedding=embedding,
                        duration=metadata.get('duration', 0),
                        quality_score=metadata.get('confidence', 1.0),
                        text=metadata.get('text', '')
                    )
                    embeddings.append(voice_embedding)
                    
            except Exception as e:
                logger.debug(f"Failed to process {audio_file.name}: {e}")
                continue
        
        logger.debug(f"Extracted {len(embeddings)} embeddings for speaker {speaker_id}")
        return embeddings
    
    def _extract_episode_name(self, filename: str) -> str:
        """Extract episode name from segment filename."""
        # Handle different naming patterns:
        # "Pokemon S01E43 March of the Exeggutor Squad_SPEAKER_04_001_processed.wav"
        # "Pokemon S01E43 March of the Exeggutor Squad_SPEAKER_04_001.wav"
        
        # Remove file extension first
        base_name = filename.rsplit('.', 1)[0]
        
        # Remove _processed suffix if present
        if base_name.endswith('_processed'):
            base_name = base_name[:-10]  # Remove "_processed"
        
        # Split and remove speaker and segment info
        parts = base_name.split('_')
        if len(parts) >= 3:
            # Remove last two parts (speaker ID and segment number)
            episode_part = '_'.join(parts[:-2])
            return episode_part
        return base_name
    
    def _extract_single_embedding(self, audio_file: Path, metadata: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract voice embedding from a single audio segment."""
        try:
            # Load audio
            audio, sr = librosa.load(str(audio_file), sr=16000)  # Standard sample rate for speaker models
            
            # Check duration constraints
            duration = len(audio) / sr
            min_dur = self.config['embedding']['min_audio_length']
            max_dur = self.config['embedding']['max_audio_length']
            
            if duration < min_dur:
                return None
            
            # Truncate if too long
            if duration > max_dur:
                max_samples = int(max_dur * sr)
                audio = audio[:max_samples]
            
            # Extract embedding
            if self.embedding_model is not None:
                # Use SpeechBrain model
                import torch
                audio_tensor = torch.tensor(audio).unsqueeze(0)
                embedding = self.embedding_model.encode_batch(audio_tensor)
                embedding = embedding.squeeze().cpu().numpy()
            else:
                # Fallback: use spectral features
                embedding = self._extract_spectral_features(audio, sr)
            
            # Normalize if requested
            if self.config['embedding']['normalize'] and embedding is not None:
                embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            logger.debug(f"Failed to extract embedding from {audio_file.name}: {e}")
            return None
    
    def _extract_spectral_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Fallback method: extract spectral features as voice embedding."""
        try:
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            
            # Extract spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
            
            # Compute statistics
            features = []
            for feature in [mfccs, spectral_centroids, spectral_rolloff, zero_crossing_rate]:
                features.extend([
                    np.mean(feature),
                    np.std(feature),
                    np.median(feature)
                ])
            
            return np.array(features)
            
        except Exception as e:
            logger.debug(f"Failed to extract spectral features: {e}")
            return None
    
    def _perform_clustering(self, embeddings: List[VoiceEmbedding]) -> ClusteringResult:
        """Perform clustering on voice embeddings."""
        # Prepare embedding matrix
        embedding_matrix = np.array([emb.embedding for emb in embeddings])
        
        # Standardize features if using spectral features
        if not SPEECHBRAIN_AVAILABLE:
            scaler = StandardScaler()
            embedding_matrix = scaler.fit_transform(embedding_matrix)
        
        # Perform clustering
        algorithm = self.config['clustering']['algorithm']
        
        if algorithm == 'dbscan':
            clustering = DBSCAN(
                eps=self.config['clustering']['eps'],
                min_samples=self.config['clustering']['min_samples'],
                metric=self.config['clustering']['distance_metric']
            )
        elif algorithm == 'hierarchical':
            n_clusters = self.config['clustering']['n_clusters'] or self._estimate_n_clusters(embeddings)
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=self.config['clustering']['linkage']
            )
        else:
            raise ValueError(f"Unsupported clustering algorithm: {algorithm}")
        
        # Fit clustering
        cluster_labels = clustering.fit_predict(embedding_matrix)
        
        # Create clusters
        clusters = self._create_clusters(embeddings, cluster_labels, embedding_matrix)
        
        # Generate speaker mapping
        speaker_mapping = self._generate_speaker_mapping(embeddings, cluster_labels)
        
        return ClusteringResult(
            clusters=clusters,
            speaker_mapping=speaker_mapping,
            cluster_assignments={},
            quality_metrics={},
            recommendations=[]
        )
    
    def _estimate_n_clusters(self, embeddings: List[VoiceEmbedding]) -> int:
        """Estimate optimal number of clusters based on expected characters."""
        # Count unique episodes and speakers
        episodes = set(emb.episode for emb in embeddings)
        speakers = set(emb.speaker_id for emb in embeddings)
        
        # For Pokemon, expect 6-8 main characters
        expected_chars = len(self.config['character_assignment']['pokemon_characters'])
        
        # Use heuristic: between expected characters and unique speakers/4
        min_clusters = max(4, expected_chars - 2)
        max_clusters = min(len(speakers) // 3, expected_chars + 3)
        
        return max(min_clusters, min(max_clusters, 12))  # Cap at 12 clusters
    
    def _create_clusters(self, embeddings: List[VoiceEmbedding], 
                        cluster_labels: np.ndarray, embedding_matrix: np.ndarray) -> List[VoiceCluster]:
        """Create cluster objects from clustering results."""
        clusters = []
        
        for cluster_id in set(cluster_labels):
            if cluster_id == -1:  # DBSCAN noise
                continue
            
            # Get embeddings for this cluster
            cluster_mask = cluster_labels == cluster_id
            cluster_embeddings = [emb for i, emb in enumerate(embeddings) if cluster_mask[i]]
            cluster_embedding_matrix = embedding_matrix[cluster_mask]
            
            # Calculate centroid
            centroid = np.mean(cluster_embedding_matrix, axis=0)
            
            # Calculate cluster statistics
            total_duration = sum(emb.duration for emb in cluster_embeddings)
            episodes = set(emb.episode for emb in cluster_embeddings)
            original_speakers = set(emb.speaker_id for emb in cluster_embeddings)
            quality_score = np.mean([emb.quality_score for emb in cluster_embeddings])
            
            # Select representative samples (highest quality, diverse episodes)
            representative_samples = self._select_representative_samples(cluster_embeddings)
            
            cluster = VoiceCluster(
                cluster_id=cluster_id,
                embeddings=cluster_embeddings,
                centroid=centroid,
                total_duration=total_duration,
                episodes=episodes,
                original_speakers=original_speakers,
                quality_score=quality_score,
                representative_samples=representative_samples
            )
            
            clusters.append(cluster)
        
        # Sort clusters by total duration (largest first)
        clusters.sort(key=lambda c: c.total_duration, reverse=True)
        
        return clusters
    
    def _select_representative_samples(self, embeddings: List[VoiceEmbedding], max_samples: int = 5) -> List[str]:
        """Select representative audio samples for a cluster."""
        # Sort by quality score
        sorted_embeddings = sorted(embeddings, key=lambda e: e.quality_score, reverse=True)
        
        # Select diverse samples from different episodes
        representatives = []
        seen_episodes = set()
        
        for emb in sorted_embeddings:
            if len(representatives) >= max_samples:
                break
            
            if emb.episode not in seen_episodes or len(representatives) < 2:
                representatives.append(emb.segment_file)
                seen_episodes.add(emb.episode)
        
        return representatives
    
    def _generate_speaker_mapping(self, embeddings: List[VoiceEmbedding], 
                                 cluster_labels: np.ndarray) -> Dict[str, Dict[str, int]]:
        """Generate mapping from episode/speaker to cluster ID."""
        mapping = defaultdict(dict)
        
        for emb, cluster_id in zip(embeddings, cluster_labels):
            if cluster_id != -1:  # Skip noise
                mapping[emb.episode][emb.speaker_id] = cluster_id
        
        return dict(mapping)
    
    def _analyze_cluster_quality(self, result: ClusteringResult):
        """Analyze and validate cluster quality."""
        quality_metrics = {}
        
        # Overall metrics
        total_clusters = len(result.clusters)
        total_duration = sum(c.total_duration for c in result.clusters)
        avg_cluster_size = np.mean([len(c.embeddings) for c in result.clusters]) if result.clusters else 0
        
        quality_metrics.update({
            'total_clusters': total_clusters,
            'total_duration_hours': total_duration / 3600,
            'avg_cluster_size': avg_cluster_size,
            'cluster_duration_distribution': [c.total_duration for c in result.clusters]
        })
        
        # Individual cluster quality
        quality_thresholds = self.config['quality']
        good_clusters = 0
        
        for cluster in result.clusters:
            is_good = (
                cluster.total_duration >= quality_thresholds['min_cluster_duration'] and
                len(cluster.episodes) >= quality_thresholds['min_episodes_per_cluster'] and
                len(cluster.embeddings) >= quality_thresholds['min_segments_per_cluster']
            )
            
            if is_good:
                good_clusters += 1
        
        quality_metrics['good_clusters'] = good_clusters
        quality_metrics['cluster_quality_rate'] = good_clusters / total_clusters if total_clusters > 0 else 0
        
        result.quality_metrics = quality_metrics
    
    def _assign_characters(self, result: ClusteringResult):
        """Assign character names to clusters based on dialogue content analysis."""
        assignments = {}
        recommendations = []
        
        # Character-specific dialogue patterns for Pokemon
        character_patterns = {
            'ash': [
                r'\bpikachu\b', r'\bgotta catch\b', r'\bi choose you\b', r'\bpok[eé]mon master\b',
                r'\bmy pokemon\b', r'\btrain\b', r'\bcatch them all\b', r'\bgo pokeball\b',
                # Additional Ash patterns based on actual dialogue
                r'\bworld.*best\b', r'\bgonna be\b', r'\bsomeday\b', r'\bpok[eé]ball\b',
                r'\bbattle\b', r'\bfight\b', r'\btrain.*pok[eé]mon\b', r'\bcome on\b',
                r'\byou can do it\b', r'\blet.*go\b', r'\bready\b', r'\bawesome\b'
            ],
            'misty': [
                r'\bwater pokemon\b', r'\bstarmie\b', r'\bpsyduck\b', r'\bgoldeen\b', 
                r'\bbike\b', r'\bmy bike\b', r'\bash ketchum\b', r'\byou better\b',
                # Additional Misty patterns
                r'\bwater\b', r'\bswim\b', r'\bpsyduck.*come back\b', r'\bstupid\b',
                r'\btomboy\b', r'\bfish\b', r'\bocean\b', r'\bpool\b', r'\bbeach\b'
            ],
            'brock': [
                r'\brock pokemon\b', r'\bonix\b', r'\bgeodude\b', r'\bbreeding\b',
                r'\bcook\b', r'\bfood\b', r'\btake care\b', r'\bzubat\b', r'\bbrock\b',
                # Additional Brock patterns
                r'\brock.*type\b', r'\bhard.*rock\b', r'\bstones?\b', r'\bmountain\b',
                r'\bearth\b', r'\bground\b', r'\bdigging\b', r'\brecipe\b'
            ],
            'jessie': [
                r'\bteam rocket\b', r'\bprepare for trouble\b', r'\barbok\b', 
                r'\bmake it double\b', r'\bjames\b', r'\bmeowth\b',
                # Additional Team Rocket patterns
                r'\btrouble\b', r'\bdouble\b', r'\bdevastating\b', r'\brunite\b',
                r'\bunite\b', r'\bnation\b', r'\bto protect\b', r'\bto unite\b'
            ],
            'james': [
                r'\bteam rocket\b', r'\bmake it double\b', r'\bekans\b', r'\bweezing\b',
                r'\bjessie\b', r'\bmeowth\b', r'\bto protect the world\b',
                # Additional James patterns
                r'\bdouble\b', r'\bdevastation\b', r'\bunite\b', r'\bnation\b',
                r'\bto protect\b', r'\bto unite\b', r'\bwhite\b', r'\bjames\b'
            ],
            'meowth': [
                r'\bmeowth\b', r'\bthat.*right\b', r'\bteam rocket\b', r'\bboss\b',
                r'\bscratch\b', r'\bpay day\b', r'\bfury swipes\b',
                # Additional Meowth patterns
                r'\bda boss\b', r'\bdat.*right\b', r'\bmeowth.*name\b', r'\btalk\b',
                r'\bcat\b', r'\bcoin\b', r'\bmoney\b', r'\brich\b'
            ],
            'narrator': [
                r'\bour hero\b', r'\bmeanwhile\b', r'\bto be continued\b', r'\bpokemon trainer\b',
                r'\bash and\b', r'\bjourney\b', r'\badventure\b', r'\bquest\b',
                # Additional narrator patterns
                r'\bwill.*continue\b', r'\bnext time\b', r'\bstory\b', r'\btale\b',
                r'\bworld.*pok[eé]mon\b', r'\btrainer.*named\b', r'\byoung.*trainer\b'
            ],
            'pokédex': [
                r'\bpok[eé]dex\b', r'\binformation\b', r'\bdata\b', r'\banalysis\b',
                r'\bspecie.*pok[eé]mon\b', r'\bheight\b', r'\bweight\b', r'\bevolves?\b',
                # Additional Pokédex patterns
                r'\btype.*pok[eé]mon\b', r'\bmeasuring\b', r'\bweighing\b', r'\bentry\b',
                r'\bregistered\b', r'\bclassified\b', r'\bcategory\b'
            ]
        }
        
        # Exclude theme song and generic patterns
        theme_song_patterns = [
            r'\bgotta catch.*all\b', r'\bpokemon.*theme\b', r'\bi want to be\b',
            r'\bvery best\b', r'\breal test\b', r'\bmy cause\b', r'\bpower.*inside\b'
        ]
        
        # Score each cluster for each character
        cluster_scores = {}
        
        for cluster in result.clusters:
            # Extract all text from cluster
            all_text = []
            dialogue_segments = 0
            theme_segments = 0
            
            for embedding in cluster.embeddings:
                text = embedding.text.lower()
                all_text.append(text)
                
                # Check if this is theme song vs dialogue
                is_theme = any(re.search(pattern, text, re.IGNORECASE) for pattern in theme_song_patterns)
                if is_theme:
                    theme_segments += 1
                else:
                    dialogue_segments += 1
            
            combined_text = ' '.join(all_text)
            
            # Calculate theme song ratio
            total_segments = len(cluster.embeddings)
            theme_ratio = theme_segments / total_segments if total_segments > 0 else 0
            
            # Score against each character
            char_scores = {}
            for character, patterns in character_patterns.items():
                score = 0
                for pattern in patterns:
                    matches = len(re.findall(pattern, combined_text, re.IGNORECASE))
                    score += matches
                
                # Normalize by cluster size and penalize theme song heavy clusters
                normalized_score = score / total_segments if total_segments > 0 else 0
                theme_penalty = max(0, 1 - (theme_ratio * 2))  # Heavy penalty for theme-heavy clusters
                char_scores[character] = normalized_score * theme_penalty
            
            # Detect mixed clusters - multiple characters with significant scores
            significant_chars = [char for char, score in char_scores.items() if score > 0.03]
            is_mixed_cluster = len(significant_chars) > 1
            
            # Also consider duration and episode spread for main characters
            duration_factor = min(1.0, cluster.total_duration / 120.0)  # Cap at 2 minutes
            episode_factor = min(1.0, len(cluster.episodes) / 4.0)  # Want presence across episodes
            
            # Apply bonuses for main character indicators
            for character in ['ash', 'misty', 'brock']:
                char_scores[character] *= (0.5 + 0.5 * duration_factor * episode_factor)
            
            cluster_scores[cluster.cluster_id] = {
                'char_scores': char_scores,
                'total_duration': cluster.total_duration,
                'episodes': len(cluster.episodes),
                'segments': total_segments,
                'theme_ratio': theme_ratio,
                'dialogue_segments': dialogue_segments,
                'is_mixed': is_mixed_cluster,
                'significant_chars': significant_chars
            }
        
        # Sort clusters by total dialogue content (not theme songs)
        dialogue_clusters = [
            (cluster_id, data) for cluster_id, data in cluster_scores.items()
            if data['dialogue_segments'] > data['segments'] * 0.5  # At least 50% dialogue
        ]
        dialogue_clusters.sort(key=lambda x: x[1]['total_duration'], reverse=True)
        
        # Assign characters using content analysis
        pokemon_chars = self.config['character_assignment']['pokemon_characters']
        used_characters = set()
        
        # First pass: assign high-confidence single-character matches
        for cluster_id, data in dialogue_clusters:
            if not data['is_mixed']:  # Only assign single-character clusters first
                best_char = max(data['char_scores'].items(), key=lambda x: x[1])
                char_name, score = best_char
                
                if score > 0.1 and char_name not in used_characters and char_name in pokemon_chars:
                    assignments[cluster_id] = char_name
                    used_characters.add(char_name)
                    
                    recommendations.append(
                        f"Cluster {cluster_id} → {char_name}: "
                        f"content score {score:.2f}, {data['dialogue_segments']} dialogue segments, "
                        f"{data['total_duration']/60:.1f}min across {data['episodes']} episodes"
                    )
        
        # Second pass: handle mixed clusters - assign to highest scoring unused character
        mixed_clusters = [(cluster_id, data) for cluster_id, data in dialogue_clusters if data['is_mixed']]
        for cluster_id, data in mixed_clusters:
            # Find best unused character
            available_scores = {char: score for char, score in data['char_scores'].items() 
                             if char not in used_characters and char in pokemon_chars}
            
            if available_scores:
                best_char = max(available_scores.items(), key=lambda x: x[1])
                char_name, score = best_char
                
                if score > 0.05:  # Lower threshold for mixed clusters
                    assignments[cluster_id] = char_name
                    used_characters.add(char_name)
                    
                    recommendations.append(
                        f"Cluster {cluster_id} → {char_name}: "
                        f"MIXED CLUSTER (also has {', '.join(data['significant_chars'])}), "
                        f"content score {score:.2f}, {data['dialogue_segments']} dialogue segments, "
                        f"{data['total_duration']/60:.1f}min across {data['episodes']} episodes"
                    )
        
        # Third pass: assign remaining main characters by duration
        remaining_main_chars = [c for c in pokemon_chars if c not in used_characters]
        unassigned_clusters = [
            (cluster_id, data) for cluster_id, data in dialogue_clusters
            if cluster_id not in assignments
        ]
        
        for i, (cluster_id, data) in enumerate(unassigned_clusters):
            if i < len(remaining_main_chars):
                char_name = remaining_main_chars[i]
                assignments[cluster_id] = char_name
                used_characters.add(char_name)
                
                recommendations.append(
                    f"Cluster {cluster_id} → {char_name}: "
                    f"fallback assignment, {data['dialogue_segments']} dialogue segments, "
                    f"{data['total_duration']/60:.1f}min across {data['episodes']} episodes"
                )
        
        # Fourth pass: assign theme song and minor clusters
        all_clusters = sorted(result.clusters, key=lambda c: c.total_duration, reverse=True)
        narrator_assigned = False
        
        for cluster in all_clusters:
            if cluster.cluster_id not in assignments:
                data = cluster_scores[cluster.cluster_id]
                
                if data['theme_ratio'] > 0.7 and not narrator_assigned:
                    # Likely theme song cluster - only assign first one as narrator
                    assignments[cluster.cluster_id] = "narrator"
                    narrator_assigned = True
                    recommendations.append(
                        f"Cluster {cluster.cluster_id} → narrator: "
                        f"theme song cluster ({data['theme_ratio']*100:.0f}% theme), "
                        f"{data['total_duration']/60:.1f}min"
                    )
                else:
                    # Minor character or additional theme clusters
                    if data['theme_ratio'] > 0.7:
                        assignments[cluster.cluster_id] = f"theme_song_{cluster.cluster_id}"
                        recommendations.append(
                            f"Cluster {cluster.cluster_id} → theme song: "
                            f"additional theme cluster ({data['theme_ratio']*100:.0f}% theme), "
                            f"{data['total_duration']/60:.1f}min"
                        )
                    else:
                        assignments[cluster.cluster_id] = f"minor_character_{cluster.cluster_id}"
                        recommendations.append(
                            f"Cluster {cluster.cluster_id} → minor character: "
                            f"{data['dialogue_segments']} dialogue segments, manual review needed"
                        )
        
        result.cluster_assignments = assignments
        result.recommendations = recommendations
    
    async def _save_results(self, result: ClusteringResult, output_path: Path):
        """Save clustering results to files."""
        # Save main results
        results_data = {
            'clusters': [
                {
                    'cluster_id': c.cluster_id,
                    'total_duration': c.total_duration,
                    'segment_count': len(c.embeddings),
                    'episodes': list(c.episodes),
                    'original_speakers': list(c.original_speakers),
                    'quality_score': c.quality_score,
                    'representative_samples': c.representative_samples,
                    'assigned_character': result.cluster_assignments.get(c.cluster_id, 'unassigned')
                }
                for c in result.clusters
            ],
            'speaker_mapping': result.speaker_mapping,
            'cluster_assignments': result.cluster_assignments,
            'quality_metrics': result.quality_metrics,
            'recommendations': result.recommendations
        }
        
        results_file = output_path / 'voice_clustering_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(results_data), f, indent=2, ensure_ascii=False)
        
        # Save speaker mapping for easy lookup
        mapping_file = output_path / 'speaker_mapping.json'
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(result.speaker_mapping), f, indent=2, ensure_ascii=False)
        
        # Save embeddings for future analysis
        embeddings_file = output_path / 'voice_embeddings.pkl'
        with open(embeddings_file, 'wb') as f:
            pickle.dump(result.clusters, f)
        
        logger.info(f"Clustering results saved to {output_path}")
    
    async def apply_clustering_to_segments(self, clustering_results_path: str, 
                                         segments_dir: str, output_dir: str) -> Dict[str, Any]:
        """Apply clustering results to reorganize segments by character."""
        logger.info(f"Applying clustering to reorganize segments: {segments_dir} → {output_dir}")
        
        # Load clustering results
        with open(clustering_results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        speaker_mapping = results['speaker_mapping']
        cluster_assignments = results['cluster_assignments']
        
        segments_path = Path(segments_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create character directories
        for cluster_id, character in cluster_assignments.items():
            char_dir = output_path / character
            char_dir.mkdir(exist_ok=True)
        
        # Reorganize segments
        reorganized_count = 0
        failed_count = 0
        
        for episode, episode_mapping in speaker_mapping.items():
            for original_speaker, cluster_id in episode_mapping.items():
                character = cluster_assignments.get(str(cluster_id), f'cluster_{cluster_id}')
                
                # Find original speaker directory
                original_speaker_dir = segments_path / original_speaker
                if not original_speaker_dir.exists():
                    continue
                
                # Move/copy segments to character directory
                char_dir = output_path / character
                
                for audio_file in original_speaker_dir.glob(f"{episode}*"):
                    try:
                        # Copy audio and metadata
                        target_file = char_dir / audio_file.name
                        if audio_file.suffix == '.wav':
                            # Copy audio file
                            import shutil
                            shutil.copy2(audio_file, target_file)
                            
                            # Copy metadata if exists  
                            metadata_file = audio_file.with_suffix('.json')
                            if metadata_file.exists():
                                target_metadata = target_file.with_suffix('.json')
                                shutil.copy2(metadata_file, target_metadata)
                            
                            reorganized_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to copy {audio_file}: {e}")
                        failed_count += 1
        
        return {
            'success': True,
            'segments_reorganized': reorganized_count,
            'segments_failed': failed_count,
            'characters_created': len(cluster_assignments),
            'output_path': str(output_path)
        } 