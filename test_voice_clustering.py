#!/usr/bin/env python3
"""
Quick test script for voice clustering system
Tests the clustering on a small subset of segments
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pipeline.stages.voice_clustering import VoiceClusteringSystem


async def test_clustering():
    """Test voice clustering on a small subset of data."""
    
    # Test with processed segments (assuming they exist)
    segments_dir = "resources/segments_processed"
    output_dir = "resources/voice_clusters_test"
    
    if not Path(segments_dir).exists():
        print(f"❌ Segments directory not found: {segments_dir}")
        print("   Run segment processing first with:")
        print("   python main.py process-segments --input resources/transcripts/segments --output resources/segments_processed")
        return 1
    
    print(f"🧪 Testing voice clustering system...")
    print(f"📁 Input: {segments_dir}")
    print(f"📁 Output: {output_dir}")
    
    # Create clustering system with test config
    config = {
        'embedding': {
            'normalize': True,
            'max_audio_length': 8.0,
            'min_audio_length': 1.5,
        },
        'clustering': {
            'algorithm': 'dbscan',
            'eps': 0.4,
            'min_samples': 3,
            'distance_metric': 'cosine',
        },
        'quality': {
            'min_cluster_duration': 20.0,
            'min_episodes_per_cluster': 1,
            'min_segments_per_cluster': 5,
        }
    }
    
    clustering_system = VoiceClusteringSystem(config)
    
    try:
        result = await clustering_system.cluster_voices(segments_dir, output_dir)
        
        if result['success']:
            print(f"✅ Test successful!")
            print(f"📊 Results:")
            print(f"   Voice embeddings: {result['total_embeddings']}")
            print(f"   Clusters found: {result['clusters_found']}")
            print(f"   Output saved to: {result['output_path']}")
            
            if result.get('cluster_assignments'):
                print(f"🎭 Character assignments:")
                for cluster_id, character in result['cluster_assignments'].items():
                    print(f"   Cluster {cluster_id} → {character}")
            
            return 0
        else:
            print(f"❌ Test failed: {result.get('error', 'Unknown error')}")
            return 1
            
    except Exception as e:
        print(f"❌ Test exception: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(test_clustering())
    sys.exit(exit_code) 