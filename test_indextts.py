#!/usr/bin/env python3
"""
Test script for IndexTTS integration
Tests voice cloning with the meowth reference audio
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.models.indextts.indextts_trainer import IndexTTSTrainer
from src.utils.logging_utils import setup_logging

def main():
    """Test IndexTTS voice cloning with meowth reference."""
    
    # Setup logging
    setup_logging(log_level="INFO")
    
    print("🧪 Testing IndexTTS Integration")
    print("=" * 50)
    
    # Check if meowth reference file exists
    reference_file = Path("temp/concatenated_references/meowth_concatenated_reference.wav")
    if not reference_file.exists():
        print(f"❌ Reference file not found: {reference_file}")
        print("   Please ensure the meowth reference file exists.")
        return 1
    
    print(f"✅ Found reference audio: {reference_file}")
    
    # Test texts for Meowth
    test_texts = [
        "That's right! Team Rocket's here to steal your Pokemon!",
        "Meowth, that's right! We're gonna catch Pikachu!",
        "Boss, we've got a great plan this time!",
        "These twerps don't know what's coming!"
    ]
    
    try:
        # Initialize IndexTTS trainer
        print("\n🔧 Initializing IndexTTS trainer...")
        trainer = IndexTTSTrainer()
        
        # Test synthesis for each text
        for i, text in enumerate(test_texts, 1):
            print(f"\n🎙️ Test {i}/4: Synthesizing text...")
            print(f"   Text: '{text}'")
            
            output_file = f"test_meowth_output_{i}.wav"
            
            # Run synthesis
            result = asyncio.run(trainer.synthesize(
                text=text,
                reference_audio=str(reference_file),
                character="meowth",
                output_path=output_file
            ))
            
            if result.success:
                print(f"   ✅ Success! Generated: {result.audio_path}")
                print(f"   ⏱️ Generation time: {result.generation_time:.2f}s")
            else:
                print(f"   ❌ Failed: {result.error}")
                return 1
        
        print("\n🎉 All tests passed! IndexTTS is working correctly.")
        print("\n📁 Output files generated:")
        for i in range(1, len(test_texts) + 1):
            output_file = f"test_meowth_output_{i}.wav"
            if Path(output_file).exists():
                print(f"   - {output_file}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 