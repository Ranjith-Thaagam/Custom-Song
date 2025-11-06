#!/usr/bin/env python3
"""
Script to validate the Tamil dataset before training
"""

import os
import sys
from pathlib import Path
import torchaudio
from datasets import load_from_disk

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def validate_audio_file(audio_path):
    """Validate an audio file"""
    try:
        audio, sr = torchaudio.load(audio_path)
        duration = audio.shape[1] / sr
        return True, duration, sr, audio.shape
    except Exception as e:
        return False, 0, 0, None

def validate_dataset(dataset_path):
    """Validate the dataset"""
    print(f"Validating dataset: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path {dataset_path} does not exist!")
        return False
    
    try:
        # Load the dataset
        dataset = load_from_disk(dataset_path)
        print(f"Dataset loaded successfully. Total samples: {len(dataset)}")
        
        # Check first few samples
        valid_samples = 0
        invalid_samples = 0
        total_duration = 0
        
        print("\nValidating samples...")
        for i in range(min(10, len(dataset))):  # Check first 10 samples
            sample = dataset[i]
            
            print(f"\nSample {i+1}:")
            print(f"  Key: {sample['keys']}")
            print(f"  Audio file: {sample['filename']}")
            print(f"  Tags: {sample['tags']}")
            print(f"  Lyrics length: {len(sample['norm_lyrics'])} characters")
            
            # Validate audio file
            is_valid, duration, sr, shape = validate_audio_file(sample['filename'])
            if is_valid:
                print(f"  Audio: Valid ({duration:.2f}s, {sr}Hz, {shape})")
                valid_samples += 1
                total_duration += duration
            else:
                print(f"  Audio: Invalid or missing")
                invalid_samples += 1
        
        print(f"\nValidation Summary:")
        print(f"  Valid samples: {valid_samples}")
        print(f"  Invalid samples: {invalid_samples}")
        if valid_samples > 0:
            print(f"  Average duration: {total_duration/valid_samples:.2f}s")
        
        return valid_samples > 0
        
    except Exception as e:
        print(f"Error validating dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main validation function"""
    
    # Validate Tamil dataset
    tamil_dataset_path = "./ta_hf_dataset"
    
    print("=== Tamil Dataset Validation ===")
    is_valid = validate_dataset(tamil_dataset_path)
    
    if is_valid:
        print("\n✅ Dataset validation passed!")
    else:
        print("\n❌ Dataset validation failed!")
        print("Please check the dataset creation process.")

if __name__ == "__main__":
    main()