#!/usr/bin/env python3
"""
Script to create Tamil dataset for ACE-Step training
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from convert2hf_dataset import create_dataset

def main():
    """Create Tamil dataset"""
    data_dir = "./data/tamil_dataset"
    output_name = "ta_hf_dataset"
    repeat_count = 1  # Start with 1 to test, increase for actual training
    
    print(f"Creating Tamil dataset from: {data_dir}")
    print(f"Output dataset name: {output_name}")
    print(f"Repeat count: {repeat_count}")
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} does not exist!")
        return
    
    # Create the dataset
    try:
        create_dataset(
            data_dir=data_dir,
            repeat_count=repeat_count,
            output_name=output_name
        )
        print(f"Successfully created dataset: {output_name}")
    except Exception as e:
        print(f"Error creating dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()