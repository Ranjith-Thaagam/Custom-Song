#!/usr/bin/env python3
"""
Training script for Tamil LoRA fine-tuning
"""

import os
import sys
import argparse
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from trainer import main as trainer_main

def create_tamil_training_args():
    """Create training arguments for Tamil dataset"""
    
    # Create argument parser
    parser = argparse.ArgumentParser(description="Train Tamil LoRA model")
    
    # Add all the arguments with Tamil-specific defaults
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)  # Reduced for smaller dataset
    parser.add_argument("--epochs", type=int, default=-1)
    parser.add_argument("--max_steps", type=int, default=10000)  # Reduced for Tamil dataset
    parser.add_argument("--every_n_train_steps", type=int, default=1000)  # More frequent checkpoints
    parser.add_argument("--dataset_path", type=str, default="./ta_hf_dataset")
    parser.add_argument("--exp_name", type=str, default="tamil_lora")
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--accumulate_grad_batches", type=int, default=2)  # Increased for smaller batch size
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--logger_dir", type=str, default="./exps/logs/")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--gradient_clip_val", type=float, default=0.5)
    parser.add_argument("--gradient_clip_algorithm", type=str, default="norm")
    parser.add_argument("--reload_dataloaders_every_n_epochs", type=int, default=1)
    parser.add_argument("--every_plot_step", type=int, default=1000)  # More frequent evaluation
    parser.add_argument("--val_check_interval", type=int, default=None)
    parser.add_argument("--lora_config_path", type=str, default="config/ta_lora_config.json")
    
    return parser

def main():
    """Main training function"""
    
    # Check if Tamil dataset exists
    dataset_path = "./ta_hf_dataset"
    if not os.path.exists(dataset_path):
        print(f"Error: Tamil dataset not found at {dataset_path}")
        print("Please run 'python create_tamil_dataset.py' first to create the dataset.")
        return
    
    # Check if LoRA config exists
    lora_config_path = "config/ta_lora_config.json"
    if not os.path.exists(lora_config_path):
        print(f"Error: LoRA config not found at {lora_config_path}")
        return
    
    print("Starting Tamil LoRA training...")
    print(f"Dataset: {dataset_path}")
    print(f"LoRA config: {lora_config_path}")
    
    # Create parser and parse arguments
    parser = create_tamil_training_args()
    args = parser.parse_args()
    
    # Start training
    try:
        trainer_main(args)
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()