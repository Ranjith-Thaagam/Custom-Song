#!/usr/bin/env python3
"""
Complete setup script for Tamil LoRA training
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Error: {e.stderr}")
        return False

def check_file_exists(file_path, description):
    """Check if a file exists"""
    if os.path.exists(file_path):
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description} missing: {file_path}")
        return False

def main():
    """Main setup function"""
    
    print("🚀 Setting up Tamil LoRA Training Environment")
    print("=" * 50)
    
    # Step 1: Check prerequisites
    print("\n📋 Checking Prerequisites...")
    
    prerequisites_ok = True
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major >= 3 and python_version.minor >= 8:
        print(f"✅ Python version: {python_version.major}.{python_version.minor}")
    else:
        print(f"❌ Python version too old: {python_version.major}.{python_version.minor}")
        prerequisites_ok = False
    
    # Check required files
    required_files = [
        ("config/ta_lora_config.json", "Tamil LoRA config"),
        ("data/tamil_dataset", "Tamil dataset directory"),
        ("acestep/pipeline_ace_step.py", "ACE-Step pipeline"),
        ("trainer.py", "Training script"),
    ]
    
    for file_path, description in required_files:
        if not check_file_exists(file_path, description):
            prerequisites_ok = False
    
    if not prerequisites_ok:
        print("\n❌ Prerequisites check failed. Please fix the issues above.")
        return False
    
    # Step 2: Create Tamil dataset
    print("\n📊 Creating Tamil Dataset...")
    if not run_command("python create_tamil_dataset.py", "Tamil dataset creation"):
        return False
    
    # Step 3: Validate dataset
    print("\n🔍 Validating Dataset...")
    if not run_command("python validate_dataset.py", "Dataset validation"):
        print("⚠️  Dataset validation failed, but continuing...")
    
    # Step 4: Check GPU availability
    print("\n🖥️  Checking GPU Availability...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU available: {gpu_name} (Count: {gpu_count})")
        else:
            print("⚠️  No GPU available, training will be slow on CPU")
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    
    # Step 5: Create output directories
    print("\n📁 Creating Output Directories...")
    output_dirs = [
        "exps/logs",
        "exps/checkpoints",
        "outputs"
    ]
    
    for dir_path in output_dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✅ Created directory: {dir_path}")
    
    # Step 6: Display training command
    print("\n🎯 Setup Complete!")
    print("=" * 50)
    print("\nTo start training, run:")
    print("python train_tamil.py")
    print("\nOr with custom parameters:")
    print("python trainer.py --dataset_path ./ta_hf_dataset --exp_name tamil_lora --lora_config_path config/ta_lora_config.json --max_steps 5000")
    
    print("\n📝 Training Tips:")
    print("- Start with fewer steps (5000-10000) for initial testing")
    print("- Monitor GPU memory usage during training")
    print("- Check logs in exps/logs/ directory")
    print("- Checkpoints will be saved in exps/logs/[timestamp]tamil_lora/checkpoints/")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Setup completed successfully!")
    else:
        print("\n💥 Setup failed. Please check the errors above.")
        sys.exit(1)