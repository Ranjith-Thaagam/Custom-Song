# Tamil LoRA Training Guide

This guide explains how to train a Tamil language LoRA adapter for the ACE-Step music generation model.

## 🚀 Quick Start

1. **Run the setup script:**
   ```bash
   python setup_tamil_training.py
   ```

2. **Start training:**
   ```bash
   python train_tamil.py
   ```

## 📋 Prerequisites

- Python 3.8+
- PyTorch with CUDA support (recommended)
- At least 8GB GPU memory
- Tamil dataset in `data/tamil_dataset/` directory

## 📁 Dataset Structure

Your Tamil dataset should be organized as follows:
```
data/tamil_dataset/
├── song1.wav (or .mp3)
├── song1_prompt.txt
├── song1_lyric.txt
├── song2.wav
├── song2_prompt.txt
├── song2_lyric.txt
└── ...
```

### File Formats:
- **Audio files**: `.wav` or `.mp3` format
- **Prompt files**: Text descriptions of the music style/genre
- **Lyric files**: Tamil lyrics in UTF-8 encoding

## 🔧 Configuration

### LoRA Configuration (`config/ta_lora_config.json`):
```json
{
    "r": 8,
    "lora_alpha": 16,
    "target_modules": [
        "linear_q", "linear_k", "linear_v",
        "to_q", "to_k", "to_v", "to_out.0"
    ]
}
```

### Training Parameters:
- **Learning Rate**: 1e-4
- **Max Steps**: 10,000 (adjust based on dataset size)
- **Batch Size**: Effective batch size of 2 (with gradient accumulation)
- **Precision**: 32-bit (change to 16-bit for memory savings)

## 🎯 Training Process

### Step 1: Dataset Creation
```bash
python create_tamil_dataset.py
```
This creates a HuggingFace dataset from your Tamil audio files.

### Step 2: Dataset Validation
```bash
python validate_dataset.py
```
Validates audio files and dataset structure.

### Step 3: Training
```bash
python train_tamil.py
```
Or with custom parameters:
```bash
python trainer.py \
    --dataset_path ./ta_hf_dataset \
    --exp_name tamil_lora \
    --lora_config_path config/ta_lora_config.json \
    --max_steps 5000 \
    --learning_rate 1e-4
```

## 📊 Monitoring Training

### TensorBoard Logs:
```bash
tensorboard --logdir exps/logs/
```

### Key Metrics to Watch:
- **Denoising Loss**: Should decrease over time
- **Learning Rate**: Should follow the schedule
- **GPU Memory**: Monitor for OOM errors

## 🔍 Troubleshooting

### Common Issues:

1. **Out of Memory (OOM)**:
   - Reduce batch size: `--accumulate_grad_batches 1`
   - Use mixed precision: `--precision 16`
   - Enable CPU offloading in pipeline

2. **Dataset Loading Errors**:
   - Check file paths in dataset
   - Ensure audio files are valid
   - Verify text encoding (UTF-8)

3. **Language Tokenization Issues**:
   - Tamil language support is added to `SUPPORT_LANGUAGES`
   - Check lyric tokenization in logs

4. **Slow Training**:
   - Use GPU if available
   - Increase `num_workers` for data loading
   - Enable `torch_compile` for faster inference

### Memory Optimization:
```bash
# For low memory systems
python trainer.py \
    --dataset_path ./ta_hf_dataset \
    --exp_name tamil_lora_lowmem \
    --precision 16 \
    --accumulate_grad_batches 4 \
    --num_workers 2
```

## 📈 Results and Evaluation

### Checkpoints:
- Saved every 1000 steps in `exps/logs/[timestamp]tamil_lora/checkpoints/`
- Each checkpoint contains the LoRA adapter weights

### Testing the Model:
```bash
# Use the trained LoRA with inference
python infer.py --lora_path path/to/checkpoint --prompt "Tamil folk song" --lyrics "தமிழ் பாடல்"
```

## 🎵 Sample Data Format

### Prompt Example (`song_prompt.txt`):
```
"Energetic Tamil folk song with traditional instruments, male vocals, and festive rhythm"
```

### Lyric Example (`song_lyric.txt`):
```
[Verse]
வானம் பார்த்து நிற்கும் மரம்
காற்றில் ஆடும் இலைகள்

[Chorus]
பாடல் பாடி வா
நம்முடன் சேர்ந்து வா
```

## 🔧 Advanced Configuration

### Custom LoRA Settings:
- Increase `r` (rank) for more parameters: 16, 32, 64
- Adjust `lora_alpha` for scaling: 16, 32, 64
- Add more target modules for broader adaptation

### Training Optimization:
- Use gradient checkpointing for memory efficiency
- Implement learning rate scheduling
- Add validation dataset for monitoring

## 📚 Additional Resources

- [ACE-Step Paper](https://arxiv.org/abs/2506.00045)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Tamil Language Processing](https://github.com/indicnlp/indicnlp_catalog)

## 🤝 Contributing

To improve Tamil language support:
1. Add more diverse Tamil music samples
2. Improve lyric tokenization for Tamil script
3. Optimize model parameters for Tamil phonetics
4. Create evaluation metrics for Tamil music generation

## 📄 License

This project follows the Apache 2.0 License as specified in the main ACE-Step repository.