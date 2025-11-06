"""
convert2hf_dataset.py
----------------------
Custom dataset builder for ACE-STEP fine-tuning (Tamil / Indic Languages)

Uses:
  - DataLoader (data_loader.py)
  - preprocess (Indic normalization + phoneme conversion)
  - HuggingFace Dataset for saving to disk


"""

import os
import argparse
from datasets import Dataset
from src.preprocessing.data_loader import DataLoader
from src.preprocessing.preprocess import normalize_text, preprocess_transcript, detect_language
from pathlib import Path

def create_dataset(data_dir="./data", repeat_count=2000, output_name="ta_hf_dataset"):
    data_path = Path(data_dir)
    all_examples = []

    for song_path in data_path.glob("*.mp3"):
        prompt_path = str(song_path).replace(".mp3", "_prompt.txt")
        lyric_path = str(song_path).replace(".mp3", "_lyric.txt")

        try:
            assert os.path.exists(prompt_path), f"Prompt file {prompt_path} missing."
            assert os.path.exists(lyric_path), f"Lyrics file {lyric_path} missing."

            with open(prompt_path, "r", encoding="utf8") as f:
                prompt = f.read().strip()

            with open(lyric_path, "r", encoding="utf8") as f:
                lyrics = f.read().strip()

            # --- Step 1: normalize ---

            normalized = normalize_text(lyrics)

            # --- Step 2: convert to phonemes ---
            phonemized = preprocess_transcript(normalized)

            example = {
                "keys": song_path.stem,
                "filename": str(song_path),
                "tags": prompt.split(", "),
                "speaker_emb_path": "",
                "norm_lyrics": phonemized or normalized,  # fallback
                "recaption": {}
            }
            all_examples.append(example)

        except AssertionError as e:
            print(f"Skipping file: {e}")
            continue

    ds = Dataset.from_list(all_examples * repeat_count)
    ds.save_to_disk(output_name)
    print(f"✅ Tamil HF dataset saved → {output_name}  | Total samples: {len(all_examples)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--repeat_count", type=int, default=100)
    parser.add_argument("--output_name", type=str, default="ta_hf_dataset")
    args = parser.parse_args()
    create_dataset(args.data_dir, args.repeat_count, args.output_name)
