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
from pathlib import Path
from datasets import Dataset
from src.preprocessing.data_loader import DataLoader
from src.preprocessing.preprocess import normalize_text, preprocess_transcript

def load_metadata(data_dir="./data", metadata_file="metadata.csv"):
    """
    Load metadata from CSV using DataLoader.
    """
    loader = DataLoader()
    
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return None
    
    metadata_path = os.path.join(data_dir, metadata_file)
    if os.path.exists(metadata_path):
        metadata = loader._load_metadata(data_dir, metadata_file)
        return metadata
    
    print(f"Metadata file not found: {metadata_path}")
    return None

def apply_datas(data_path=None, metadata=None):
    """
    Convert audio + lyrics + prompts into a list of examples.
    """
    all_examples = []

    if metadata is not None:
        # Metadata-driven processing
        for _, row in metadata.iterrows():
            song_path = Path(row["audio_path"])
            prompt_path = row["prompt_path"]
            lyric_path = row["lyrics_path"]

            try:
                assert song_path.exists(), f"Audio file {song_path} missing."
                assert os.path.exists(prompt_path), f"Prompt file {prompt_path} missing."
                assert os.path.exists(lyric_path), f"Lyrics file {lyric_path} missing."

                with open(lyric_path, "r", encoding="utf8") as f:
                    lyrics = f.read().strip()
                with open(prompt_path, "r", encoding="utf8") as f:
                    prompt = f.read().strip()

                # --- Step 1: normalize ---
                normalized = normalize_text(lyrics)

                # --- Step 2: convert to phonemes ---
                phonemized = preprocess_transcript(normalized)

                example = {
                    "keys": song_path.stem,
                    "filename": str(song_path),
                    "tags": prompt.split(", "),
                    "speaker_emb_path": "",
                    "norm_lyrics": phonemized or normalized,
                    "recaption": {}
                }
                all_examples.append(example)

            except AssertionError as e:
                print(f"Skipping file: {e}")
                continue

    elif data_path is not None:
        data_path = Path(data_path)
        for song_path in data_path.glob("*.wav"):
            prompt_path = str(song_path).replace(".wav", "_prompt.txt")
            lyric_path = str(song_path).replace(".wav", "_lyric.txt")

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
                    "norm_lyrics": phonemized or normalized,
                    "recaption": {}
                }
                all_examples.append(example)

            except AssertionError as e:
                print(f"Skipping file: {e}")
                continue

    return all_examples

def create_dataset(data_dir="./data", repeat_count=2000, output_name="ta_hf_dataset", metadata_load=False):
    """
    Build HuggingFace Dataset and save to disk.
    """
    if metadata_load:
        metadata = load_metadata(data_dir)
        if metadata is None:
            return
        all_examples = apply_datas(metadata=metadata)
    else:
        all_examples = apply_datas(data_path=data_dir)

    if not all_examples:
        print("No valid examples found. Dataset not created.")
        return

    ds = Dataset.from_list(all_examples * repeat_count)
    ds.save_to_disk(output_name)
    print(f"✅ Tamil HF dataset saved → {output_name} | Total samples: {len(all_examples) * repeat_count}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert dataset to HuggingFace format for ACE-STEP fine-tuning")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--repeat_count", type=int, default=100)
    parser.add_argument("--output_name", type=str, default="ta_hf_dataset")
    parser.add_argument("--metadata_load", type=bool , default =False ,help="Load metadata from CSV instead of scanning folder")

    args = parser.parse_args()
    create_dataset(args.data_dir, args.repeat_count, args.output_name, metadata_load=args.metadata_load)
