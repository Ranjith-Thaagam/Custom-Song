from src.preprocessing.preprocess import normalize_text , preprocess_transcript , detect_language

from datasets import load_from_disk

datas = load_from_disk('./ta_hf_dataset')
print(f"Total samples: {len(datas)}")
print(f"Keys : {datas[0].keys()}")

print(f"Lyrics : {datas[0]['norm_lyrics']}")

if __name__ == "__main__":
    print(f"Total samples: {len(datas)}")
    print(f"Keys : {datas[0].keys()}")
    print(f"Lyrics : {datas[0]['norm_lyrics']}")
    