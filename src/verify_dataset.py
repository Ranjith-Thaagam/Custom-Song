from datasets import load_from_disk

ds = load_from_disk('./ta_hf_dataset')
print(ds[0])