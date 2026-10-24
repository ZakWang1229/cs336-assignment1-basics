import json
import pickle
import numpy as np
from pathlib import Path
from cs336_basics._rust import Tokenlizer

script_dir = Path(__file__).resolve().parent
train_dir = script_dir.parent / "train"
data_dir = script_dir.parents[2] / "data"

with open(train_dir / "vocab.pkl", "rb") as f:
    vocab = pickle.load(f)
with open(train_dir / "merges.pkl", "rb") as f:
    merges = pickle.load(f)
with open(train_dir / "summary.json", "r", encoding="utf-8") as f:
    special_tokens = json.load(f)["special_tokens"]

tokenlizer = Tokenlizer.form_vocab(vocab, merges, special_tokens)

with open(data_dir / "TinyStoriesV2-GPT4-valid.txt", "r", encoding="utf-8") as f:
    tiny_story = f.read()
with open(data_dir / "owt_valid_sample_20MiB.txt", "r", encoding="utf-8") as f:
    owt = f.read()

print("Start tokenlizing tiny story")
tiny_story_encoded = tokenlizer.encode(tiny_story)
np.save(
    script_dir / "tiny_story_valid_tokens.npy",
    np.asarray(tiny_story_encoded, dtype=np.uint16),
)
print("Tokenlizing finished tiny story")
print(f"tiny_story:\nbefore {len(tiny_story)}\nafter {len(tiny_story_encoded)}")
print("Start detokenlizing tiny story")
tiny_story_decoded = tokenlizer.decode(tiny_story_encoded)
print("Tokenlizing definished tiny story")
print(f"cmp {tiny_story == tiny_story_decoded}")

print("Start tokenlizing owt")
owt_encoded = tokenlizer.encode(owt)
np.save(
    script_dir / "owt_valid_sample_20MiB_tokens.npy",
    np.asarray(owt_encoded, dtype=np.uint16),
)
print("Tokenlizing finished owt")
print(f"owt: {len(owt)} vs {len(owt_encoded)}")
owt_decoded = tokenlizer.decode(owt_encoded)
print("Tokenlizing definished tiny story")
print(f"cmp {owt == owt_decoded}")
