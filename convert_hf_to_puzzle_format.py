import os
import json
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

# ==== CONFIG ====
HF_DATASET = "wikitext"  # or e.g. "wikitext", "openwebtext", "imdb", "ag_news"
HF_SPLIT = "train"       # dataset split
TOKENIZER_NAME = "gpt2"  # or any HF tokenizer
MAX_SEQ_LEN = 128        # model's input length
PAD_ID = 50256           # GPT-2's pad_id equivalent (if no pad, pick unused ID)
IGNORE_LABEL_ID = -100   # standard PyTorch loss ignore index
OUTPUT_DIR = "./data/nlp-converted/train"
SET_NAME = "all"         # name of the set (like "all" in Sudoku)
# =================

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading tokenizer: {TOKENIZER_NAME}")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = PAD_ID

print(f"Loading dataset: {HF_DATASET} [{HF_SPLIT}]")
dataset = load_dataset(HF_DATASET, split=HF_SPLIT)

# Ensure dataset has a 'text' column
if "text" not in dataset.column_names:
    raise ValueError(f"Dataset {HF_DATASET} has no 'text' column!")

all_inputs = []
all_labels = []
puzzle_indices = [0]  # pointer to where each document's examples start
puzzle_identifiers = []
group_indices = [0]   # only 1 group here

example_counter = 0
puzzle_counter = 0

for doc in dataset["text"]:
    if not doc or doc.strip() == "":
        continue
    
    # Tokenize the entire document
    tokens = tokenizer.encode(doc, add_special_tokens=False)
    
    if not tokens:
        continue

    # Break into chunks
    for i in range(0, len(tokens), MAX_SEQ_LEN):
        chunk = tokens[i:i + MAX_SEQ_LEN]
        
        # Pad to fixed length
        if len(chunk) < MAX_SEQ_LEN:
            chunk = chunk + [tokenizer.pad_token_id] * (MAX_SEQ_LEN - len(chunk))

        # For causal LM: labels = same as inputs
        labels = chunk.copy()

        all_inputs.append(chunk)
        all_labels.append(labels)
        puzzle_identifiers.append(puzzle_counter)
        example_counter += 1

    puzzle_counter += 1
    puzzle_indices.append(example_counter)

group_indices.append(puzzle_counter)

# Convert to numpy
all_inputs = np.array(all_inputs, dtype=np.int32)
all_labels = np.array(all_labels, dtype=np.int32)
puzzle_indices = np.array(puzzle_indices, dtype=np.int64)
puzzle_identifiers = np.array(puzzle_identifiers, dtype=np.int32)
group_indices = np.array(group_indices, dtype=np.int64)

# Save npy files
np.save(os.path.join(OUTPUT_DIR, f"{SET_NAME}__inputs.npy"), all_inputs)
np.save(os.path.join(OUTPUT_DIR, f"{SET_NAME}__labels.npy"), all_labels)
np.save(os.path.join(OUTPUT_DIR, f"{SET_NAME}__puzzle_indices.npy"), puzzle_indices)
np.save(os.path.join(OUTPUT_DIR, f"{SET_NAME}__puzzle_identifiers.npy"), puzzle_identifiers)
np.save(os.path.join(OUTPUT_DIR, f"{SET_NAME}__group_indices.npy"), group_indices)

# Save dataset.json metadata
metadata = {
    "sets": [SET_NAME],
    "pad_id": tokenizer.pad_token_id,
    "blank_identifier_id": 0,
    "ignore_label_id": IGNORE_LABEL_ID
}

with open(os.path.join(OUTPUT_DIR, "dataset.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Conversion complete! Saved to {OUTPUT_DIR}")
print(f"Examples: {all_inputs.shape}, Puzzles: {len(puzzle_indices)-1}, Groups: {len(group_indices)-1}")
