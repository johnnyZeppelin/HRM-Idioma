import os
import json
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

def prepare_nlp_dataset_split(
    dataset_name: str,
    dataset_config: str = None,   # <-- new
    text_field: str = "text",
    model_name: str = "gpt2",
    output_dir: str = "../data/my-nlp-dataset",
    seq_len: int = 128,
    group_size: int = 1000,
    test_size: float = 0.1,
):
    """
    Converts a NLP dataset into the Sudoku-style PuzzleDataset format.
    Long texts are split into `seq_len` chunks.
    Chunks are grouped by `group_size` to form group_indices.
    """
    # 1. Load dataset from HuggingFace
    # dataset = load_dataset(dataset_name)
    if dataset_config is None:
        dataset = load_dataset(dataset_name)
    else:
        dataset = load_dataset(dataset_name, dataset_config)
    # Changed Above
    if "train" not in dataset:
        dataset = dataset["train"].train_test_split(test_size=test_size)
    else:
        dataset = {
            "train": dataset["train"],
            "test": dataset["test"] if "test" in dataset else dataset["train"].train_test_split(test_size=test_size)["test"]
        }

    # 2. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    # 3. Process each split
    for split in ["train", "test"]:
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        all_inputs = []
        all_labels = []
        puzzle_indices = [0]
        puzzle_identifiers = []
        group_indices = [0]

        # Iterate over all texts
        for i, example in enumerate(dataset[split]):
            text = example[text_field]
            enc = tokenizer.encode(text)
            if not enc:
                continue

            # Split into seq_len chunks
            for start in range(0, len(enc), seq_len):
                chunk = enc[start:start+seq_len]
                if len(chunk) < seq_len:
                    chunk = chunk + [pad_id] * (seq_len - len(chunk))

                all_inputs.append(chunk)
                all_labels.append(chunk)
                puzzle_identifiers.append(0)  # Single identifier
                puzzle_indices.append(len(all_inputs))

                # Create group indices based on configurable group_size
                if len(all_inputs) % group_size == 0:
                    group_indices.append(len(all_inputs))

        # Final group index
        if group_indices[-1] != len(all_inputs):
            group_indices.append(len(all_inputs))

        # Convert to numpy arrays
        all_inputs = np.array(all_inputs, dtype=np.int32)
        all_labels = np.array(all_labels, dtype=np.int32)
        puzzle_indices = np.array(puzzle_indices, dtype=np.int64)
        puzzle_identifiers = np.array(puzzle_identifiers, dtype=np.int32)
        group_indices = np.array(group_indices, dtype=np.int64)

        # dataset.json
        dataset_json = {
            "sets": ["all"],
            "pad_id": pad_id,
            "blank_identifier_id": 0,
            "ignore_label_id": None
        }

        # Save files
        with open(os.path.join(split_dir, "dataset.json"), "w") as f:
            json.dump(dataset_json, f)
        np.save(os.path.join(split_dir, "all__inputs.npy"), all_inputs)
        np.save(os.path.join(split_dir, "all__labels.npy"), all_labels)
        np.save(os.path.join(split_dir, "all__puzzle_indices.npy"), puzzle_indices)
        np.save(os.path.join(split_dir, "all__puzzle_identifiers.npy"), puzzle_identifiers)
        np.save(os.path.join(split_dir, "all__group_indices.npy"), group_indices)

    # identifiers.json
    with open(os.path.join(output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)

    print(f"✅ Dataset saved to {output_dir}")


# Example usage:
prepare_nlp_dataset_split("wikitext", dataset_config="wikitext-2-raw-v1", text_field="text", model_name="gpt2", seq_len=128, group_size=1000)
