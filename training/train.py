import argparse
import json
import random
from pathlib import Path
from collections import defaultdict

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_DATASET_PATH = PROJECT_ROOT / "datasets" / "custom_dataset.json"
OUTPUT_MODEL_DIR = PROJECT_ROOT / "models" / "mindspark_model"

# Keep training size manageable while still mixing all sources.
DEFAULT_MAX_SAMPLES = {
    "custom": 5000,
    "empathetic_dialogues": 4000,
    "daily_dialog": 4000,
    "eli5": 4000,
    "go_emotions": 4000,
}

FAST_MAX_SAMPLES = {
    "custom": 1000,
    "empathetic_dialogues": 1000,
    "daily_dialog": 1000,
    "eli5": 0,
    "go_emotions": 1000,
}

EMOTION_LABELS = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]


def _safe_text(value, default=""):
    if value is None:
        return default
    if isinstance(value, list):
        value = " ".join(str(item) for item in value if item is not None)
    text = str(value).strip()
    return text if text else default


def load_local_records(path, max_samples):
    if not path.exists():
        raise FileNotFoundError(
            f"Local dataset file not found: {path}. Run from project root or keep dataset at datasets/custom_dataset.json"
        )

    with path.open("r", encoding="utf-8") as file:
        rows = json.load(file)

    records = []
    for row in rows[: max_samples["custom"]]:
        records.append(
            {
                "instruction": _safe_text(row.get("instruction"), "Respond helpfully and kindly."),
                "input": _safe_text(row.get("input"), ""),
                "response": _safe_text(row.get("response"), ""),
            }
        )
    return records


def map_empathetic_dialogues(dataset, max_samples):
    records = []
    for row in dataset.select(range(min(len(dataset), max_samples["empathetic_dialogues"]))):
        prompt = _safe_text(row.get("prompt") or row.get("instruction"), "Respond empathetically to the user.")
        context = _safe_text(row.get("context") or row.get("input"), "")
        utterance = _safe_text(row.get("utterance") or row.get("response") or row.get("output"), "")
        if utterance:
            records.append(
                {
                    "instruction": prompt,
                    "input": context,
                    "response": utterance,
                }
            )
    return records


def map_daily_dialog(dataset, max_samples):
    records = []
    grouped_turns = defaultdict(list)

    for row in dataset.select(range(min(len(dataset), max_samples["daily_dialog"]))):
        # Original DailyDialog schema: a list of turns in "dialog".
        dialog = row.get("dialog")
        if isinstance(dialog, list) and len(dialog) >= 2:
            turns = [_safe_text(turn) for turn in dialog if _safe_text(turn)]
            for index in range(len(turns) - 1):
                records.append(
                    {
                        "instruction": "Continue this conversation politely.",
                        "input": turns[index],
                        "response": turns[index + 1],
                    }
                )
                if len(records) >= max_samples["daily_dialog"]:
                    return records
            continue

        # Fallback schema often has one utterance per row with a dialog id.
        dialog_id = row.get("dialog_id")
        utterance = _safe_text(row.get("utterance"), "")
        if dialog_id is not None and utterance:
            grouped_turns[dialog_id].append(utterance)

    if len(records) < max_samples["daily_dialog"]:
        for turns in grouped_turns.values():
            for index in range(len(turns) - 1):
                records.append(
                    {
                        "instruction": "Continue this conversation politely.",
                        "input": turns[index],
                        "response": turns[index + 1],
                    }
                )
                if len(records) >= max_samples["daily_dialog"]:
                    return records

    return records


def _first_available(row, keys):
    for key in keys:
        if key in row and row.get(key) not in (None, ""):
            return row.get(key)
    return None


def map_eli5(dataset, max_samples):
    records = []
    for row in dataset.select(range(min(len(dataset), max_samples["eli5"]))):
        question = _safe_text(_first_available(row, ["title", "question", "query", "prompt"]), "")
        answers = row.get("answers", {})
        best_answer = ""

        if isinstance(answers, dict):
            answer_texts = answers.get("text", [])
            if isinstance(answer_texts, list) and answer_texts:
                best_answer = _safe_text(answer_texts[0], "")
        elif isinstance(answers, list) and answers:
            best_answer = _safe_text(answers[0], "")

        if not best_answer:
            best_answer = _safe_text(
                _first_available(row, ["answer", "response", "output", "best_answer", "text"]),
                "",
            )

        if question and best_answer:
            records.append(
                {
                    "instruction": "Explain this in a simple and child-friendly way.",
                    "input": question,
                    "response": best_answer,
                }
            )
    return records


def map_go_emotions(dataset, max_samples):
    records = []
    for row in dataset.select(range(min(len(dataset), max_samples["go_emotions"]))):
        text = _safe_text(row.get("text"), "")
        labels = row.get("labels", [])
        emotion_names = []
        if isinstance(labels, list):
            for label_index in labels:
                if isinstance(label_index, int) and 0 <= label_index < len(EMOTION_LABELS):
                    emotion_names.append(EMOTION_LABELS[label_index])

        detected = ", ".join(emotion_names) if emotion_names else "neutral"
        if text:
            records.append(
                {
                    "instruction": "Identify the emotion and respond supportively.",
                    "input": text,
                    "response": f"Detected emotion: {detected}. Supportive response: I understand how you feel.",
                }
            )
    return records


def pick_split(dataset_dict, preferred_splits):
    for split_name in preferred_splits:
        if split_name in dataset_dict:
            return dataset_dict[split_name]
    first_split = next(iter(dataset_dict.keys()))
    return dataset_dict[first_split]


def load_hf_dataset_with_fallback(dataset_names):
    last_error = None
    for dataset_name in dataset_names:
        try:
            print(f"Loading dataset: {dataset_name}")
            return load_dataset(dataset_name)
        except Exception as error:
            last_error = error
            print(f"Skipping dataset source {dataset_name} ({error})")

    return None


def build_training_records(max_samples, include_eli5=True):
    all_records = []

    all_records.extend(load_local_records(LOCAL_DATASET_PATH, max_samples))

    empathetic_ds = load_hf_dataset_with_fallback(
        ["empathetic_dialogues", "brianist/empathetic_dialogues", "pixelsandpointers/empathetic_dialogues_for_lm"]
    )
    if empathetic_ds is not None:
        empathetic_split = pick_split(empathetic_ds, ["train", "validation", "test"])
        all_records.extend(map_empathetic_dialogues(empathetic_split, max_samples))

    daily_ds = load_hf_dataset_with_fallback(
        ["daily_dialog", "pixelsandpointers/better_daily_dialog", "pixelsandpointers/daily_dialog_w_turn_templates"]
    )
    if daily_ds is not None:
        daily_split = pick_split(daily_ds, ["train", "validation", "test"])
        all_records.extend(map_daily_dialog(daily_split, max_samples))

    if include_eli5 and max_samples["eli5"] > 0:
        eli5_ds = load_hf_dataset_with_fallback(["eli5", "P1ayer-1/eli5", "Pavithree/eli5_split"])
        if eli5_ds is not None:
            eli5_split = pick_split(eli5_ds, ["train_eli5", "train", "validation_eli5", "validation", "test_eli5", "test"])
            all_records.extend(map_eli5(eli5_split, max_samples))

    go_emotions_ds = load_hf_dataset_with_fallback(["go_emotions", "google-research-datasets/go_emotions"])
    if go_emotions_ds is not None:
        go_emotions_split = pick_split(go_emotions_ds, ["train", "validation", "test"])
        all_records.extend(map_go_emotions(go_emotions_split, max_samples))

    random.shuffle(all_records)
    if not all_records:
        raise RuntimeError("No training records were prepared. Check dataset paths and internet connectivity.")
    print(f"Total merged records: {len(all_records)}")
    return all_records



def parse_args():
    parser = argparse.ArgumentParser(description="Train the MindSpark model.")
    parser.add_argument("--fast", action="store_true", help="Use a smaller, faster dataset mix (skips ELI5).")
    parser.add_argument("--skip-eli5", action="store_true", help="Skip ELI5 even in full mode.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    return parser.parse_args()


def main():
    args = parse_args()

    max_samples = FAST_MAX_SAMPLES.copy() if args.fast else DEFAULT_MAX_SAMPLES.copy()
    include_eli5 = not (args.fast or args.skip_eli5)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Local dataset path: {LOCAL_DATASET_PATH}")
    print(f"Fast mode: {args.fast}")
    print(f"Include ELI5: {include_eli5}")

    records = build_training_records(max_samples=max_samples, include_eli5=include_eli5)
    dataset = Dataset.from_list(records)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto" if torch.cuda.is_available() else None,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    def tokenize_function(example):
        text = (
            f"Instruction: {example['instruction']}\n"
            f"Input: {example['input']}\n"
            f"Response: {example['response']}"
        )
        tokenized = tokenizer(text, truncation=True, padding="max_length", max_length=256)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = dataset.map(tokenize_function)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_MODEL_DIR),
        per_device_train_batch_size=1,
        num_train_epochs=args.epochs,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    trainer.train()

    model.save_pretrained(OUTPUT_MODEL_DIR)
    tokenizer.save_pretrained(OUTPUT_MODEL_DIR)


if __name__ == "__main__":
    main()
