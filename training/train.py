from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

dataset = load_dataset("json", data_files="datasets/custom_dataset.json")

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto" if torch.cuda.is_available() else None,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
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
    output_dir="./models/mindspark_model",
    per_device_train_batch_size=1,
    num_train_epochs=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
)

trainer.train()

model.save_pretrained("models/mindspark_model")
tokenizer.save_pretrained("models/mindspark_model")
