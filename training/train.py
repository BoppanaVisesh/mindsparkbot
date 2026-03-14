from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

model_name = "microsoft/phi-2"

dataset = load_dataset("json", data_files="datasets/custom_dataset.json")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


def tokenize_function(example):
    return tokenizer(example["instruction"], truncation=True)


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
