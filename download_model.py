from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TARGET_DIR = Path("models") / "qwen2.5-0.5b-instruct"


def main() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"Downloading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype="auto",
    )

    print(f"Saving tokenizer to: {TARGET_DIR}")
    tokenizer.save_pretrained(TARGET_DIR)

    print(f"Saving model to: {TARGET_DIR}")
    model.save_pretrained(TARGET_DIR)

    print("Qwen download complete.")


if __name__ == "__main__":
    main()
