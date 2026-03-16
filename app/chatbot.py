import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from app.prompt import SYSTEM_PROMPT

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
torch.set_num_interop_threads(1)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto",
    attn_implementation="sdpa",
)
model.eval()
model.config.use_cache = True
MODEL_DEVICE = next(model.parameters()).device


def generate_response(user_input):
    prompt = SYSTEM_PROMPT + "\nUser: " + user_input + "\nAssistant:"

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(MODEL_DEVICE, non_blocking=True) for key, value in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Assistant:" in decoded:
        return decoded.split("Assistant:", 1)[1].strip()

    return decoded
