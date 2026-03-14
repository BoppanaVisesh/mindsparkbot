from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from app.prompt import SYSTEM_PROMPT

model_name = "microsoft/phi-2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


def generate_response(user_input):
    prompt = SYSTEM_PROMPT + "\nUser: " + user_input + "\nAssistant:"

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_length=200,
        do_sample=True,
        temperature=0.7,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response
