from transformers import AutoTokenizer, AutoModelForCausalLM
from app.prompt import SYSTEM_PROMPT

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto",
)


def generate_response(user_input):
    prompt = SYSTEM_PROMPT + "\nUser: " + user_input + "\nAssistant:"

    inputs = tokenizer(prompt, return_tensors="pt")
    model_device = next(model.parameters()).device
    inputs = {key: value.to(model_device) for key, value in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Assistant:" in decoded:
        return decoded.split("Assistant:", 1)[1].strip()

    return decoded
