from app.prompt import build_prompt


def chat(user_message: str) -> str:
    prompt = build_prompt(user_message)
    return f"Echo response for prompt: {prompt}"
