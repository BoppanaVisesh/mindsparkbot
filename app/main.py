from app.chatbot import chat


def main() -> None:
    print("MindSpark chatbot backend is ready.")
    user_input = input("You: ")
    print(f"Bot: {chat(user_input)}")


if __name__ == "__main__":
    main()
