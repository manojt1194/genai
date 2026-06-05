# def greet_user(name: str, message: str) -> str:
#     return f"Hello {name}, you said: {message}"


# def dummy_summary(text: str) -> str:
#     return "Dummy summary generated"


# def square_number(num: int) -> int:
#     if num < 0:
#         raise ValueError("num must be a non-negative integer")
#     return num * num


# def divide_numbers(a: float, b: float) -> float:
#     if b == 0:
#         raise ValueError("Division by zero is not allowed")
#     return a / b

import os
import requests


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3")

# In-memory chat history for learning purposes
# This resets when the server restarts.
CHAT_MEMORY = {}


def greet_user(name: str, message: str) -> str:
    return f"Hello {name}, you said: {message}"


def dummy_summary(text: str) -> str:
    return "Dummy summary generated"


def square_number(num: int) -> int:
    if num < 0:
        raise ValueError("num must be a non-negative integer")
    return num * num


def divide_numbers(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("Division by zero is not allowed")
    return a / b


def call_local_llm(messages: list) -> str:
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
    }

    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()

    return data["message"]["content"]


def chat_with_memory(session_id: str, user_message: str) -> str:
    if session_id not in CHAT_MEMORY:
        CHAT_MEMORY[session_id] = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Remember the conversation and answer using context."
            }
        ]

    CHAT_MEMORY[session_id].append(
        {
            "role": "user",
            "content": user_message
        }
    )

    answer = call_local_llm(CHAT_MEMORY[session_id])

    CHAT_MEMORY[session_id].append(
        {
            "role": "assistant",
            "content": answer
        }
    )
    
    print(CHAT_MEMORY)

    return answer