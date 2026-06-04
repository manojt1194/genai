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


def call_local_llm(user_message: str, system_prompt: str = "You are a helpful assistant.") -> str:
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
    }

    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()

    return data["message"]["content"]