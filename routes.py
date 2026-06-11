# from fastapi import APIRouter, HTTPException, status
# from models import Message, ChatRequest, SummarizeRequest
# from services import greet_user, dummy_summary, square_number, divide_numbers

# router = APIRouter()


# @router.get("/")
# def home():
#     return {
#         "message": "Welcome to my GenAI learning journey"
#     }


# @router.get("/hello")
# def hello(name: str):
#     return {
#         "message": f"Hello {name}"
#     }


# @router.get("/add")
# def add(a: int, b: int):
#     return {
#         "result": a + b
#     }


# @router.post("/chat")
# def chat(message: Message):
#     return {
#         "received": message.text
#     }


# @router.post("/personal-chat", status_code=status.HTTP_200_OK)
# def personal_chat(request: ChatRequest):
#     return {
#         "reply": greet_user(request.name, request.message)
#     }


# @router.post("/summarize", status_code=status.HTTP_200_OK)
# def summarize(request: SummarizeRequest):
#     return {
#         "summary": dummy_summary(request.text),
#         "original_length": len(request.text)
#     }


# @router.get("/square")
# def square(num: int):
#     try:
#         return {
#             "result": square_number(num)
#         }
#     except ValueError as e:
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail=str(e)
#         )


# @router.get("/divide")
# def divide(a: float, b: float):
#     try:
#         return {
#             "result": divide_numbers(a, b)
#         }
#     except ValueError as e:
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail=str(e)
#         )


# @router.get("/about")
# def about():
#     return {
#         "name": "Sneh's GenAI App",
#         "version": "1.0"
#     }

from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse
from models import Message, ChatRequest, SummarizeRequest, MemoryChatRequest
from services import (
    greet_user,
    dummy_summary,
    square_number,
    divide_numbers,
    call_local_llm,
    chat_with_memory,
)
from prompts.summarizer import build_summarization_prompt

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
INDEX_FILE = STATIC_DIR / "index.html"

router = APIRouter()


@router.get("/")
def home():
    return FileResponse(INDEX_FILE)


@router.get("/api")
def api_root():
    return {
        "message": "Welcome to my GenAI learning journey"
    }


@router.get("/hello")
def hello(name: str):
    return {
        "message": f"Hello {name}"
    }


@router.get("/add")
def add(a: int, b: int):
    return {
        "result": a + b
    }


@router.post("/chat")
def chat(message: Message):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": message.text
        }
    ]

    answer = call_local_llm(messages)

    return {
        "answer": answer
    }


@router.post("/memory-chat")
def memory_chat(request: MemoryChatRequest):
    answer = chat_with_memory(request.session_id, request.message)

    return {
        "session_id": request.session_id,
        "answer": answer
    }


@router.post("/personal-chat", status_code=status.HTTP_200_OK)
def personal_chat(request: ChatRequest):
    prompt = f"User name is {request.name}. User message: {request.message}"
    messages = [
        {
            "role": "system",
            "content": "You are a friendly Python tutor."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    answer = call_local_llm(messages)

    return {
        "reply": answer
    }


@router.post("/summarize", status_code=status.HTTP_200_OK)
def summarize(request: SummarizeRequest):
   
    prompt = build_summarization_prompt(request.text)

    messages = [
        {
            "role": "system",
            "content": "You are a concise summarization assistant. Summarize in 3 bullet points."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    answer = call_local_llm(messages)

    return {
        "summary": answer,
        "original_length": len(request.text)
    }


@router.get("/square")
def square(num: int):
    try:
        return {
            "result": square_number(num)
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get("/divide")
def divide(a: float, b: float):
    try:
        return {
            "result": divide_numbers(a, b)
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get("/about")
def about():
    return {
        "name": "Amex's GenAI App",
        "version": "1.0"
    }