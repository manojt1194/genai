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

from fastapi import APIRouter, HTTPException, status
from models import Message, ChatRequest, SummarizeRequest
from services import (
    greet_user,
    dummy_summary,
    square_number,
    divide_numbers,
    call_local_llm,
)

router = APIRouter()


@router.get("/")
def home():
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
    # Lesson 8: now this uses the local model
    answer = call_local_llm(message.text)
    return {
        "answer": answer
    }


@router.post("/personal-chat", status_code=status.HTTP_200_OK)
def personal_chat(request: ChatRequest):
    prompt = f"User name is {request.name}. User message: {request.message}"
    answer = call_local_llm(
        prompt,
        system_prompt="You are a friendly Python tutor."
    )
    return {
        "reply": answer
    }


@router.post("/summarize", status_code=status.HTTP_200_OK)
def summarize(request: SummarizeRequest):
    prompt = f"Summarize this text in 3 bullet points:\n\n{request.text}"
    answer = call_local_llm(
        prompt,
        system_prompt="You are a concise summarization assistant."
    )
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
        "name": "Sneh's GenAI App",
        "version": "1.0"
    }