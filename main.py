from fastapi import FastAPI
from routes import router

app = FastAPI(
    title="GenAI Learning API",
    description="FastAPI practice app for learning backend basics before GenAI",
    version="1.0.0"
)

app.include_router(router)