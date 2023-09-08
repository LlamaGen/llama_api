from fastapi import FastAPI

from app.model import LLama
import os

app = FastAPI()

model = LLama(os.environ.get("MODEL_PATH"))


@app.get("/message")
async def get_message(message: str) -> dict[str, str]:
    return {"message": model.interact(message)}
