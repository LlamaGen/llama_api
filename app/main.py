from fastapi import FastAPI

from app.model import LLama

app = FastAPI()

model = LLama("model/gguf-model-q4_1.bin")


@app.get("/message")
async def get_message(message: str) -> dict[str, str]:
    return {"message": model.interact(message)}
