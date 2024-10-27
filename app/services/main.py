# from app.services.stt import whisper_pyannote
from fastapi import FastAPI

app = FastAPI()
# app.include_router(whisper_pyannote.router)

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}