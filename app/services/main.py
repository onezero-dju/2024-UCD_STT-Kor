from app.routes import stt
from fastapi import FastAPI

app = FastAPI()
app.include_router(stt.router)

@app.get("/")
def read_root():
    return {"Hello": "World"}
