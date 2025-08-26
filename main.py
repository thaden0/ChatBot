from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from routes import chat

app = FastAPI()

@app.get("/")
def read_root():
    return RedirectResponse(url="/docs")
app.include_router(chat.router)
