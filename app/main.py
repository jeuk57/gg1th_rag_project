# FastAPI 서버. GET /는 채팅 UI를 보여주고, POST /ask는 질문을 받아
# pipeline.answer_question() 결과를 반환한다.
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from rag.pipeline import answer_question

app = FastAPI()


@app.get("/")
def root():
    return FileResponse("app/static/index.html")


class AskRequest(BaseModel):
    question: str


class Source(BaseModel):
    article: str
    content: str


class AskResponse(BaseModel):
    answer: str
    grounded: bool
    sources: list[Source]


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    return answer_question(request.question)