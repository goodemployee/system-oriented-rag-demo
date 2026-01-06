from typing import Any
from fastapi import APIRouter, Request

router = APIRouter(prefix="/api")

@router.post("/ask")
async def ask(request: Request, body: str) -> dict[str, Any]:
    usecase = request.app.state.ask_question_usecase
    return usecase.execute(body)
