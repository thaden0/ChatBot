from fastapi import APIRouter, Depends
from schemas.chat import ChatMessageCreate, ChatMessageResponse
from auth import require_bearer
from services.llm import LLMService


router = APIRouter(
    prefix="/chat",
    tags=["chat"],
    dependencies=[Depends(require_bearer)],
)

llm_service = LLMService()

@router.post("/messages", response_model=ChatMessageResponse, summary="Submit a chat message")
async def create_message(payload: ChatMessageCreate) -> ChatMessageResponse:
    response_message = await llm_service.achat(payload.message)
    return ChatMessageResponse(session=payload.session, message=response_message)
