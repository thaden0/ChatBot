from fastapi import APIRouter, Depends
from schemas.chat import ChatMessageCreate, ChatMessageResponse
from auth import require_bearer
from orchestrators.chat import ChatOrchestrator

router = APIRouter(
    prefix="/chat",
    tags=["chat"],
    dependencies=[Depends(require_bearer)],
)

# Initialize the chat orchestrator
chat_orchestrator = ChatOrchestrator()

@router.post("/messages", response_model=ChatMessageResponse, summary="Submit a chat message")
async def create_message(payload: ChatMessageCreate) -> ChatMessageResponse:
    """
    Create a new chat message and get an AI-generated response.
    
    Args:
        payload: The chat message request
        
    Returns:
        ChatMessageResponse with the AI-generated response
    """
    return await chat_orchestrator.chat(payload)
