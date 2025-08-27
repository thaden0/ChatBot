# orchestrators/chat.py
from __future__ import annotations
from typing import Optional
from services.llm import LLMService
from schemas.chat import ChatMessageCreate, ChatMessageResponse
import logging

logger = logging.getLogger(__name__)

class ChatOrchestrator:
    """
    Orchestrates chat operations by coordinating between different services.
    Handles business logic for chat message processing.
    """
    
    def __init__(self, llm_service: Optional[LLMService] = None) -> None:
        """
        Initialize the chat orchestrator.
        
        Args:
            llm_service: LLM service instance. If None, creates a new one.
        """
        self._llm_service = llm_service or LLMService()
        logger.info("ChatOrchestrator initialized")
    
    async def chat(self, payload: ChatMessageCreate) -> ChatMessageResponse:
        """
        Process a chat message and generate a response.
        
        Args:
            payload: The chat message request containing session and message
            
        Returns:
            ChatMessageResponse with the AI-generated response
            
        Raises:
            Exception: If there's an error processing the chat request
        """
        try:
            logger.info(f"Processing chat message for session: {payload.session}")
            
            # Call the LLM service to generate a response
            response_message = await self._llm_service.achat(payload.message)
            
            logger.info(f"Generated response for session: {payload.session}")
            
            # Create and return the response
            return ChatMessageResponse(
                session=payload.session,
                message=response_message,
                sender="assistant"
            )
            
        except Exception as e:
            logger.error(f"Error processing chat message for session {payload.session}: {str(e)}")
            raise 