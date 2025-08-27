# orchestrators/chat.py
from __future__ import annotations
from typing import Optional
import os
from services.llm import LLMService
from schemas.chat import ChatMessageCreate, ChatMessageResponse
import logging

logger = logging.getLogger(__name__)

class ChatOrchestrator:
    def __init__(self, llm_service: Optional[LLMService] = None) -> None:
        if llm_service is not None:
            self._llm_service = llm_service
        else:
            self._llm_service = LLMService(
                model="hf.co/mradermacher/0824-Qwen2.5-0.5B-Instructt-16bit-3E-GGUF:Q3_K_S",
                base_url="http://127.0.0.1:11434",
                temperature=0.2,
                data_dir="./data",                  # hardcoded directory
                embed_model="nomic-embed-text",
                k=4,
            )
        logger.info("ChatOrchestrator initialized")

    async def chat(self, payload: ChatMessageCreate) -> ChatMessageResponse:
        try:
            logger.info(f"Processing chat message for session: {payload.session}")
            response_message = await self._llm_service.achat(payload.message)
            logger.info(f"Generated response for session: {payload.session}")
            return ChatMessageResponse(
                session=payload.session,
                message=response_message,
                sender="assistant"
            )
        except Exception as e:
            logger.error(f"Error processing chat message for session {payload.session}: {str(e)}")
            raise
