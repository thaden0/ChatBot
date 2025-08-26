# services/llm.py
from __future__ import annotations
from typing import Sequence, Tuple, TypedDict, cast
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSerializable

class PromptVars(TypedDict):
    prompt: str

Chain = RunnableSerializable[PromptVars, str]

class LLMService:
    def __init__(self, model: str = "llama3.2-vision:latest",
                 base_url: str = "http://127.0.0.1:11434",
                 temperature: float = 0.2) -> None:
        messages: Sequence[Tuple[str, str]] = [
            ("system", "You are a concise assistant. Answer briefly and clearly."),
            ("user", "{prompt}"),
        ]
        self._prompt = ChatPromptTemplate.from_messages(messages)  # type: ignore
        self._model = ChatOllama(model=model, base_url=base_url, temperature=temperature)
        self._parser = StrOutputParser()
        self._chain: Chain = cast(Chain, self._prompt | self._model | self._parser)

    def chat(self, prompt: str) -> str:
        return self._chain.invoke(PromptVars(prompt=prompt))

    async def achat(self, prompt: str) -> str:
        return await self._chain.ainvoke(PromptVars(prompt=prompt))
