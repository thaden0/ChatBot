# services/llm.py
from __future__ import annotations
from typing import Sequence, Tuple, TypedDict, cast, List
from operator import itemgetter

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSerializable, RunnableLambda, RunnablePassthrough

# RAG bits
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

class PromptVars(TypedDict):
    prompt: str

Chain = RunnableSerializable[PromptVars, str]

def _format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)

class LLMService:
    def __init__(
        self,
        model: str = "hf.co/mradermacher/0824-Qwen2.5-0.5B-Instructt-16bit-3E-GGUF:Q3_K_S",
        base_url: str = "http://127.0.0.1:11434",
        temperature: float = 0.2,
        data_dir: str | None = None,
        embed_model: str = "nomic-embed-text",  # any Ollama embedding model you have
        k: int = 4
    ) -> None:
        # Prompt now accepts {context}
        messages: Sequence[Tuple[str, str]] = [
            ("system",
             "You are Leo-Bot, a helpful assistant. You're role is to provide answers to questions asked by the user. Answer them professionally and honestly."
             "If the answer is not in context, say you don't know.\n\nContext:\n{context}"),
            ("user", "{prompt}"),
        ]
        self._prompt = ChatPromptTemplate.from_messages(messages)  # type: ignore
        self._model = ChatOllama(model=model, base_url=base_url, temperature=temperature)
        self._parser = StrOutputParser()

        # Default chain: no retrieval
        base_chain = self._prompt | self._model | self._parser

        if data_dir:
            # 1) Load files (here: *.txt, *.md). Add more loaders if needed.
            loader = DirectoryLoader(
                data_dir,
                glob="**/*.*",
                loader_cls=TextLoader,      # handles .txt/.md; extend if needed
                show_progress=False,        # <- disables tqdm
                use_multithreading=True
            )
            docs = loader.load()
            
            # 2) Chunk
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = splitter.split_documents(docs)

            # 3) Embed + index
            embeddings = OllamaEmbeddings(model=embed_model, base_url=base_url)
            vectordb = FAISS.from_documents(chunks, embeddings)
            retriever = vectordb.as_retriever(search_kwargs={"k": k})

            # 4) Build RAG chain that fills {context} automatically
            rag_inputs = {
                "context": itemgetter("prompt") | retriever | RunnableLambda(_format_docs),
                "prompt": itemgetter("prompt"),
            }
            self._chain: Chain = cast(Chain, rag_inputs | base_chain)
        else:
            self._chain = cast(Chain, base_chain)

    def chat(self, prompt: str) -> str:
        return self._chain.invoke(PromptVars(prompt=prompt))

    async def achat(self, prompt: str) -> str:
        return await self._chain.ainvoke(PromptVars(prompt=prompt))
