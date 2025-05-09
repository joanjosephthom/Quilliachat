import os
import google.generativeai as genai
from typing import List, Dict, Any, Optional
from utils import build_token_limited_context

try:
    import streamlit as st
    _st_secrets = st.secrets
except (ImportError, AttributeError):
    _st_secrets = {}

def get_secret(key, default=""):
    return _st_secrets.get(key, os.environ.get(key, default))

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")  # Or use st.secrets if on Streamlit Cloud

class GeminiFlashLLM:
    def __init__(self, api_key: Optional[str] = None, model: str = "models/gemini-1.5-flash-latest"):
        self.api_key = api_key or GEMINI_API_KEY
        if not self.api_key:
            raise ValueError("Gemini API key is required. Set GEMINI_API_KEY in env or secrets.")
        genai.configure(api_key=self.api_key)
        self.model = model

    def generate_response(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        include_sources: bool = True,
        max_sources: int = 2,
        snippet_len: int = 200,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> str:
        context_text, used_chunks = build_token_limited_context(
            context_chunks, max_tokens=8000, model_name=self.model
        )
        prompt = self._create_prompt(query, context_text)
        model = genai.GenerativeModel(self.model)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
        )
        answer = response.text.strip()
        if include_sources:
            sources_text = self._format_sources(context_chunks, max_sources, snippet_len)
            if sources_text:
                return f"{answer}\n\n---\n\n**Sources:**\n{sources_text}"
            else:
                return answer
        else:
            return answer

    def stream_generate_response(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ):
        context_text, used_chunks = build_token_limited_context(
            context_chunks, max_tokens=8000, model_name=self.model
        )
        prompt = self._create_prompt(query, context_text)
        model = genai.GenerativeModel(self.model)
        stream = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
            stream=True
        )
        for chunk in stream:
            if hasattr(chunk, "text"):
                yield chunk.text

    def _create_prompt(self, query: str, context: str) -> str:
        return (
            f"You are a helpful assistant answering questions using only the provided context, which is extracted from a document.\n\n"
            f"GUIDELINES:\n"
            f"- Respond based on the context. If the answer is not directly stated but can be reasonably inferred, provide your best answer and explain your reasoning.\n"
            f"- Do **not** use external knowledge or make assumptions.\n"
            f"- If the answer isn't present or is unclear, respond: \"I don't have enough information to answer this question.\"\n"
            f"- Where possible, **quote relevant text** or mention the **section title** or **page** (if available).\n"
            f"- Be concise and accurate. Use bullet points for multi-part answers.\n"
            f"\n---\n\n"
            f"CONTEXT (Extracted from Document):\n"
            f"\"\"\"\n{context}\n\"\"\"\n\n"
            f"QUESTION:\n{query}\n\n"
            f"ANSWER:"
        )

    def _format_sources(
        self,
        chunks: List[Dict[str, Any]],
        max_sources: int = 2,
        snippet_len: int = 200
    ) -> str:
        if not chunks:
            return ""
        lines = []
        for i, chunk in enumerate(chunks[:max_sources]):
            section = chunk.get("section", "")
            page = chunk.get("page", "")
            doc = chunk.get("document", "")
            text = chunk.get("text", "")
            snippet = text.strip().split(".")[0][:snippet_len] + "..."
            line = f"**Source {i+1}:** {doc}"
            if section:
                line += f" | *Section:* {section}"
            if page:
                line += f" | *Page:* {page}"
            line += f"\n> {snippet}"
            lines.append(line)
        return "\n\n".join(lines)
