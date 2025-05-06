"""Module for generating responses using DeepSeek R1 via OpenRouter."""

import os
import requests
from typing import List, Dict, Any, Optional
from utils import build_token_limited_context
# Try to import st.secrets for Streamlit Cloud compatibility
try:
    import streamlit as st
    _st_secrets = st.secrets
except (ImportError, AttributeError):
    _st_secrets = {}

def get_secret(key, default=""):
    return _st_secrets.get(key, os.environ.get(key, default))

# OpenRouter settings
OPENROUTER_API_KEY = get_secret("OPENROUTER_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek/deepseek-r1"  # <-- Correct OpenRouter model name

class DeepSeekR1:
    """
    Wrapper for DeepSeek R1 LLM via OpenRouter API.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = DEEPSEEK_MODEL):
        self.api_key = api_key or OPENROUTER_API_KEY
        if not self.api_key:
            raise ValueError("OpenRouter API key is required. Set OPENROUTER_API_KEY in secrets or env.")
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
        """
        Generate a response to a query using the provided context chunks.
        """
        # Format the context
        context_text, used_chunks = build_token_limited_context(
        context_chunks, max_tokens=8000, model_name=self.model
    )
        # Create the prompt
        prompt = self._create_prompt(query, context_text)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            answer = data["choices"][0]["message"]["content"].strip()

            if include_sources:
                sources_text = self._format_sources(context_chunks, max_sources, snippet_len)
                if sources_text:
                    return f"{answer}\n\n---\n\n**Sources:**\n{sources_text}"
                else:
                    return answer
            else:
                return answer

        except Exception as e:
            return f"Error communicating with OpenRouter API: {e}"

    def _create_prompt(self, query: str, context: str) -> str:
        return (
            f"You are a helpful assistant answering questions using only the provided context, which is extracted from a document.\n\n"
            f"GUIDELINES:\n"
            f"- Respond based on the context. If the answer is not directly stated but can be reasonably inferred, provide your best answer and explain your reasoning .\n"
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
