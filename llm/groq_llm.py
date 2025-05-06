"""Module for generating responses using Groq API with LLama models."""

import os
from typing import List, Dict, Any, Optional
import groq
from utils import build_token_limited_context
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GROQ_API_KEY, LLM_MODEL

class GroqLLM:
    """Generate responses using Groq API with LLama models."""

    def __init__(self, api_key: Optional[str] = None, model: str = LLM_MODEL):
        """Initialize the Groq client with API key and model."""
        self.api_key = api_key or GROQ_API_KEY or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key is required. Set it in config.py or as an environment variable.")

        self.model = model
        self.client = groq.Client(api_key=self.api_key)

    def generate_response(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        include_sources: bool = True,
        max_sources: int = 2,
        snippet_len: int = 200,
    ) -> str:
        """Generate a response to a query using the provided context chunks."""
        # Format the context
        context_text, used_chunks = build_token_limited_context(
        context_chunks, max_tokens=8000, model_name=self.model
        )

        # Create the prompt
        prompt = self._create_prompt(query, context_text)

        # Generate the response
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1024
        )

        answer = response.choices[0].message.content.strip()

        if include_sources:
            sources_text = self._format_sources(context_chunks, max_sources, snippet_len)
            if sources_text:
                return f"{answer}\n\n---\n\n**Sources:**\n{sources_text}"
            else:
                return answer
        else:
            return answer

    def _create_prompt(self, query: str, context: str) -> str:
        """Create a highly structured, reference-aware prompt for grounded QA."""
        return f"""
You are a helpful assistant answering questions using only the provided context, which is extracted from a document.

GUIDELINES:
- Respond based on the context. If the answer is not directly stated but can be reasonably inferred, provide your best answer and explain your reasoning.
- Do **not** use external knowledge or make assumptions.
- If the answer isn't present or is unclear, respond: "I don't have enough information to answer this question."
- Where possible, **quote relevant text** or mention the **section title** or **page** (if available).
- Be concise and accurate. Use bullet points for multi-part answers.

---

CONTEXT (Extracted from Document):
\"\"\"
{context}
\"\"\"

QUESTION:
{query}

ANSWER:
"""

    def _format_sources(
        self,
        chunks: List[Dict[str, Any]],
        max_sources: int = 2,
        snippet_len: int = 200
    ) -> str:
        """Format source chunks for display, showing only a snippet."""
        if not chunks:
            return ""
        lines = []
        for i, chunk in enumerate(chunks[:max_sources]):
            section = chunk.get("section", "")
            page = chunk.get("page", "")
            doc = chunk.get("document", "")
            text = chunk.get("text", "")
            # Get a short snippet (first sentence or first N chars)
            snippet = text.strip().split(".")[0][:snippet_len] + "..."
            line = f"**Source {i+1}:** {doc}"
            if section:
                line += f" | *Section:* {section}"
            if page:
                line += f" | *Page:* {page}"
            line += f"\n> {snippet}"
            lines.append(line)
        return "\n\n".join(lines)
