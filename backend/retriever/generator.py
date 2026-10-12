"""
Answer Generation with Citations using Groq (Llama 3.3 70B).

Prompt design:
1. Numbered sources [1], [2], ... — enables precise citation
2. Each source includes file path + line numbers
3. Explicit "only from context" instruction — reduces hallucination
"""

import os


class Generator:
    """
    Generates natural language answers from retrieved code chunks using Groq.
    Uses llama-3.3-70b-versatile via Groq's free inference tier.
    """

    def __init__(self, api_key: str | None = None):
        """
        Initialize the Groq client.

        Args:
            api_key: Groq API key override. Falls back to GROQ_API_KEY env var.
        """
        from groq import Groq

        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Groq API key required. Get a FREE key at: https://console.groq.com/keys\n"
                "Set it as GROQ_API_KEY in your .env file."
            )

        self.client = Groq(api_key=self.api_key)
        self.provider = "groq"
        self.default_model = "llama-3.3-70b-versatile"

    def build_prompt(self, query: str, retrieved_chunks: list) -> str:
        """Construct the RAG prompt with numbered, cited sources."""
        context_block = ""
        for i, chunk in enumerate(retrieved_chunks, start=1):
            p = chunk["payload"]
            context_block += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SOURCE [{i}]
Location: {p['context_header']}
Language: {p.get('language', 'unknown')}
Type: {p.get('node_type', 'unknown')}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{p['raw_text']}

"""

        prompt = f"""You are CodeLens, an expert software engineer specializing in code comprehension.

Your job is to answer questions ONLY using the retrieved code context provided to you.

## Core Principles

1. Base every answer on the retrieved code.
2. Never invent functionality that is not supported by the retrieved code.
3. If information is missing, explicitly say it is not present in the retrieved context.
4. Distinguish between:
   - Direct observations from the code
   - Reasonable inferences based on the code
5. Be confident when the code clearly supports a conclusion.
6. Do not be unnecessarily defensive or repeatedly say:
   - "appears to be"
   - "might be"
   - "cannot fully determine"
   unless the retrieved code genuinely lacks enough information.

---

## Project Overview Questions

When the user asks:

- What is this project?
- Explain this repository.
- What does this code do?
- Summarize this project.

First identify the major components from the retrieved code.

If no README or documentation is retrieved, infer the project's purpose from:
- file names
- exported functions
- classes
- APIs
- business logic
- comments
- naming conventions

Do NOT apologize for missing documentation.

Instead write:

### Project Overview

<State what the retrieved code implements.>

### Main Components

<List the major modules/features discovered from the code.>

### How It Works

Explain how the components interact.

### Notes

Mention only genuinely missing information if it affects the answer.

Example:

"The retrieved code focuses on color generation logic. No documentation or HTML describing the user interface was retrieved."

Do NOT add a generic "Limitations" section.

---

## Function/Class Questions

When explaining code:

1. Purpose
2. Inputs
3. Outputs
4. Internal logic
5. Important design decisions
6. Time complexity (if applicable)

---

## Architecture Questions

Explain using a top-down approach.

Project
↓
Modules
↓
Classes
↓
Functions
↓
Execution Flow

Use bullet points and diagrams whenever useful.

---

## Code References

Whenever making a statement, cite the supporting source.

For example:

copyRGBToClipboard() copies the RGB value to the clipboard. [script.js:39-44]

Do not mention code that was not retrieved.

---

## Inference Rules

Reasonable inferences are allowed only when strongly supported by the retrieved code.

Example:

Observed:
- RGB values update the background
- Hex values are generated randomly

Inference:
"This project is a color generator utility."

Do not infer technologies, architecture, or features without evidence.

---

## Writing Style

- Be concise but complete.
- Prefer technical clarity over verbosity.
- Avoid unnecessary hedging.
- Avoid repeating the same disclaimer.
- Sound like a senior software engineer explaining the code to another engineer.

Never fabricate missing information.
Never answer beyond the retrieved context.

RETRIEVED SOURCES:
{context_block}

USER QUESTION: {query}

ANSWER:"""

        return prompt

    def generate(
        self,
        query: str,
        retrieved_chunks: list,
        model: str | None = None,
        max_tokens: int = 1024
    ) -> dict:
        """
        Generate a cited answer from retrieved code chunks.

        Args:
            query: The user's question
            retrieved_chunks: List of chunk dicts from the reranker
            model: Model override (Groq model name)
            max_tokens: Maximum response length

        Returns:
            Dict with answer text, structured citations, and metadata
        """
        prompt = self.build_prompt(query, retrieved_chunks)
        model_name = model or self.default_model

        chat_completion = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
            max_tokens=max_tokens,
            temperature=0.3,
        )

        answer_text = chat_completion.choices[0].message.content
        usage = {
            "input_tokens": getattr(chat_completion.usage, "prompt_tokens", 0),
            "output_tokens": getattr(chat_completion.usage, "completion_tokens", 0),
        }

        citations = []
        for i, chunk in enumerate(retrieved_chunks, start=1):
            p = chunk["payload"]
            raw_text = p.get("raw_text", "")
            citations.append({
                "index": i,
                "file_path": p["file_path"],
                "start_line": p["start_line"],
                "end_line": p["end_line"],
                "snippet": raw_text[:200] + "..." if len(raw_text) > 200 else raw_text,
                "full_code": raw_text,
                "node_type": p.get("node_type", "unknown"),
                "parent_class": p.get("parent_class"),
                "language": p.get("language", "unknown"),
                "rerank_score": chunk.get("rerank_score", 0.0),
                "rrf_score": chunk.get("rrf_score", 0.0),
            })

        return {
            "answer": answer_text,
            "citations": citations,
            "model_used": model_name,
            "provider": self.provider,
            "chunks_retrieved": len(retrieved_chunks),
            "query": query,
            "usage": usage,
        }
