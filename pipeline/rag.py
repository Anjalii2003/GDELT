# # pipeline/rag.py
# """
# Phase 10: Generate answers using retrieved chunks + Ollama LLM.
# """

# from openai import OpenAI
# from config import LLM_MODEL, LLM_BASE_URL
# client = OpenAI(
#     base_url=LLM_BASE_URL,
#     api_key="EMPTY"  # required but unused in vLLM
# )

# SYSTEM_PROMPT = """You are a knowledgeable assistant that answers questions about world events
# using information from news articles. Always base your answers on the provided context.
# If the context doesn't contain enough information, say so clearly.
# Always mention your sources at the end."""

# def build_prompt(query: str, chunks: list[dict]) -> str:
#     """Build the RAG prompt by injecting retrieved chunks."""

#     context_parts = []
#     for i, chunk in enumerate(chunks, 1):
#         context_parts.append(
#             f"[Source {i}] {chunk['title']} ({chunk['date_str']}, {chunk['location']})\n"
#             f"URL: {chunk['source_url']}\n"
#             f"{chunk['chunk_text']}\n"
#         )

#     context = "\n---\n".join(context_parts)

#     prompt = f"""Use the following news article excerpts to answer the question.

# CONTEXT:
# {context}

# QUESTION: {query}

# INSTRUCTIONS:
# - Answer based on the context above
# - Be specific and factual
# - At the end, list the source URLs you used
# - If context is insufficient, say "I don't have enough information"

# ANSWER:"""

#     return prompt

# def generate_answer(query: str, chunks: list[dict]) -> dict:
#     """Call vLLM (Mistral) to generate an answer from retrieved chunks."""

#     if not chunks:
#         return {
#             "answer": "No relevant articles found for your query.",
#             "sources": []
#         }

#     prompt = build_prompt(query, chunks)

#     response = client.chat.completions.create(
#         model=LLM_MODEL,
#         messages=[
#             {"role": "system", "content": SYSTEM_PROMPT},
#             {"role": "user",   "content": prompt}
#         ],
#         temperature=0.1,
#         max_tokens=512
#     )

#     answer = response.choices[0].message.content
#     sources = list({chunk["source_url"] for chunk in chunks})

#     return {
#         "answer": answer,
#         "sources": sources,
#         "chunks_used": len(chunks)
#     }

# pipeline/rag.py
"""
Phase 10: Generate answers using retrieved chunks + vLLM (Mistral).
"""

from openai import OpenAI
from config import LLM_MODEL, LLM_BASE_URL

# vLLM OpenAI-compatible client
client = OpenAI(
    base_url=LLM_BASE_URL,
    api_key="EMPTY"  # required but unused
)

SYSTEM_PROMPT = """You are a knowledgeable assistant that answers questions about world events
using information from news articles. Always base your answers on the provided context.
If the context doesn't contain enough information, say so clearly.
Always mention your sources at the end."""


# ----------------------------------
# 🧠 BUILD PROMPT (FIXED)
# ----------------------------------
def build_prompt(query: str, chunks: list[dict]) -> str:
    """
    Build the RAG prompt by grouping chunks by article (deduplicated).
    """

    grouped = {}

    # Group chunks by article URL
    for chunk in chunks:
        url = chunk["source_url"]

        if url not in grouped:
            grouped[url] = {
                "title": chunk["title"],
                "date": chunk["date_str"],
                "location": chunk["location"],
                "chunks": []
            }

        grouped[url]["chunks"].append(chunk["chunk_text"])

    # Build context (one entry per article)
    context_parts = []
    for i, (url, data) in enumerate(grouped.items(), 1):
        combined_text = "\n".join(data["chunks"])

        context_parts.append(
            f"[Source {i}] {data['title']} ({data['date']}, {data['location']})\n"
            f"URL: {url}\n"
            f"{combined_text}\n"
        )

    context = "\n---\n".join(context_parts)

    prompt = f"""Use the following news article excerpts to answer the question.

CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
- Answer based on the context above
- Be specific and factual
- Do NOT list sources explicitly
- If context is insufficient, say "I don't have enough information"

ANSWER:"""

    return prompt


# ----------------------------------
# 🤖 GENERATE ANSWER (FIXED)
# ----------------------------------
def generate_answer(query: str, chunks: list[dict]) -> dict:
    """
    Call vLLM (Mistral) to generate an answer from retrieved chunks.
    """

    if not chunks:
        return {
            "answer": "No relevant articles found for your query.",
            "sources": []
        }

    prompt = build_prompt(query, chunks)

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.1,
        max_tokens=512
    )

    answer = response.choices[0].message.content

    # ✅ Deduplicate sources (clean order)
    sources = []
    seen = set()

    for chunk in chunks:
        url = chunk["source_url"]
        if url not in seen:
            seen.add(url)
            sources.append(url)

    return {
        "answer": answer,
        "sources": sources,
        "chunks_used": len(chunks)
    }