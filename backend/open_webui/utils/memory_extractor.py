"""
Long-term memory auto-extraction and deduplication.

After each assistant response, extracts memorable facts about the user
from the conversation and stores them in the memory system with deduplication.
"""

import json
import logging
import asyncio
from typing import Optional

from open_webui.models.memories import Memories
from open_webui.retrieval.vector.factory import VECTOR_DB_CLIENT
from open_webui.utils.chat import generate_chat_completion
from open_webui.utils.task import get_task_model_id
from open_webui.utils.misc import get_last_user_message, get_last_assistant_message

log = logging.getLogger(__name__)

MEMORY_EXTRACTION_PROMPT = """You are analyzing a conversation to extract facts worth remembering long-term about the user.

Extract ONLY concrete, specific, and useful facts about the user from the last exchange.
Focus on:
- Personal facts (name, age, location, family)
- Work/professional context (job, company, field, skills)
- Preferences and habits
- Ongoing projects or goals
- Important decisions or situations they described

Rules:
- Write each fact as a short, clear statement starting with "User" (e.g. "User is a backend developer")
- Do NOT extract what the user asked, only facts ABOUT the user
- Do NOT duplicate vague or trivial information
- If nothing worth remembering: return []
- Return ONLY a valid JSON array of strings, no explanation, no markdown

Conversation:
{conversation}

JSON array of facts to remember:"""

# Similarity threshold above which we consider a memory a duplicate
# ChromaDB distance: lower = more similar. 0 = identical, 2 = opposite.
DEDUP_DISTANCE_THRESHOLD = 0.35


def _format_conversation_for_extraction(messages: list) -> str:
    """Format the last 6 messages for the extraction prompt."""
    relevant = []
    for msg in messages:
        role = msg.get("role", "")
        if role not in ("user", "assistant"):
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            # Multimodal content — grab text parts only
            content = " ".join(
                p.get("text", "") for p in content if p.get("type") == "text"
            )
        if content and content.strip():
            relevant.append({"role": role, "content": content.strip()})

    # Take only last 6 messages to keep extraction focused
    recent = relevant[-6:]
    return "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in recent
    )


async def _call_llm_for_extraction(request, model_id: str, conversation: str, user) -> list[str]:
    """Call the LLM to extract memorable facts from the conversation."""
    models = request.app.state.MODELS
    task_model_id = get_task_model_id(
        model_id,
        request.app.state.config.TASK_MODEL,
        request.app.state.config.TASK_MODEL_EXTERNAL,
        models,
    )

    if task_model_id not in models:
        # Fallback to the chat model if task model is unavailable
        task_model_id = model_id

    if task_model_id not in models:
        log.warning(f"memory_extractor: model {task_model_id} not found, skipping extraction")
        return []

    prompt = MEMORY_EXTRACTION_PROMPT.format(conversation=conversation)

    payload = {
        "model": task_model_id,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_completion_tokens": 512,
        "temperature": 0.1,
        "metadata": {
            "task": "memory_extraction",
        },
    }

    try:
        res = await generate_chat_completion(request, payload, user)
        if not res or not isinstance(res, dict):
            return []

        content = (
            res.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )

        # Strip markdown code fences if present
        if content.startswith("```"):
            content = content[content.find("[") : content.rfind("]") + 1]

        facts = json.loads(content)
        if isinstance(facts, list):
            return [f for f in facts if isinstance(f, str) and f.strip()]
    except Exception as e:
        log.debug(f"memory_extractor: LLM extraction failed: {e}")

    return []


async def _is_duplicate(request, user_id: str, fact: str, user) -> Optional[str]:
    """
    Check if a similar memory already exists.
    Returns the existing memory ID if duplicate, None otherwise.
    """
    try:
        collection_name = f"user-memory-{user_id}"
        existing = Memories.get_memories_by_user_id(user_id)
        if not existing:
            return None

        vector = await request.app.state.EMBEDDING_FUNCTION(fact, user=user)

        results = VECTOR_DB_CLIENT.search(
            collection_name=collection_name,
            vectors=[vector],
            limit=1,
        )

        if not results or not hasattr(results, "distances"):
            return None

        if (
            results.distances
            and results.distances[0]
            and results.distances[0][0] < DEDUP_DISTANCE_THRESHOLD
        ):
            # Close enough — treat as duplicate
            if results.ids and results.ids[0]:
                return results.ids[0][0]

    except Exception as e:
        log.debug(f"memory_extractor: dedup check failed: {e}")

    return None


async def _upsert_memory(request, user_id: str, memory_id: Optional[str], content: str, user):
    """Insert a new memory or update an existing one, with vector re-embedding."""
    try:
        if memory_id:
            # Update existing
            memory = Memories.update_memory_by_id_and_user_id(memory_id, user_id, content)
        else:
            memory = Memories.insert_new_memory(user_id, content)

        if not memory:
            return

        vector = await request.app.state.EMBEDDING_FUNCTION(memory.content, user=user)
        VECTOR_DB_CLIENT.upsert(
            collection_name=f"user-memory-{user_id}",
            items=[
                {
                    "id": memory.id,
                    "text": memory.content,
                    "vector": vector,
                    "metadata": {
                        "created_at": memory.created_at,
                        "updated_at": memory.updated_at,
                    },
                }
            ],
        )
        action = "updated" if memory_id else "added"
        log.info(f"memory_extractor: {action} memory for user {user_id}: {content[:80]}")
    except Exception as e:
        log.debug(f"memory_extractor: upsert failed: {e}")


async def extract_and_store_memories(request, messages: list, model_id: str, user):
    """
    Main entry point: extract facts from conversation and store them.
    Designed to run as a background task — does not raise exceptions.
    """
    if not getattr(request.app.state.config, "ENABLE_MEMORIES", False):
        return

    try:
        conversation = _format_conversation_for_extraction(messages)
        if not conversation:
            return

        facts = await _call_llm_for_extraction(request, model_id, conversation, user)
        if not facts:
            log.debug("memory_extractor: no facts extracted from this turn")
            return

        log.info(f"memory_extractor: extracted {len(facts)} fact(s) for user {user.id}")

        # Process facts sequentially to avoid vector DB race conditions
        for fact in facts:
            duplicate_id = await _is_duplicate(request, user.id, fact, user)
            await _upsert_memory(request, user.id, duplicate_id, fact, user)

    except Exception as e:
        log.debug(f"memory_extractor: unexpected error: {e}")