"""
Long-term memory auto-extraction with conflict resolution and temporal relevance.

Improvements over v1:
- Processes only the CURRENT exchange (1 user + 1 assistant message), not a window of 6.
  This ensures every turn is captured regardless of conversation length.
- LLM-based conflict arbitration: when a new fact is similar to an existing one,
  asks the model whether to UPDATE, EXTEND, or IGNORE instead of blind deduplication.
- Scope classification: personal / work / preference / general.
- source_date stored per fact for temporal relevance during retrieval.
- Full-conversation scan available as a separate entry point (used on first run or manually).
"""

import json
import logging
import asyncio
import time
from typing import Optional

from open_webui.models.memories import Memories, MemoryModel
from open_webui.retrieval.vector.factory import VECTOR_DB_CLIENT
from open_webui.utils.chat import generate_chat_completion
from open_webui.utils.task import get_task_model_id

log = logging.getLogger(__name__)

# -----------------------------------------------------------------
# Prompts
# -----------------------------------------------------------------

EXTRACTION_PROMPT = """Analyze the conversation exchange below and extract concrete facts worth remembering long-term about the USER (not about the topic discussed).

Focus on:
- Personal facts (name, age, city, family)
- Professional context (role, company, industry, tools used)
- Stated preferences (communication style, format, language)
- Ongoing projects, goals, decisions
- Constraints (budget, deadlines, team size)

Rules:
- Each fact must start with "User" (e.g. "User works as a Python developer at Sber")
- Do NOT extract what the user asked — only facts ABOUT the user
- Do NOT invent facts not stated in the conversation
- Assign a scope: "personal" | "work" | "preference" | "general"
- If nothing worth remembering: return []
- Return ONLY a valid JSON array, no markdown, no explanation

Format:
[
  {{"fact": "User prefers concise answers without markdown", "scope": "preference"}},
  {{"fact": "User is leading a mobile payments project with a Q3 deadline", "scope": "work"}}
]

Conversation:
{conversation}

JSON array:"""

CONFLICT_RESOLUTION_PROMPT = """You are a memory manager. A new fact may conflict with or update an existing memory.

Existing memory: {existing}
New fact: {new_fact}

Decide the action:
- "UPDATE": new fact REPLACES existing (e.g. budget changed, role changed, preference reversed)
- "EXTEND": new fact ADDS information without contradiction (e.g. more details about same topic)
- "IGNORE": new fact is a duplicate or less specific than what is already stored

Respond with ONLY a JSON object:
{{"action": "UPDATE"|"EXTEND"|"IGNORE", "merged": "<merged text if UPDATE or EXTEND, else null>"}}"""

# ChromaDB distance threshold for "close enough to check for conflict"
# Lower = stricter (only very similar memories trigger conflict check)
SIMILARITY_THRESHOLD = 0.45


# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------

def _extract_current_exchange(messages: list) -> str:
    """
    Extract only the LAST user message + LAST assistant message.
    This guarantees every turn is processed regardless of conversation length.
    """
    user_msg = ""
    assistant_msg = ""

    for msg in reversed(messages):
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") for p in content if p.get("type") == "text"
            )
        content = (content or "").strip()
        if not content:
            continue
        if role == "assistant" and not assistant_msg:
            assistant_msg = content
        elif role == "user" and not user_msg:
            user_msg = content
        if user_msg and assistant_msg:
            break

    if not user_msg:
        return ""
    parts = [f"User: {user_msg}"]
    if assistant_msg:
        parts.append(f"Assistant: {assistant_msg}")
    return "\n".join(parts)


def _extract_full_conversation(messages: list) -> str:
    """
    Format the entire conversation for a full-scan extraction.
    Used on first run or when triggered manually.
    """
    result = []
    for msg in messages:
        role = msg.get("role", "")
        if role not in ("user", "assistant"):
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") for p in content if p.get("type") == "text"
            )
        content = (content or "").strip()
        if content:
            result.append(f"{'User' if role == 'user' else 'Assistant'}: {content}")
    return "\n".join(result)


async def _call_llm(request, model_id: str, prompt: str, user, max_tokens: int = 512) -> str:
    """Call the task model with a prompt and return raw text content."""
    models = request.app.state.MODELS
    task_model_id = get_task_model_id(
        model_id,
        request.app.state.config.TASK_MODEL,
        request.app.state.config.TASK_MODEL_EXTERNAL,
        models,
    )
    if task_model_id not in models:
        task_model_id = model_id
    if task_model_id not in models:
        return ""

    payload = {
        "model": task_model_id,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_completion_tokens": max_tokens,
        "temperature": 0.1,
        "metadata": {"task": "memory_extraction"},
    }
    try:
        res = await generate_chat_completion(request, payload, user)
        if res and isinstance(res, dict):
            return (
                res.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
    except Exception as e:
        log.debug(f"memory_extractor: LLM call failed: {e}")
    return ""


async def _extract_facts(request, model_id: str, conversation: str, user) -> list[dict]:
    """Ask the LLM to extract facts from the conversation. Returns list of {fact, scope}."""
    if not conversation:
        return []

    raw = await _call_llm(
        request, model_id,
        EXTRACTION_PROMPT.format(conversation=conversation),
        user,
    )
    if not raw:
        return []

    # Strip markdown fences if present
    if "```" in raw:
        start = raw.find("[")
        end = raw.rfind("]") + 1
        raw = raw[start:end] if start != -1 and end > 0 else raw

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            result = []
            for item in parsed:
                if isinstance(item, dict) and "fact" in item:
                    result.append({
                        "fact": str(item["fact"]).strip(),
                        "scope": str(item.get("scope", "general")),
                    })
                elif isinstance(item, str):
                    # Fallback: plain string without scope
                    result.append({"fact": item.strip(), "scope": "general"})
            return result
    except Exception as e:
        log.debug(f"memory_extractor: JSON parse failed: {e}, raw={raw[:200]}")

    return []


async def _resolve_conflict(
    request, model_id: str, existing: MemoryModel, new_fact: str, user
) -> tuple[str, Optional[str]]:
    """
    Ask the LLM to decide: UPDATE / EXTEND / IGNORE.
    Returns (action, merged_text_or_None).
    """
    raw = await _call_llm(
        request, model_id,
        CONFLICT_RESOLUTION_PROMPT.format(
            existing=existing.content,
            new_fact=new_fact,
        ),
        user,
        max_tokens=256,
    )
    if not raw:
        return "IGNORE", None

    try:
        if "```" in raw:
            raw = raw[raw.find("{"):raw.rfind("}") + 1]
        data = json.loads(raw)
        action = str(data.get("action", "IGNORE")).upper()
        merged = data.get("merged") or None
        if action not in ("UPDATE", "EXTEND", "IGNORE"):
            action = "IGNORE"
        return action, merged
    except Exception as e:
        log.debug(f"memory_extractor: conflict resolution parse failed: {e}")
        return "IGNORE", None


async def _find_similar_memory(
    request, user_id: str, fact: str, user
) -> Optional[MemoryModel]:
    """
    Vector search for a similar existing memory.
    Returns the memory if found within threshold, else None.
    """
    try:
        existing = Memories.get_memories_by_user_id(user_id)
        if not existing:
            return None

        vector = await request.app.state.EMBEDDING_FUNCTION(fact, user=user)
        results = VECTOR_DB_CLIENT.search(
            collection_name=f"user-memory-{user_id}",
            vectors=[vector],
            limit=1,
        )
        if (
            results
            and hasattr(results, "distances")
            and results.distances
            and results.distances[0]
            and results.distances[0][0] < SIMILARITY_THRESHOLD
            and results.ids
            and results.ids[0]
        ):
            mem_id = results.ids[0][0]
            return Memories.get_memory_by_id(mem_id)
    except Exception as e:
        log.debug(f"memory_extractor: similarity search failed: {e}")
    return None


async def _upsert_memory(
    request,
    user_id: str,
    memory_id: Optional[str],
    content: str,
    scope: str,
    source_date: int,
    user,
):
    """Insert new or update existing memory, then re-embed."""
    try:
        if memory_id:
            memory = Memories.update_memory_by_id_and_user_id(
                memory_id, user_id, content, scope=scope, source_date=source_date
            )
        else:
            memory = Memories.insert_new_memory(
                user_id, content, scope=scope, source_date=source_date
            )

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
                        "scope": memory.scope,
                        "source_date": memory.source_date,
                    },
                }
            ],
        )
        action = "updated" if memory_id else "added"
        log.info(
            f"memory_extractor: {action} [{scope}] for user {user_id}: {content[:80]}"
        )
    except Exception as e:
        log.debug(f"memory_extractor: upsert failed: {e}")


# -----------------------------------------------------------------
# Public entry points
# -----------------------------------------------------------------

async def _process_facts(
    request, model_id: str, facts: list[dict], user, source_date: int
):
    """
    For each extracted fact:
    1. Find similar existing memory via vector search
    2. If found → LLM arbitration (UPDATE / EXTEND / IGNORE)
    3. If not found → insert new
    """
    for item in facts:
        fact = item["fact"]
        scope = item.get("scope", "general")

        similar = await _find_similar_memory(request, user.id, fact, user)

        if similar:
            # Temporal relevance: only consider updating if new fact is from a later date
            existing_source = similar.source_date or similar.created_at
            if source_date < existing_source:
                # New fact is OLDER than what we already know — skip
                log.debug(
                    f"memory_extractor: skipping older fact for memory {similar.id}"
                )
                continue

            action, merged = await _resolve_conflict(
                request, model_id, similar, fact, user
            )

            if action == "IGNORE":
                log.debug(f"memory_extractor: IGNORE (duplicate) — {fact[:60]}")
                continue
            elif action in ("UPDATE", "EXTEND"):
                final_content = merged if merged else fact
                await _upsert_memory(
                    request, user.id, similar.id,
                    final_content, scope, source_date, user
                )
        else:
            await _upsert_memory(
                request, user.id, None,
                fact, scope, source_date, user
            )


async def extract_and_store_memories(request, messages: list, model_id: str, user):
    """
    Incremental extraction: process only the current exchange (last user + assistant).
    Called as fire-and-forget after each response.
    """
    if not getattr(request.app.state.config, "ENABLE_MEMORIES", False):
        return

    try:
        conversation = _extract_current_exchange(messages)
        if not conversation:
            return

        facts = await _extract_facts(request, model_id, conversation, user)
        if not facts:
            log.debug("memory_extractor: no facts in current exchange")
            return

        log.info(
            f"memory_extractor: extracted {len(facts)} fact(s) for user {user.id}"
        )
        source_date = int(time.time())
        await _process_facts(request, model_id, facts, user, source_date)

    except Exception as e:
        log.debug(f"memory_extractor: unexpected error: {e}")


async def extract_and_store_memories_full(
    request, messages: list, model_id: str, user
):
    """
    Full-conversation extraction: process all messages.
    Use on first activation or manual trigger via API.
    """
    if not getattr(request.app.state.config, "ENABLE_MEMORIES", False):
        return []

    conversation = _extract_full_conversation(messages)
    if not conversation:
        return []

    facts = await _extract_facts(request, model_id, conversation, user)
    if not facts:
        return []

    source_date = int(time.time())
    await _process_facts(request, model_id, facts, user, source_date)
    return facts
