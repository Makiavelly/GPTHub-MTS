"""
Long-term memory system v3 — production-grade, competition-ready.

Architecture inspired by MemGPT, Mem0, ChatGPT Memory:

EXTRACTION (after each response):
  1. Extract only current exchange (last user + assistant)
  2. LLM extracts [{fact, scope}] as structured JSON
  3. Per fact: vector search → LLM conflict arbitration (UPDATE/EXTEND/IGNORE)
  4. Temporal guard: never overwrite a newer fact with an older one
  5. Recompute importance_score for touched memories
  6. Every PROFILE_REGEN_INTERVAL new/updated facts: regenerate user profile

INJECTION (at query time — called by chat_memory_handler):
  TIER 1: User profile summary (always present, ~150 words, compressed by LLM)
  TIER 2: Top-5 semantically relevant facts (vector search on current query)
  TIER 3: Top-3 most important facts not already in Tier 2

IMPORTANCE SCORING:
  score = scope_weight * recency_factor * (1 + log1p(access_count) * 0.1)
  Decays slowly for general facts, stays high for personal/work.
"""

import json
import logging
import math
import time
from typing import Optional

from open_webui.models.memories import (
    Memories,
    MemoryModel,
    MemoryProfiles,
    SCOPE_WEIGHTS,
    PROFILE_REGEN_INTERVAL,
)
from open_webui.retrieval.vector.factory import VECTOR_DB_CLIENT
from open_webui.utils.chat import generate_chat_completion
from open_webui.utils.task import get_task_model_id

log = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Tunables
# ------------------------------------------------------------------ #

# ChromaDB distance below which we check for conflict (0 = identical)
SIMILARITY_THRESHOLD = 0.45

# How many days before a "general" fact starts losing importance
DECAY_HALF_LIFE_DAYS = 30

# Profile generation prompt word limit
PROFILE_MAX_WORDS = 150

# ------------------------------------------------------------------ #
# Prompts
# ------------------------------------------------------------------ #

EXTRACTION_PROMPT = """Analyze the conversation and extract concrete, specific facts worth remembering long-term ABOUT THE USER.

Focus ONLY on:
- Identity / personal (name, age, city, family)
- Professional context (role, company, tools, projects, budget, deadlines)
- Stated preferences (communication style, response format, language, shortcuts)
- Ongoing goals or decisions

Rules:
- Each fact starts with "User" ("User is a Python developer at Sber", "User prefers bullet lists")
- Do NOT extract questions or topics — only facts ABOUT the user
- Do NOT invent facts not stated
- Scope must be exactly one of: personal | work | preference | general
- Return [] if nothing worth remembering
- Output ONLY a valid JSON array, no markdown, no commentary

[
  {{"fact": "...", "scope": "work"}},
  {{"fact": "...", "scope": "preference"}}
]

Conversation:
{conversation}

JSON:"""

CONFLICT_PROMPT = """Memory manager. Decide how to handle a new fact vs existing stored memory.

Stored: {existing}
New:    {new_fact}

Rules:
- UPDATE  → new fact REPLACES stored (changed role, budget updated, preference reversed)
- EXTEND  → new fact adds detail without contradiction (same topic, more info)
- IGNORE  → new fact is a duplicate or less specific

Respond ONLY with JSON — no markdown:
{{"action": "UPDATE"|"EXTEND"|"IGNORE", "merged": "<final text if UPDATE or EXTEND, else null>"}}"""

PROFILE_PROMPT = """Synthesize a concise user profile from these memory facts.
Write in third person, max {max_words} words, plain text, no markdown.
Include: who they are, what they do, key preferences, active projects.
Omit vague or trivial facts.

Facts:
{facts}

Profile:"""

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _current_exchange(messages: list) -> str:
    """Return the single most recent user + assistant turn."""
    user, assistant = "", ""
    for msg in reversed(messages):
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content if p.get("type") == "text")
        content = (content or "").strip()
        if not content:
            continue
        if role == "assistant" and not assistant:
            assistant = content
        elif role == "user" and not user:
            user = content
        if user and assistant:
            break
    if not user:
        return ""
    return f"User: {user}\nAssistant: {assistant}" if assistant else f"User: {user}"


def _full_conversation(messages: list) -> str:
    parts = []
    for msg in messages:
        role = msg.get("role", "")
        if role not in ("user", "assistant"):
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content if p.get("type") == "text")
        content = (content or "").strip()
        if content:
            parts.append(f"{'User' if role == 'user' else 'Assistant'}: {content}")
    return "\n".join(parts)


async def _llm(request, model_id: str, prompt: str, user, max_tokens: int = 512) -> str:
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
    try:
        res = await generate_chat_completion(
            request,
            {
                "model": task_model_id,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "max_completion_tokens": max_tokens,
                "temperature": 0.1,
                "metadata": {"task": "memory_extraction"},
            },
            user,
        )
        if res and isinstance(res, dict):
            return res.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    except Exception as e:
        log.debug(f"memory: LLM call failed: {e}")
    return ""


def _strip_fences(text: str) -> str:
    if "```" in text:
        start = text.find("[")
        end = text.rfind("]") + 1
        if start != -1 and end > 0:
            return text[start:end]
    return text


async def _extract_facts(request, model_id: str, conversation: str, user) -> list[dict]:
    if not conversation:
        return []
    raw = await _llm(request, model_id, EXTRACTION_PROMPT.format(conversation=conversation), user)
    if not raw:
        return []
    try:
        parsed = json.loads(_strip_fences(raw))
        if isinstance(parsed, list):
            result = []
            for item in parsed:
                if isinstance(item, dict) and "fact" in item:
                    scope = str(item.get("scope", "general"))
                    if scope not in SCOPE_WEIGHTS:
                        scope = "general"
                    result.append({"fact": str(item["fact"]).strip(), "scope": scope})
                elif isinstance(item, str) and item.strip():
                    result.append({"fact": item.strip(), "scope": "general"})
            return result
    except Exception as e:
        log.debug(f"memory: extraction parse error: {e} — raw: {raw[:200]}")
    return []


async def _resolve_conflict(
    request, model_id: str, existing: MemoryModel, new_fact: str, user
) -> tuple[str, Optional[str]]:
    raw = await _llm(
        request, model_id,
        CONFLICT_PROMPT.format(existing=existing.content, new_fact=new_fact),
        user, max_tokens=256,
    )
    if not raw:
        return "IGNORE", None
    try:
        raw_clean = raw
        if "```" in raw:
            raw_clean = raw[raw.find("{"):raw.rfind("}") + 1]
        data = json.loads(raw_clean)
        action = str(data.get("action", "IGNORE")).upper()
        merged = data.get("merged") or None
        if action not in ("UPDATE", "EXTEND", "IGNORE"):
            action = "IGNORE"
        return action, merged
    except Exception as e:
        log.debug(f"memory: conflict parse error: {e}")
        return "IGNORE", None


async def _find_similar(request, user_id: str, fact: str, user) -> Optional[MemoryModel]:
    try:
        if not Memories.get_memories_by_user_id(user_id):
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
            and len(results.distances[0]) > 0
            and results.distances[0][0] < SIMILARITY_THRESHOLD
            and results.ids
            and results.ids[0]
            and len(results.ids[0]) > 0
        ):
            return Memories.get_memory_by_id(results.ids[0][0])
    except Exception as e:
        log.debug(f"memory: similarity search error: {e}")
    return None


def _compute_importance(memory: MemoryModel) -> float:
    """
    importance = scope_weight × recency_factor × frequency_boost
    - scope_weight:   personal=1.0, work=0.9, preference=0.8, general=0.5
    - recency_factor: exponential decay with DECAY_HALF_LIFE_DAYS
    - frequency_boost: logarithmic, capped to avoid runaway
    """
    scope_w = SCOPE_WEIGHTS.get(memory.scope, 0.5)

    ts = memory.updated_at or memory.created_at or int(time.time())
    age_days = (time.time() - ts) / 86400
    # personal/work decay very slowly; general decays faster
    half_life = DECAY_HALF_LIFE_DAYS * (2.0 if scope_w >= 0.9 else 1.0)
    recency = math.exp(-0.693 * age_days / half_life)  # 0.693 = ln(2)

    freq_boost = 1.0 + math.log1p(memory.access_count or 0) * 0.1

    return min(1.0, scope_w * recency * freq_boost)


async def _upsert(
    request, user_id: str, memory_id: Optional[str],
    content: str, scope: str, source_date: int, user,
) -> Optional[MemoryModel]:
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
            return None

        # Re-embed
        vector = await request.app.state.EMBEDDING_FUNCTION(memory.content, user=user)
        VECTOR_DB_CLIENT.upsert(
            collection_name=f"user-memory-{user_id}",
            items=[{
                "id": memory.id,
                "text": memory.content,
                "vector": vector,
                "metadata": {
                    "created_at": memory.created_at,
                    "updated_at": memory.updated_at,
                    "scope": memory.scope,
                    "source_date": memory.source_date,
                    "importance_score": memory.importance_score,
                },
            }],
        )

        # Recompute importance
        score = _compute_importance(memory)
        Memories.update_importance_score(memory.id, score)

        action = "updated" if memory_id else "added"
        log.info(f"memory: {action} [{scope}] importance={score:.2f} — {content[:70]}")
        return memory
    except Exception as e:
        log.debug(f"memory: upsert error: {e}")
        return None


async def _maybe_regenerate_profile(request, model_id: str, user, source_date: int):
    """
    Regenerate the compressed user profile if enough new facts have accumulated.
    The profile is ALWAYS injected into the system prompt (Tier 1).
    """
    try:
        total = Memories.count_memories_by_user_id(user.id)
        profile = MemoryProfiles.get_profile(user.id)
        last_count = profile.fact_count_at_generation if profile else 0

        if profile and total - last_count < PROFILE_REGEN_INTERVAL:
            return  # Not enough new facts to justify regeneration

        all_memories = Memories.get_memories_by_user_id(user.id)
        if not all_memories:
            return

        # Sort by importance so the prompt for profile generation is focused
        sorted_mems = sorted(all_memories, key=lambda m: m.importance_score, reverse=True)
        facts_text = "\n".join(
            f"[{m.scope}] {m.content}" for m in sorted_mems[:40]
        )

        raw = await _llm(
            request, model_id,
            PROFILE_PROMPT.format(max_words=PROFILE_MAX_WORDS, facts=facts_text),
            user, max_tokens=300,
        )
        if raw:
            MemoryProfiles.upsert_profile(user.id, raw, total)
            log.info(f"memory: regenerated profile for user {user.id} ({total} facts)")
    except Exception as e:
        log.debug(f"memory: profile regeneration error: {e}")


async def _process_facts(
    request, model_id: str, facts: list[dict], user, source_date: int
) -> int:
    """Process extracted facts. Returns number of memories added/updated."""
    count = 0
    for item in facts:
        fact = item["fact"]
        scope = item.get("scope", "general")

        similar = await _find_similar(request, user.id, fact, user)

        if similar:
            # Temporal guard: don't overwrite a newer fact with an older one
            existing_date = similar.source_date or similar.created_at
            if source_date < existing_date:
                log.debug(f"memory: temporal guard — skipping older fact: {fact[:60]}")
                continue

            action, merged = await _resolve_conflict(request, model_id, similar, fact, user)
            if action == "IGNORE":
                log.debug(f"memory: IGNORE (duplicate): {fact[:60]}")
                continue
            final_content = merged if merged else fact
            result = await _upsert(request, user.id, similar.id, final_content, scope, source_date, user)
        else:
            result = await _upsert(request, user.id, None, fact, scope, source_date, user)

        if result:
            count += 1

    return count


# ------------------------------------------------------------------ #
# Public: retrieval helpers (called from chat_memory_handler)
# ------------------------------------------------------------------ #

async def get_tiered_context(request, query: str, user) -> str:
    """
    Build the memory context string for injection into the system prompt.

    TIER 1: User profile (always present)
    TIER 2: Top-5 semantically relevant facts
    TIER 3: Top-3 most important facts not already in Tier 2
    """
    sections = []

    # --- Tier 1: Profile ---
    profile = MemoryProfiles.get_profile(user.id)
    if profile and profile.content.strip():
        sections.append(f"[User Profile]\n{profile.content.strip()}")

    # --- Tier 2: Relevant facts ---
    relevant_ids: set[str] = set()
    try:
        all_memories = Memories.get_memories_by_user_id(user.id)
        if all_memories:
            vector = await request.app.state.EMBEDDING_FUNCTION(query, user=user)
            results = VECTOR_DB_CLIENT.search(
                collection_name=f"user-memory-{user.id}",
                vectors=[vector],
                limit=5,
            )
            tier2_lines = []
            if results and hasattr(results, "ids") and results.ids:
                hit_ids = results.ids[0] if results.ids else []
                hit_metas = results.metadatas[0] if hasattr(results, "metadatas") and results.metadatas else []
                hit_docs = results.documents[0] if hasattr(results, "documents") and results.documents else []

                # Record access for scoring
                Memories.record_access(hit_ids)

                for i, mem_id in enumerate(hit_ids):
                    relevant_ids.add(mem_id)
                    doc = hit_docs[i] if i < len(hit_docs) else ""
                    meta = hit_metas[i] if i < len(hit_metas) else {}
                    src_date = meta.get("source_date") or meta.get("created_at")
                    date_str = time.strftime("%Y-%m-%d", time.localtime(src_date)) if src_date else "?"
                    scope = meta.get("scope", "")
                    scope_tag = f"[{scope}] " if scope else ""
                    tier2_lines.append(f"  {i+1}. {scope_tag}[{date_str}] {doc}")

            if tier2_lines:
                sections.append("[Relevant Context]\n" + "\n".join(tier2_lines))
    except Exception as e:
        log.debug(f"memory: tier2 retrieval error: {e}")

    # --- Tier 3: Top important facts not in Tier 2 ---
    try:
        top = Memories.get_top_memories_by_importance(user.id, limit=8)
        tier3_lines = []
        for mem in top:
            if mem.id not in relevant_ids:
                src_date = mem.source_date or mem.created_at
                date_str = time.strftime("%Y-%m-%d", time.localtime(src_date)) if src_date else "?"
                tier3_lines.append(f"  • [{mem.scope}] [{date_str}] {mem.content}")
            if len(tier3_lines) >= 3:
                break
        if tier3_lines:
            sections.append("[Key Facts]\n" + "\n".join(tier3_lines))
    except Exception as e:
        log.debug(f"memory: tier3 error: {e}")

    return "\n\n".join(sections) if sections else ""


# ------------------------------------------------------------------ #
# Public: extraction entry points
# ------------------------------------------------------------------ #

async def extract_and_store_memories(request, messages: list, model_id: str, user):
    """
    Incremental extraction — current exchange only.
    Called fire-and-forget after each response via asyncio.create_task.
    """
    if not getattr(request.app.state.config, "ENABLE_MEMORIES", False):
        return
    try:
        conversation = _current_exchange(messages)
        if not conversation:
            return

        facts = await _extract_facts(request, model_id, conversation, user)
        if not facts:
            log.debug("memory: no facts in current exchange")
            return

        log.info(f"memory: extracted {len(facts)} fact(s) for user {user.id}")
        source_date = int(time.time())
        added = await _process_facts(request, model_id, facts, user, source_date)

        if added > 0:
            await _maybe_regenerate_profile(request, model_id, user, source_date)
    except Exception as e:
        log.debug(f"memory: extraction error: {e}")


async def extract_and_store_memories_full(request, messages: list, model_id: str, user) -> list[dict]:
    """
    Full-conversation scan — use on first activation or manual trigger.
    Returns list of extracted facts.
    """
    if not getattr(request.app.state.config, "ENABLE_MEMORIES", False):
        return []
    try:
        conversation = _full_conversation(messages)
        if not conversation:
            return []

        facts = await _extract_facts(request, model_id, conversation, user)
        if not facts:
            return []

        source_date = int(time.time())
        await _process_facts(request, model_id, facts, user, source_date)
        await _maybe_regenerate_profile(request, model_id, user, source_date)
        return facts
    except Exception as e:
        log.debug(f"memory: full extraction error: {e}")
        return []
