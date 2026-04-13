"""
Long Context management — warm layer engine.

Three-layer architecture:
  HOT   — last CHAT_HOT_WINDOW_SIZE messages, passed verbatim to the model.
  WARM  — older messages compressed into an incremental summary stored in chat.meta.
  COLD  — all messages in ChromaDB, searchable via search_chat_history tool.

Summarization is triggered at CHAT_WARM_THRESHOLD (40%) of the model's context window.
Updates are incremental: when new messages cross into the warm zone, the existing summary
is updated by passing it + the new messages to the LLM — the full history is never
reprocessed from scratch after the first summary exists.

If the summary itself grows beyond CHAT_WARM_SUMMARY_MAX_TOKENS, it is recompressed
(recursive summarization / second level of the hierarchy).
"""

import logging
import time
from typing import Optional, List, Tuple

from open_webui.utils.token_counter import (
    count_messages_tokens,
    count_tokens,
    should_summarize,
)
from open_webui.utils.chat import generate_chat_completion
from open_webui.retrieval.vector.factory import VECTOR_DB_CLIENT

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHAT_HOT_WINDOW_SIZE = 30          # Last N messages always sent verbatim
CHAT_WARM_THRESHOLD  = 0.4         # Trigger summarization at 40% of context limit
CHAT_MIN_SUMMARIZE_TOKENS = 8_000  # Don't compress tiny chats
CHAT_WARM_SUMMARY_MAX_TOKENS = 1_500  # Recompress the summary itself when it exceeds this

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_FULL_COMPRESS_PROMPT = """\
Compress this conversation to roughly 25–30% of its original length.

MUST PRESERVE:
- All specific facts: names, dates, numbers, URLs, file paths, code snippets
- Sequence of events and cause-and-effect relationships
- Decisions made and agreements reached
- Open questions that still need an answer
- Emotional context when significant (user said they were frustrated, excited, etc.)

OK TO REMOVE:
- Greetings and pleasantries
- Exact repetitions of the same point
- Filler phrases and acknowledgements ("I see", "Sure", "Got it")
- Off-topic tangents that led nowhere
- Detailed explanations of obvious things

OUTPUT: A concise paragraph or bullet list in the same language as the conversation.
No [summary] markers, no meta-commentary, just the compressed content.

CONVERSATION:
{conversation}

COMPRESSED SUMMARY:"""


_INCREMENTAL_UPDATE_PROMPT = """\
You have an existing summary of an ongoing conversation and a batch of new messages \
that just moved into the archive. Update the summary to include the new information.

Rules:
- Integrate new facts, decisions, and context into the existing summary
- Keep the same compression level as the existing summary
- If the new messages repeat or confirm something already in the summary, merge rather than duplicate
- ALWAYS preserve all numbers, dates, code, file paths, decisions, agreements
- Output only the updated summary — no commentary, no markers

EXISTING SUMMARY:
{existing_summary}

NEW MESSAGES TO ADD:
{new_messages}

UPDATED SUMMARY:"""


_RECOMPRESS_PROMPT = """\
The following summary has grown too long. Compress it to roughly 40% of its current length.

Rules:
- MUST keep all specific facts: names, dates, numbers, code, decisions, agreements
- Merge related points into single sentences
- Drop vague observations and generic conclusions
- Output only the recompressed text — no commentary

OVERSIZED SUMMARY:
{summary}

RECOMPRESSED:"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _format_messages(messages: List[dict]) -> str:
    parts = []
    for msg in messages:
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get('type') == 'text':
                        text_parts.append(item.get('text', ''))
                    elif item.get('type') == 'image_url':
                        text_parts.append('[IMAGE]')
            content = ' '.join(text_parts).strip()
        elif not isinstance(content, str):
            content = str(content)
        if content:
            label = role.upper() if role != 'tool' else f"TOOL({msg.get('tool_call_id', '')})"
            parts.append(f"{label}: {content.strip()}")
    return '\n'.join(parts)


async def _call_llm(
    user,
    model_id: str,
    request,
    prompt_template: str,
    max_tokens: int,
    **fmt_kwargs,
) -> Optional[str]:
    if not request or not request.app.state:
        return None
    prompt = prompt_template.format(**fmt_kwargs)
    try:
        result = await generate_chat_completion(
            request,
            {
                'model': model_id,
                'messages': [{'role': 'user', 'content': prompt}],
                'stream': False,
                'max_completion_tokens': max_tokens,
                'temperature': 0.1,
                'metadata': {'task': 'chat_summarization'},
            },
            user,
        )
        if result and isinstance(result, dict):
            text_out = (
                result.get('choices', [{}])[0]
                .get('message', {})
                .get('content', '')
                .strip()
            )
            return text_out or None
    except Exception as e:
        log.debug(f'chat_summarizer: LLM call failed: {e}')
    return None


async def _recompress_if_oversized(
    summary: str,
    model_id: str,
    user,
    request,
) -> str:
    """
    If summary exceeds CHAT_WARM_SUMMARY_MAX_TOKENS, run a second-pass compression.
    This is the recursive summarization step — prevents the warm summary from growing
    indefinitely through repeated incremental updates.
    """
    token_count = count_tokens(summary, model_id)
    if token_count <= CHAT_WARM_SUMMARY_MAX_TOKENS:
        return summary

    log.info(
        f'chat_summarizer: summary oversized ({token_count} tokens > '
        f'{CHAT_WARM_SUMMARY_MAX_TOKENS}), recompressing'
    )
    recompressed = await _call_llm(
        user, model_id, request,
        _RECOMPRESS_PROMPT,
        max_tokens=CHAT_WARM_SUMMARY_MAX_TOKENS // 2,
        summary=summary,
    )
    if recompressed:
        log.info(
            f'chat_summarizer: recompressed summary '
            f'{token_count} → {count_tokens(recompressed, model_id)} tokens'
        )
        return recompressed
    return summary  # recompression failed, keep oversized summary


# ---------------------------------------------------------------------------
# Warm summary DB persistence (delegates to Chats model)
# ---------------------------------------------------------------------------

def _get_warm_summary(chat_id: str) -> Tuple[Optional[str], int]:
    """Return (summary_text, msg_count) stored in chat.meta, or (None, 0)."""
    try:
        from open_webui.models.chats import Chats
        return Chats.get_warm_summary(chat_id)
    except Exception as e:
        log.debug(f'chat_summarizer: get_warm_summary error: {e}')
        return None, 0


def _save_warm_summary(chat_id: str, summary: str, msg_count: int) -> None:
    """Persist warm summary to chat.meta (best-effort, swallows errors)."""
    try:
        from open_webui.models.chats import Chats
        Chats.update_warm_summary(chat_id, summary, msg_count)
    except Exception as e:
        log.debug(f'chat_summarizer: save_warm_summary error: {e}')


# ---------------------------------------------------------------------------
# Public: warm layer processing
# ---------------------------------------------------------------------------

async def maybe_summarize_chat_warm_layer(
    chat_id: str,
    user,
    model_id: str,
    messages: List[dict],
    request=None,
) -> Tuple[List[dict], Optional[str]]:
    """
    Conditionally compress old messages into a warm summary.

    Returns:
        (hot_messages, warm_summary)
        - hot_messages: last CHAT_HOT_WINDOW_SIZE messages (always verbatim)
        - warm_summary: compressed text of older messages, or None

    Behaviour:
    1. Below threshold AND no existing summary → pass through unchanged.
    2. Below threshold BUT existing summary exists → keep using it
       (the chat was compressed before; don't un-compress just because context shrank).
    3. Above threshold:
       a. No existing summary → full compression of all warm messages.
       b. Existing summary → incremental update with only new warm messages.
    4. After any update: if summary exceeds CHAT_WARM_SUMMARY_MAX_TOKENS →
       recursive recompression (second-level summarization).
    """
    if not messages or len(messages) < 2:
        return messages, None

    # Always load existing summary first
    existing_summary, last_msg_count = (None, 0)
    if chat_id:
        existing_summary, last_msg_count = _get_warm_summary(chat_id)

    above_threshold = should_summarize(messages, model_id, threshold=CHAT_WARM_THRESHOLD)

    # Case 1: below threshold, no existing summary — pass through
    if not above_threshold and not existing_summary:
        return messages, None

    # Split hot / warm
    if len(messages) <= CHAT_HOT_WINDOW_SIZE:
        hot_messages = messages
        warm_messages = []
    else:
        hot_messages = messages[-CHAT_HOT_WINDOW_SIZE:]
        warm_messages = messages[:-CHAT_HOT_WINDOW_SIZE]

    # Case 2: below threshold but existing summary — keep split, return existing
    if not above_threshold and existing_summary:
        if not warm_messages:
            return messages, None
        return hot_messages, existing_summary

    # Case 3: above threshold — need to create or update summary
    if not warm_messages:
        return hot_messages, existing_summary  # nothing to compress yet

    total_tokens = count_messages_tokens(messages, model_id)
    if total_tokens < CHAT_MIN_SUMMARIZE_TOKENS:
        return messages, None

    try:
        if existing_summary and last_msg_count > 0:
            # Incremental: find messages that crossed into warm since last summarization
            # prev warm boundary: last_msg_count - CHAT_HOT_WINDOW_SIZE
            prev_warm_end = max(0, last_msg_count - CHAT_HOT_WINDOW_SIZE)
            new_warm_messages = warm_messages[prev_warm_end:]

            if not new_warm_messages:
                # Nothing new in warm zone — but check if summary needs recompression
                summary = await _recompress_if_oversized(
                    existing_summary, model_id, user, request
                )
                if summary != existing_summary and chat_id:
                    _save_warm_summary(chat_id, summary, last_msg_count)
                return hot_messages, summary

            new_text = _format_messages(new_warm_messages)
            # max_tokens: leave room for both existing summary and new content
            max_tokens = max(400, len(existing_summary) // 4 + len(new_text) // 8)

            log.info(
                f'chat {chat_id}: incremental warm update '
                f'({len(new_warm_messages)} new msgs → warm, total warm={len(warm_messages)})'
            )
            summary = await _call_llm(
                user, model_id, request,
                _INCREMENTAL_UPDATE_PROMPT,
                max_tokens=max_tokens,
                existing_summary=existing_summary,
                new_messages=new_text,
            )
        else:
            # Full compression — first time or summary was lost
            warm_text = _format_messages(warm_messages)
            max_tokens = max(400, len(warm_text) // 8)

            log.info(
                f'chat {chat_id}: full warm compression '
                f'({len(warm_messages)} msgs, {total_tokens} tokens)'
            )
            summary = await _call_llm(
                user, model_id, request,
                _FULL_COMPRESS_PROMPT,
                max_tokens=max_tokens,
                conversation=warm_text,
            )

        if not summary:
            log.warning(f'chat {chat_id}: warm layer LLM returned empty summary')
            if existing_summary:
                return hot_messages, existing_summary
            return messages, None

        # Recursive recompression guard: if summary grew too large, compress it
        summary = await _recompress_if_oversized(summary, model_id, user, request)

        # Persist
        if chat_id:
            _save_warm_summary(chat_id, summary, len(messages))

        log.info(f'chat {chat_id}: warm summary saved ({count_tokens(summary, model_id)} tokens)')
        return hot_messages, summary

    except Exception as e:
        log.error(f'chat {chat_id}: warm layer error: {e}')
        if existing_summary:
            return hot_messages, existing_summary
        return messages, None


def build_tiered_messages(
    hot_messages: List[dict],
    warm_summary: Optional[str],
    system_prompt: Optional[str] = None,
) -> List[dict]:
    """
    Assemble the final message list: [system_prompt?, warm_summary_msg?, ...hot_messages].

    The warm summary is injected as a system message with explicit framing so the model
    understands this is a compressed representation of older exchanges — not live messages.
    It sits at the front of the message list, before all user/assistant turns.
    """
    result = []
    if system_prompt:
        result.append({'role': 'system', 'content': system_prompt})
    if warm_summary:
        result.append({
            'role': 'system',
            'content': (
                'CONVERSATION HISTORY (compressed archive of earlier messages — '
                'these exchanges are no longer in context but the key information is preserved below):\n\n'
                + warm_summary
                + '\n\n'
                '--- End of archived history. Recent messages follow. ---'
            ),
        })
    result.extend(hot_messages)
    return result


# ---------------------------------------------------------------------------
# Public: cold layer — save messages to ChromaDB
# ---------------------------------------------------------------------------

async def save_chat_messages_to_vectordb(
    chat_id: str,
    user,
    messages: List[dict],
    request=None,
) -> None:
    """
    Upsert chat messages into a per-chat ChromaDB collection (cold layer).
    Called fire-and-forget after each response.
    """
    if not request or not chat_id or not messages:
        return

    try:
        msgs_to_save = [m for m in messages if m.get('role') in ('user', 'assistant')]
        if not msgs_to_save:
            return

        collection_name = f'chat-{chat_id}'
        current_time = int(time.time())
        items = []

        for idx, msg in enumerate(msgs_to_save):
            content = msg.get('content', '')
            if isinstance(content, list):
                content = ' '.join(
                    item.get('text', '')
                    for item in content
                    if isinstance(item, dict) and item.get('type') == 'text'
                ).strip()
            elif not isinstance(content, str):
                content = str(content)

            if not content:
                continue

            try:
                vector = await request.app.state.EMBEDDING_FUNCTION(content, user=user)
                items.append({
                    'id': f'{chat_id}-{idx}-{current_time}',
                    'text': content,
                    'vector': vector,
                    'metadata': {
                        'role': msg.get('role'),
                        'timestamp': current_time,
                        'message_index': idx,
                        'importance': 0.5,    # initial importance score
                        'access_count': 0,
                        'last_accessed_at': current_time,
                    },
                })
            except Exception as e:
                log.debug(f'chat_summarizer: embed error for msg {idx}: {e}')

        if items:
            try:
                VECTOR_DB_CLIENT.upsert(collection_name=collection_name, items=items)
                log.debug(f'cold layer: saved {len(items)} msgs for chat-{chat_id}')
            except Exception as e:
                log.debug(f'chat_summarizer: ChromaDB upsert error: {e}')

    except Exception as e:
        log.debug(f'save_chat_messages_to_vectordb error: {e}')
