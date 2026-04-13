"""
Chat summarization engine for warm layer management.

Implements incremental summarization when chat messages exceed context limits.
Keeps hot layer (recent messages) fresh and cold layer available for search.
"""

import json
import logging
import time
import asyncio
from typing import Optional, List, Tuple

from open_webui.utils.token_counter import (
    count_messages_tokens,
    get_model_context_limit,
    should_summarize,
)
from open_webui.utils.chat import generate_chat_completion
from open_webui.retrieval.vector.factory import VECTOR_DB_CLIENT

log = logging.getLogger(__name__)

# Summarization configuration
CHAT_HOT_WINDOW_SIZE = 30  # Last N messages always hot
CHAT_WARM_THRESHOLD = 0.5  # Summarize at 50% context limit
CHAT_SUMMARY_COMPRESSION = 0.3  # Compress to ~30% of original size
CHAT_MIN_SUMMARIZE_TOKENS = 10_000  # Don't summarize short chats
CHAT_FORCE_SUMMARIZE_COUNT = 200  # Force summary every N messages

COMPRESSION_PROMPT = """Compress this conversation to approximately {target_ratio}% of its original length.

MUST PRESERVE:
- All factual statements (names, dates, numbers, links, code)
- Sequence of events (chronological order)
- Decisions made and agreements
- Questions that remain unanswered
- Any code snippets or technical details
- Emotional context if it was significant (user said they were frustrated, excited, etc)

OK TO REMOVE:
- Greetings and pleasantries
- Excessive repetition
- Filler words and acknowledging messages
- Off-topic tangents
- Detailed explanations of obvious things
- Meta-comments about the conversation itself

OUTPUT FORMAT:
Return a concise paragraph or bullet list that reads naturally.
Do NOT include markers like [summary] or instruction notes.
Preserve the conversational tone but in compressed form.

CONVERSATION TO COMPRESS:
{conversation_text}

COMPRESSED VERSION:"""


async def maybe_summarize_chat_warm_layer(
    chat_id: str,
    user,
    model_id: str,
    messages: List[dict],
    request=None,
) -> Tuple[List[dict], Optional[str]]:
    """
    Conditionally summarize old messages when approaching context limit.

    Args:
        chat_id: ID of the current chat
        user: User object
        model_id: Model ID for context window lookup
        messages: List of current messages to evaluate
        request: FastAPI request (needed for LLM call)

    Returns:
        Tuple of (hot_messages, warm_summary)
        - hot_messages: Last CHAT_HOT_WINDOW_SIZE messages (unchanged)
        - warm_summary: Compressed text of older messages, or None if no summarization needed
    """
    if not messages or len(messages) < 2:
        return messages, None

    # Check if summarization is needed
    if not should_summarize(messages, model_id, threshold=CHAT_WARM_THRESHOLD):
        return messages, None

    # Don't summarize very short chats
    total_tokens = count_messages_tokens(messages, model_id)
    if total_tokens < CHAT_MIN_SUMMARIZE_TOKENS:
        return messages, None

    try:
        # Split into hot and warm layers
        hot_messages = messages[-CHAT_HOT_WINDOW_SIZE:] if len(messages) > CHAT_HOT_WINDOW_SIZE else messages
        warm_messages = messages[:-CHAT_HOT_WINDOW_SIZE] if len(messages) > CHAT_HOT_WINDOW_SIZE else []

        if not warm_messages:
            return hot_messages, None

        # Build conversation text from warm layer
        warm_text = _format_messages_for_summarization(warm_messages)

        log.info(f"chat {chat_id}: summarizing warm layer ({len(warm_messages)} messages, {total_tokens} tokens)")

        # Generate summary
        summary = await _generate_summary(
            warm_text,
            user,
            model_id,
            request,
            compression_ratio=CHAT_SUMMARY_COMPRESSION,
        )

        if not summary:
            log.warning(f"chat {chat_id}: failed to generate summary")
            return messages, None

        log.info(f"chat {chat_id}: summary generated ({len(summary)} chars)")
        return hot_messages, summary

    except Exception as e:
        log.error(f"chat {chat_id}: summarization error: {e}")
        return messages, None


async def _generate_summary(
    conversation_text: str,
    user,
    model_id: str,
    request,
    compression_ratio: float = 0.3,
) -> Optional[str]:
    """
    Generate a summary of conversation text using LLM.

    Args:
        conversation_text: Formatted conversation to summarize
        user: User object
        model_id: Model to use for summarization
        request: FastAPI request
        compression_ratio: Target compression (0.3 = 30% of original)

    Returns:
        Summary text or None if generation failed
    """
    if not request or not request.app.state:
        return None

    target_percentage = int(compression_ratio * 100)
    prompt = COMPRESSION_PROMPT.format(
        target_ratio=target_percentage,
        conversation_text=conversation_text,
    )

    try:
        result = await generate_chat_completion(
            request,
            {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "max_completion_tokens": int(len(conversation_text) // 10) + 100,
                "temperature": 0.1,
                "metadata": {"task": "chat_summarization"},
            },
            user,
        )

        if result and isinstance(result, dict):
            summary = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            return summary if summary else None

        return None
    except Exception as e:
        log.debug(f"Summary generation error: {e}")
        return None


def _format_messages_for_summarization(messages: List[dict]) -> str:
    """
    Format messages into readable text for summarization.

    Args:
        messages: List of message dicts

    Returns:
        Formatted conversation text
    """
    parts = []

    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        # Handle multimodal content
        if isinstance(content, list):
            content_text = ""
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        content_text += item.get("text", "") + " "
                    elif item.get("type") == "image_url":
                        content_text += "[IMAGE] "
            content = content_text.strip()
        elif not isinstance(content, str):
            content = str(content)

        if content:
            # Format as "Role: content"
            role_label = role.upper() if role != "tool" else f"TOOL ({msg.get('tool_call_id', 'unknown')})"
            parts.append(f"{role_label}: {content.strip()}")

    return "\n".join(parts)


def build_tiered_messages(
    hot_messages: List[dict],
    warm_summary: Optional[str],
    system_prompt: Optional[str] = None,
) -> List[dict]:
    """
    Build final message list with hot + warm layers.

    Args:
        hot_messages: Recent messages (hot layer)
        warm_summary: Compressed summary of older messages (warm layer)
        system_prompt: System prompt (if any)

    Returns:
        Final message list ready to send to model
    """
    result = []

    # Add system prompt first
    if system_prompt:
        result.append({"role": "system", "content": system_prompt})

    # Add warm summary as system message (so it's indexed but not confused with chat)
    if warm_summary:
        result.append({
            "role": "system",
            "content": f"[Earlier conversation summary]\n{warm_summary}"
        })

    # Add hot layer (recent messages)
    result.extend(hot_messages)

    return result


async def save_chat_messages_to_vectordb(
    chat_id: str,
    user,
    messages: List[dict],
    request=None,
) -> None:
    """
    Save chat messages to ChromaDB for vector search.

    Args:
        chat_id: ID of the chat
        user: User object
        messages: List of messages to save
        request: FastAPI request (needed for embedding function)
    """
    if not request or not chat_id or not messages:
        return

    try:
        # Filter to only user and assistant messages
        msgs_to_save = [m for m in messages if m.get('role') in ('user', 'assistant')]
        if not msgs_to_save:
            return

        collection_name = f'chat-{chat_id}'

        # Generate embeddings for each message
        items = []
        current_time = int(time.time())

        for msg_idx, msg in enumerate(msgs_to_save):
            content = msg.get('content', '')

            # Handle multi modal content
            if isinstance(content, list):
                content_text = ""
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        content_text += item.get('text', '') + " "
                content = content_text.strip()
            elif not isinstance(content, str):
                content = str(content)

            if not content:
                continue

            try:
                # Generate embedding
                vector = await request.app.state.EMBEDDING_FUNCTION(content, user=user)

                items.append({
                    'id': f"{chat_id}-{msg_idx}-{current_time}",
                    'text': content,
                    'vector': vector,
                    'metadata': {
                        'role': msg.get('role'),
                        'timestamp': current_time,
                        'message_index': msg_idx,
                    }
                })
            except Exception as e:
                log.debug(f"Failed to embed message {msg_idx}: {e}")
                continue

        # Upsert to ChromaDB
        if items:
            try:
                VECTOR_DB_CLIENT.upsert(
                    collection_name=collection_name,
                    items=items,
                )
                log.debug(f"Saved {len(items)} messages to chat-{chat_id} collection")
            except Exception as e:
                log.debug(f"Failed to save messages to ChromaDB: {e}")

    except Exception as e:
        log.debug(f"save_chat_messages_to_vectordb error: {e}")
