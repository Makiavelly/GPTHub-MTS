"""
Token counting utility for managing message context limits.

Supports multiple tokenization strategies:
- tiktoken for OpenAI models (GPT-3.5, GPT-4, etc.)
- Fallback estimation for other models
- Cached counts to avoid recalculation
"""

import logging
from typing import List, Dict, Optional
import tiktoken

log = logging.getLogger(__name__)

# Model context windows (in tokens)
MODEL_CONTEXT_WINDOWS = {
    # OpenAI
    'gpt-4-turbo': 128_000,
    'gpt-4-turbo-preview': 128_000,
    'gpt-4': 8_192,
    'gpt-4-32k': 32_768,
    'gpt-3.5-turbo': 4_096,
    'gpt-3.5-turbo-16k': 16_384,

    # Anthropic Claude
    'claude-3-opus': 200_000,
    'claude-3-sonnet': 200_000,
    'claude-3-haiku': 200_000,
    'claude-2': 100_000,
    'claude-1': 100_000,

    # Meta Llama
    'llama-2-70b': 4_096,
    'llama-2-13b': 4_096,
    'llama-2-7b': 4_096,
    'llama-3-70b': 8_192,
    'llama-3-8b': 8_192,

    # Mistral
    'mistral-7b': 8_192,
    'mistral-large': 32_000,

    # Qwen / MWS models
    'qwen2.5-72b-instruct': 128_000,
    'qwen2.5-vl': 128_000,
    'qwen3-coder-480b-a35b': 128_000,
    'qwq-32b': 131_072,
    'mws-gpt-alpha': 128_000,

    # Auto-router: uses the same 128K pool as the underlying MWS models
    'auto': 128_000,

    # Open source / defaults
    'default': 4_096,
    'large': 16_384,
    'xlarge': 32_768,
}

# Tiktoken encoding cache
_ENCODING_CACHE: Dict[str, tiktoken.Encoding] = {}


def get_encoding(model_id: str) -> Optional[tiktoken.Encoding]:
    """
    Get tiktoken encoding for a model.
    Returns None if model uses fallback estimation.
    """
    if model_id in _ENCODING_CACHE:
        return _ENCODING_CACHE[model_id]

    # Try to get encoding from tiktoken
    try:
        # Map model names to tiktoken model names
        if 'gpt-4' in model_id.lower():
            encoding = tiktoken.encoding_for_model('gpt-4')
        elif 'gpt-3.5' in model_id.lower():
            encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
        elif 'claude' in model_id.lower():
            # Claude uses different tokenization, approximate as GPT-3.5
            encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
        else:
            # Default to cl100k_base (used by GPT-4)
            encoding = tiktoken.get_encoding('cl100k_base')

        _ENCODING_CACHE[model_id] = encoding
        return encoding
    except Exception as e:
        log.debug(f"Could not get tiktoken encoding for {model_id}: {e}")
        return None


def count_tokens(text: str, model_id: str) -> int:
    """
    Count tokens in a text string using the appropriate tokenizer for the model.

    Args:
        text: The text to count tokens for
        model_id: The model ID

    Returns:
        Estimated token count
    """
    if not text:
        return 0

    encoding = get_encoding(model_id)
    if encoding:
        try:
            return len(encoding.encode(text))
        except Exception as e:
            log.debug(f"Error counting tokens with tiktoken: {e}")

    # Fallback: rough estimation (1 token ≈ 4 characters)
    return len(text) // 4 + 1


def count_messages_tokens(messages: List[dict], model_id: str) -> int:
    """
    Count total tokens in a messages array.

    Includes overhead for message formatting:
    - ~4 tokens per message (role, content keys)
    - ~2 tokens per message for separators

    Args:
        messages: List of message dictionaries with 'role' and 'content'
        model_id: The model ID

    Returns:
        Estimated token count
    """
    total = 0

    for message in messages:
        # Count content tokens
        content = message.get('content', '')
        if isinstance(content, str):
            total += count_tokens(content, model_id)
        elif isinstance(content, list):
            # Multimodal content (text + images)
            for item in content:
                if isinstance(item, dict) and item.get('type') == 'text':
                    total += count_tokens(item.get('text', ''), model_id)
                elif isinstance(item, dict) and item.get('type') == 'image_url':
                    # Image tokens (very rough estimate)
                    total += 85  # Typical image token cost

        # Add message overhead
        total += 6  # ~4 for message structure + ~2 for separators

    return total


def get_model_context_limit(model_id: str) -> int:
    """
    Get the context window size for a model.

    Args:
        model_id: The model ID or name

    Returns:
        Context window size in tokens
    """
    # Exact match (case-insensitive)
    model_lower = model_id.lower()
    if model_lower in MODEL_CONTEXT_WINDOWS:
        return MODEL_CONTEXT_WINDOWS[model_lower]

    # Substring matching (case-insensitive)
    for key, value in MODEL_CONTEXT_WINDOWS.items():
        if key not in ('default', 'large', 'xlarge') and key in model_lower:
            return value

    # Fallback to default
    return MODEL_CONTEXT_WINDOWS['default']


def get_available_context_budget(
    messages: List[dict],
    model_id: str,
    reserve_for_response: int = 2000
) -> int:
    """
    Calculate how many tokens are available for the response.

    Args:
        messages: Current messages array
        model_id: The model ID
        reserve_for_response: How many tokens to reserve for completion (default 2000)

    Returns:
        Available tokens for response
    """
    context_limit = get_model_context_limit(model_id)
    used_tokens = count_messages_tokens(messages, model_id)
    available = max(0, context_limit - used_tokens - reserve_for_response)

    return available


def should_summarize(
    messages: List[dict],
    model_id: str,
    threshold: float = 0.5
) -> bool:
    """
    Check if messages should be summarized (warm layer triggered).

    Triggers when context usage exceeds threshold.

    Args:
        messages: Current messages array
        model_id: The model ID
        threshold: Trigger summarization at this % of context (default 50%)

    Returns:
        True if summarization should be triggered
    """
    context_limit = get_model_context_limit(model_id)
    used_tokens = count_messages_tokens(messages, model_id)
    ratio = used_tokens / context_limit

    return ratio >= threshold
