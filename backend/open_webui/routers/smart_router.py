import asyncio
import base64
import io as _io
import json
import logging
import re

import aiohttp
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

# Model used for decomposition — fastest available
ROUTER_MODEL = 'mws-gpt-alpha'

# MWS API base URL — used for all smart router calls directly, bypassing OpenWebUI proxy
MWS_BASE_URL = 'https://api.gpt.mws.ru/v1'

# Whisper model for audio transcription
WHISPER_MODEL = 'whisper-turbo-local'

# Maps task type → actual model ID on MWS (tools use None)
MODEL_REGISTRY = {
    'text': 'qwen2.5-72b-instruct',
    'reasoning': 'QwQ-32B',
    'coding': 'qwen3-coder-480b-a35b',
    'vlm': 'qwen2.5-vl',
    'image_generation': 'qwen-image-lightning',
    'web_search': None,  # tool, no LLM model
    'web_parse': None,  # tool, no LLM model
}

URL_RE = re.compile(r'https?://[^\s\])"\']+')

DECOMPOSE_SYSTEM_PROMPT = """\
You are a task router. Your ONLY output must be a single JSON object. No explanations, no comments, no markdown, no text before or after the JSON.

Task types:
- "text" — writing text, answering questions, creating descriptions, posts, articles
- "reasoning" — math, logic, deep analysis
- "coding" — writing or fixing code
- "vlm" — analyzing an attached image (ONLY if [Note: the user has attached an image] is present)
- "image_generation" — creating/drawing/generating a new image, logo, illustration, picture
- "web_search" — searching the internet for information (when user asks to find/search online)
- "web_parse" — reading a specific URL from the message to get its content

IMPORTANT rules:
- If user asks to generate/draw/create an image/logo → "image_generation"
- If message contains a URL and user wants to discuss/summarize/analyze it → "web_parse", put the URL in prompt
- If user asks to search the web/internet → "web_search"
- Add "vlm" ONLY when [Note: the user has attached an image] appears in the message
- Split compound requests into multiple tasks
- For image_generation: write a detailed visual description as the prompt
- ALL tasks must be inside a single JSON object with a "tasks" array

STRICT output format — one JSON object, nothing else:
{"tasks":[{"id":0,"type":"TYPE","prompt":"PROMPT","depends_on":[]}]}

Examples:
User: "generate a logo and write an ad post for a bakery"
{"tasks":[{"id":0,"type":"image_generation","prompt":"Logo for a bakery, minimalist style, warm colors","depends_on":[]},{"id":1,"type":"text","prompt":"Write an ad post for a bakery","depends_on":[]}]}

User: "найди в интернете последние новости про AI"
{"tasks":[{"id":0,"type":"web_search","prompt":"последние новости про AI 2024","depends_on":[]},{"id":1,"type":"text","prompt":"Summarize the search results about AI news","depends_on":[0]}]}

User: "что написано на этом сайте https://example.com"
{"tasks":[{"id":0,"type":"web_parse","prompt":"https://example.com","depends_on":[]},{"id":1,"type":"text","prompt":"Summarize the content of the page","depends_on":[0]}]}"""


_PLAN_SEPARATOR = '\n\n---\n\n'
_RAG_CONTEXT_RE = re.compile(r'<context>.*?</context>', re.DOTALL)
# Matches the '### Task:\n...\n\n' preamble injected by OpenWebUI's RAG template
_RAG_PREAMBLE_RE = re.compile(r'^###\s*Task:.*?(?=\n\n|\Z)', re.DOTALL)


def _strip_plan_header(text: str) -> str:
    """Remove the '**Задачи:**...---' plan block from assistant messages."""
    if _PLAN_SEPARATOR in text:
        return text.split(_PLAN_SEPARATOR, 1)[1].strip()
    return text


def _decompose_text(user_text: str) -> str:
    """Extract clean user intent for the decomposer.

    Strips OpenWebUI RAG injections:
    - '### Task:...\\n\\n' preamble (from DEFAULT_RAG_TEMPLATE)
    - '<context>...</context>' blocks with retrieved documents
    - '<attached_files>...</attached_files>' file reference tags

    The execution models already receive the full context via original_messages.
    """
    text = _RAG_CONTEXT_RE.sub('', user_text)
    text = re.sub(r'<attached_files>.*?</attached_files>', '', text, flags=re.DOTALL)
    text = re.sub(r'^###\s*Task:.*?(?:\n\n|\Z)', '', text, flags=re.DOTALL)
    return text.strip()


def _extract_text(message: dict) -> str:
    content = message.get('content', '')
    if isinstance(content, list):
        return ' '.join(p.get('text', '') for p in content if p.get('type') == 'text')
    return str(content)


def _has_image(messages: list) -> bool:
    """Check only the LAST user message for images — not the full history."""
    last_user = next((m for m in reversed(messages) if m.get('role') == 'user'), None)
    if not last_user:
        return False
    content = last_user.get('content', [])
    return isinstance(content, list) and any(p.get('type') == 'image_url' for p in content)


def _has_audio(messages: list) -> bool:
    """Check only the LAST user message for audio attachments."""
    last_user = next((m for m in reversed(messages) if m.get('role') == 'user'), None)
    if not last_user:
        return False
    content = last_user.get('content', [])
    if not isinstance(content, list):
        return False
    for p in content:
        ptype = p.get('type', '')
        if ptype in ('audio', 'input_audio', 'audio_url'):
            return True
        if ptype == 'file' and p.get('file', {}).get('mime_type', '').startswith('audio/'):
            return True
    return False


async def _transcribe_audio(messages: list, api_key: str) -> str:
    """Extract audio from the last user message and transcribe via Whisper.

    Handles multiple content formats:
    - input_audio  (OpenAI Realtime format: {type, input_audio: {data, format}})
    - audio_url    ({type, audio_url: {url}}) — data URL or http URL
    - file         ({type, file, file: {url, mime_type}}) — OpenWebUI file upload
    """
    last_user = next((m for m in reversed(messages) if m.get('role') == 'user'), None)
    if not last_user:
        return ''
    content = last_user.get('content', [])
    if not isinstance(content, list):
        return ''

    audio_parts = [
        p
        for p in content
        if p.get('type') in ('audio', 'input_audio', 'audio_url')
        or (p.get('type') == 'file' and p.get('file', {}).get('mime_type', '').startswith('audio/'))
    ]

    transcriptions: list[str] = []

    for part in audio_parts:
        audio_data: bytes | None = None
        filename = 'audio.mp3'
        ptype = part.get('type', '')

        try:
            if ptype == 'input_audio':
                inp = part.get('input_audio', {})
                fmt = inp.get('format', 'mp3')
                filename = f'audio.{fmt}'
                audio_data = base64.b64decode(inp.get('data', ''))

            elif ptype in ('audio', 'audio_url'):
                url = part.get('audio_url', {}).get('url') or part.get('url') or ''
                if url.startswith('data:'):
                    header, b64data = url.split(',', 1)
                    mime = header.split(';')[0].split(':')[1]
                    ext = mime.split('/')[1] if '/' in mime else 'mp3'
                    filename = f'audio.{ext}'
                    audio_data = base64.b64decode(b64data)
                elif url:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                            audio_data = await resp.read()
                            ct = resp.headers.get('Content-Type', '')
                            if '/' in ct:
                                ext = ct.split('/')[1].split(';')[0].strip()
                                filename = f'audio.{ext}'

            elif ptype == 'file':
                file_info = part.get('file', {})
                url = file_info.get('url', '')
                mime = file_info.get('mime_type', 'audio/mp3')
                ext = mime.split('/')[1] if '/' in mime else 'mp3'
                filename = f'audio.{ext}'
                if url:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                            audio_data = await resp.read()

        except Exception as e:
            log.warning(f'[smart_router] could not extract audio data ({ptype}): {e}')
            continue

        if not audio_data:
            continue

        # Send to Whisper
        try:
            ext = filename.rsplit('.', 1)[-1]
            form = aiohttp.FormData()
            form.add_field('model', WHISPER_MODEL)
            form.add_field(
                'file',
                _io.BytesIO(audio_data),
                filename=filename,
                content_type=f'audio/{ext}',
            )
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f'{MWS_BASE_URL}/audio/transcriptions',
                    headers={'Authorization': f'Bearer {api_key}'},
                    data=form,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    result = await resp.json()
            text = result.get('text', '').strip()
            if text:
                log.info(f'[smart_router] whisper transcription: {text!r}')
                transcriptions.append(text)
            else:
                log.warning(f'[smart_router] whisper returned empty text, full response: {result}')
        except Exception as e:
            log.error(f'[smart_router] whisper transcription failed for {filename}: {e}')

    return '\n\n'.join(transcriptions)


async def _decompose(user_text: str, has_image: bool, api_key: str) -> list[dict]:
    note = '[Note: the user has attached an image to this message]\n\n' if has_image else ''
    messages = [
        {'role': 'system', 'content': DECOMPOSE_SYSTEM_PROMPT},
        {'role': 'user', 'content': f'{note}{user_text}'},
    ]

    log.info(f'[smart_router] _decompose → calling {ROUTER_MODEL}')
    log.info(f'[smart_router] user_text = {user_text!r}')

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f'{MWS_BASE_URL}/chat/completions',
            headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
            json={
                'model': ROUTER_MODEL,
                'messages': messages,
                'temperature': 0,
                'max_tokens': 1024,
                'response_format': {'type': 'json_object'},
            },
        ) as resp:
            data = await resp.json()

    if 'choices' not in data:
        log.error(f"[smart_router] decomposer returned no 'choices': {data}")
        raise KeyError('choices')

    raw = data['choices'][0]['message']['content']
    log.info(f'[smart_router] decomposer raw response:\n{raw}')

    cleaned = re.sub(r'```(?:json)?\s*|\s*```', '', raw).strip()
    # Some models wrap output in <think>...</think> — strip it
    cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL).strip()
    # Parse ALL JSON objects in the response and merge their tasks.
    # The model sometimes outputs multiple {"tasks":[...]} objects instead of one —
    # we collect every task from every object so nothing is lost.
    decoder = json.JSONDecoder()
    tasks: list[dict] = []
    pos = 0
    while pos < len(cleaned):
        start = cleaned.find('{', pos)
        if start == -1:
            break
        try:
            obj, end = decoder.raw_decode(cleaned, start)
            if isinstance(obj, dict) and 'tasks' in obj:
                tasks.extend(obj['tasks'])
            pos = end
        except json.JSONDecodeError:
            pos = start + 1  # skip this '{' and keep scanning
    if not tasks:
        log.error(f'[smart_router] no valid tasks found in decomposer response: {raw!r}')
        raise ValueError('no valid tasks in decomposer response')

    log.info(f'[smart_router] parsed tasks: {json.dumps(tasks, ensure_ascii=False, indent=2)}')
    return tasks


async def _run_image_generation(prompt: str, api_key: str) -> str:
    log.info(f'[smart_router] image_generation → model=qwen-image-lightning')
    log.info(f'[smart_router] image prompt: {prompt!r}')
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f'{MWS_BASE_URL}/images/generations',
            headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
            json={'model': 'qwen-image-lightning', 'prompt': prompt, 'n': 1},
        ) as resp:
            data = await resp.json()
    log.info(f'[smart_router] image_generation response keys: {list(data.keys())}')

    item = data['data'][0]
    url = item.get('url')
    if url:
        return f'![Generated image]({url})'
    b64 = item.get('b64_json', '')
    return f'![Generated image](data:image/png;base64,{b64})'


async def _run_chat(messages: list, model: str, api_key: str) -> str:
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f'{MWS_BASE_URL}/chat/completions',
            headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
            json={'model': model, 'messages': messages, 'temperature': 0.7},
        ) as resp:
            data = await resp.json()
    return data['choices'][0]['message']['content']


_SEARCH_REWRITE_SYSTEM_PROMPT = """\
You are a search query optimizer. Your only job is to output a concise, effective search engine query.

Rules:
- Use the conversation history ONLY if it helps clarify what the user means by vague references like "this", "it", "that topic", "about it", etc.
- If the search intent is already clear from the query alone — ignore the history completely.
- Output ONLY the search query string. No explanation, no punctuation at the end, no quotes."""


async def _rewrite_search_query(raw_query: str, original_messages: list[dict], api_key: str) -> str:
    """Rewrite a vague search query into a concrete one using full conversation context.

    Uses qwen2.5-72b-instruct with the entire chat history. The original system prompt
    is replaced with the search-rewrite instruction so the model acts as a query optimizer,
    not as a general assistant.
    """
    model = MODEL_REGISTRY['text']

    # Strip the original system message and substitute our rewrite instruction.
    # This prevents role confusion (model seeing two conflicting system prompts).
    non_sys = [m for m in original_messages if m.get('role') != 'system']
    messages = (
        [{'role': 'system', 'content': _SEARCH_REWRITE_SYSTEM_PROMPT}]
        + non_sys
        + [{'role': 'user', 'content': f'Search intent: {raw_query}'}]
    )

    log.info(f'[smart_router] rewriting search query via {model}: {raw_query!r}')
    try:
        rewritten = await _run_chat(messages, model, api_key)
        rewritten = rewritten.strip().strip('"\'')
        log.info(f'[smart_router] rewritten search query: {rewritten!r}')
        return rewritten or raw_query
    except Exception as exc:
        log.warning(f'[smart_router] query rewrite failed: {exc!r}, using original')
        return raw_query


def _web_search_sync(query: str, max_results: int = 5) -> str:
    from ddgs import DDGS

    results = list(DDGS().text(query, max_results=max_results))
    if not results:
        return 'Поиск не дал результатов.'
    lines = []
    for r in results:
        lines.append(f'**{r.get("title", "")}**\n{r.get("body", "")}\n{r.get("href", "")}')
    return '\n\n'.join(lines)


async def _run_web_search(query: str) -> str:
    log.info(f'[smart_router] web_search query: {query!r}')
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _web_search_sync, query)


async def _run_web_parse(url: str) -> str:
    log.info(f'[smart_router] web_parse url: {url!r}')
    # Extract first URL from prompt if needed
    found = URL_RE.search(url)
    target = found.group(0) if found else url
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                target,
                timeout=aiohttp.ClientTimeout(total=15),
                headers={'User-Agent': 'Mozilla/5.0'},
            ) as resp:
                html = await resp.text(errors='replace')
        soup = BeautifulSoup(html, 'html.parser')
        for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            tag.decompose()
        text = soup.get_text(separator='\n', strip=True)
        # Trim to avoid huge context
        text = '\n'.join(line for line in text.splitlines() if line.strip())[:5000]
        log.info(f'[smart_router] web_parse fetched {len(text)} chars from {target}')
        return f'Содержимое страницы {target}:\n\n{text}'
    except Exception as e:
        log.error(f'[smart_router] web_parse failed: {e}')
        return f'[Не удалось загрузить страницу {target}: {e}]'


def _get_user_memories(user_id: str) -> str:
    """Return user memories as a formatted string, or empty string."""
    if not user_id:
        return ''
    try:
        from open_webui.models.memories import Memories

        memories = Memories.get_memories_by_user_id(user_id)
        if not memories:
            return ''
        return '\n'.join(f'- {m.content}' for m in memories)
    except Exception as e:
        log.warning(f'[smart_router] could not load memories: {e}')
        return ''


async def _execute_task(
    task: dict,
    context: dict,
    original_messages: list,
    api_key: str,
) -> dict:
    task_type = task['type']
    model = MODEL_REGISTRY.get(task_type, MODEL_REGISTRY['text'])
    log.info(f'[smart_router] executing task id={task["id"]} type={task_type} → model={model}')

    # Build prompt, appending output of dependency tasks as context
    prompt = task['prompt']
    for dep_id in task.get('depends_on', []):
        if dep_id in context:
            prompt += f'\n\nContext from previous step:\n{context[dep_id]}'

    log.info(f'[smart_router] task id={task["id"]} final prompt: {prompt!r}')

    if task_type == 'web_search':
        search_query = await _rewrite_search_query(prompt, original_messages, api_key)
        content = await _run_web_search(search_query)
        return {'id': task['id'], 'type': 'text', 'task_type': task_type, 'model': 'DuckDuckGo', 'content': content}

    if task_type == 'web_parse':
        content = await _run_web_parse(prompt)
        return {'id': task['id'], 'type': 'text', 'task_type': task_type, 'model': 'web', 'content': content}

    if task_type == 'image_generation':
        content = await _run_image_generation(prompt, api_key)
        return {'id': task['id'], 'type': 'image', 'task_type': task_type, 'model': model, 'content': content}

    # For all chat-based tasks (text / reasoning / coding / vlm)
    # Use conversation history up to (not including) the last user message
    history = [m for m in original_messages if m.get('role') in ('system', 'assistant')]
    user_msg: dict = {'role': 'user', 'content': prompt}

    # For vlm: carry the image parts from the original user message
    if task_type == 'vlm':
        last_user = next((m for m in reversed(original_messages) if m.get('role') == 'user'), None)
        if last_user:
            orig_content = last_user.get('content', [])
            if isinstance(orig_content, list):
                image_parts = [p for p in orig_content if p.get('type') == 'image_url']
                if image_parts:
                    user_msg['content'] = [{'type': 'text', 'text': prompt}] + image_parts

    messages = history + [user_msg]
    content = await _run_chat(messages, model, api_key)
    return {'id': task['id'], 'type': 'text', 'task_type': task_type, 'model': model, 'content': content}


TASK_LABELS = {
    'text': 'текст',
    'reasoning': 'рассуждение',
    'coding': 'код',
    'vlm': 'анализ изображения',
    'image_generation': 'генерация изображения',
    'web_search': 'поиск в интернете',
    'web_parse': 'чтение страницы',
}

# Human-readable tool/model name shown in the task plan
TASK_DISPLAY = {
    'text': lambda: MODEL_REGISTRY['text'],
    'reasoning': lambda: MODEL_REGISTRY['reasoning'],
    'coding': lambda: MODEL_REGISTRY['coding'],
    'vlm': lambda: MODEL_REGISTRY['vlm'],
    'image_generation': lambda: MODEL_REGISTRY['image_generation'],
    'web_search': lambda: 'DuckDuckGo',
    'web_parse': lambda: 'веб-парсер',
}

# Task types whose output is intermediate (fed into next tasks) — not shown to user directly
INTERMEDIATE_TYPES = {'web_search', 'web_parse', 'vlm'}


def _plan_lines(tasks: list, has_audio: bool = False) -> list[str]:
    """Return plan as a list of lines to stream one by one."""
    lines = ['**Задачи:**\n']
    if has_audio:
        lines.append(f'  - `{WHISPER_MODEL}` — транскрипция аудио\n')
    # Build set of task IDs that are dependencies of others
    dep_ids = {d for t in tasks for d in t.get('depends_on', [])}
    for t in tasks:
        display = TASK_DISPLAY.get(t['type'], lambda: '?')()
        label = TASK_LABELS.get(t['type'], t['type'])
        deps = t.get('depends_on', [])
        dep_str = f' | на основе: {", ".join(TASK_LABELS.get(tasks[d]["type"], str(d)) for d in deps)}' if deps else ''
        # Mark intermediate tasks visually
        intermediate = t['id'] in dep_ids and t['type'] in INTERMEDIATE_TYPES
        prefix = '  ' if intermediate else ''
        # For web_search: show the query-rewriting step as a preceding sub-task
        if t['type'] == 'web_search':
            lines.append(f'  - `{MODEL_REGISTRY["text"]}` — формирование поискового запроса\n')
            lines.append(f'{prefix}- `{display}` — {label} | на основе: формирование запроса{dep_str}\n')
        else:
            lines.append(f'{prefix}- `{display}` — {label}{dep_str}\n')
    lines.append('\n---\n\n')
    return lines


def _sse_chunk(text: str) -> str:
    chunk = {
        'id': 'chatcmpl-auto',
        'object': 'chat.completion.chunk',
        'choices': [{'index': 0, 'delta': {'content': text}, 'finish_reason': None}],
        'model': 'auto',
    }
    return f'data: {json.dumps(chunk, ensure_ascii=False)}\n\n'


def _sse_done() -> str:
    return 'data: [DONE]\n\n'


async def stream_route(payload: dict, api_key: str, user_id: str = ''):
    """
    Async generator yielding SSE chunks.

    Flow:
      1. Inject long-term user memories into system context
      2. Run decomposer (fast LLM call)
      3. Yield task plan immediately — user sees which models will run
      4. Execute tasks respecting dependencies
      5. Yield combined results
    """
    api_key = api_key.removeprefix('Bearer ').strip()
    log.info(f'[smart_router] api_key preview: {api_key[:8]!r}... (len={len(api_key)})')

    messages = list(payload.get('messages', []))

    # Inject long-term memories into system context
    if user_id:
        memory_text = _get_user_memories(user_id)
        if memory_text:
            log.info(f'[smart_router] injecting {len(memory_text)} chars of user memories')
            memory_block = f'\n\nИзвестные факты о пользователе:\n{memory_text}'
            sys_idx = next((i for i, m in enumerate(messages) if m.get('role') == 'system'), -1)
            if sys_idx >= 0:
                messages[sys_idx] = {**messages[sys_idx], 'content': messages[sys_idx]['content'] + memory_block}
            else:
                messages.insert(0, {'role': 'system', 'content': memory_block.strip()})

    has_image = _has_image(messages)
    last_user = next((m for m in reversed(messages) if m.get('role') == 'user'), None)
    user_text = _extract_text(last_user) if last_user else ''

    # Internal OpenWebUI system tasks (title gen, follow-ups, tags) — route directly.
    # RAG-injected messages also start with '### Task:' (from DEFAULT_RAG_TEMPLATE)
    # but they contain a <context> block — those must go through decomposition.
    if user_text.startswith('### Task:') and '<context>' not in user_text:
        log.debug('[smart_router] internal system task, skipping routing')
        model = MODEL_REGISTRY['text']
        content = await _run_chat(messages, model, api_key)
        yield _sse_chunk(content)
        yield _sse_done()
        return

    # Transcribe audio if present — runs before decomposition so the transcription
    # becomes part of the text that the decomposer sees.
    audio_transcription = ''
    if _has_audio(messages):
        log.info('[smart_router] audio attachment detected, transcribing...')
        audio_transcription = await _transcribe_audio(messages, api_key)
        if audio_transcription:
            note = f'[Транскрипция аудио]: {audio_transcription}'
            user_text = f'{note}\n\n{user_text}' if user_text else note

    # Strip RAG preamble/context from user_text before decomposition.
    # The decomposer only needs the clean user intent ("о чём аудиофайл"),
    # not the injected ### Task: header or <context> blocks.
    # Execution models receive the full context via original_messages.
    clean_text = _decompose_text(user_text)
    log.info(f'[smart_router] clean decompose text: {clean_text!r}')

    # Decompose
    try:
        tasks = await _decompose(clean_text or user_text, has_image, api_key)
    except Exception as exc:
        log.error(f'[smart_router] decompose FAILED: {exc!r}, falling back to text')
        tasks = []

    if not tasks:
        tasks = [{'id': 0, 'type': 'text', 'prompt': clean_text or user_text, 'depends_on': []}]

    log.info(f'[smart_router] task plan: {[t["type"] for t in tasks]}')

    # Stream plan line by line so it appears smoothly before tasks run
    for line in _plan_lines(tasks, has_audio=bool(audio_transcription)):
        yield _sse_chunk(line)
        await asyncio.sleep(0.06)

    # Execute tasks respecting dependency order
    context: dict[int, str] = {}
    results: list[dict] = []
    remaining = list(tasks)

    while remaining:
        ready = [t for t in remaining if all(d in context for d in t.get('depends_on', []))]
        if not ready:
            log.error('[smart_router] dependency cycle or unsatisfied deps, aborting')
            break

        batch = await asyncio.gather(
            *[_execute_task(t, context, messages, api_key) for t in ready],
            return_exceptions=True,
        )

        for task, res in zip(ready, batch):
            if isinstance(res, Exception):
                log.error(f'[smart_router] task {task["id"]} ({task["type"]}) failed: {res}')
                fallback = f"[Ошибка задачи '{TASK_LABELS.get(task['type'], task['type'])}': {res}]"
                context[task['id']] = fallback
                results.append(
                    {'id': task['id'], 'type': 'text', 'task_type': task['type'], 'model': '?', 'content': fallback}
                )
            else:
                context[res['id']] = res['content']
                results.append(res)

        remaining = [t for t in remaining if t not in ready]

    # Show leaf results (not consumed by other tasks) + always show image_generation
    # results even if another task depends on them — the user always expects to see
    # the generated image regardless of whether a follow-up text task references it.
    dep_ids = {d for t in tasks for d in t.get('depends_on', [])}
    visible_results = sorted(
        [r for r in results if r['id'] not in dep_ids or r.get('task_type') == 'image_generation'],
        key=lambda r: r['id'],
    )
    body = '\n\n---\n\n'.join(r['content'] for r in visible_results)
    yield _sse_chunk(body)
<<<<<<< HEAD
    yield _sse_done()
=======
    yield _sse_done()
>>>>>>> 012b61dd558d078e9eee6293195237889af68938
