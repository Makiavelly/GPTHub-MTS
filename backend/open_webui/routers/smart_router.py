import asyncio
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
You are a task router. Analyze the user message and output a JSON task list.

Task types:
- "text" — writing text, answering questions, creating descriptions, posts, articles
- "reasoning" — math, logic, deep analysis
- "coding" — writing or fixing code
- "vlm" — analyzing an attached image
- "image_generation" — creating/drawing/generating a new image, logo, illustration, picture
- "web_search" — searching the internet for information (when user asks to find/search online)
- "web_parse" — reading a specific URL from the message to get its content

IMPORTANT rules:
- If user asks to generate/draw/create an image/logo → "image_generation"
- If message contains a URL and user wants to discuss/summarize/analyze it → "web_parse", put the URL in prompt
- If user asks to search the web/internet → "web_search"
- Split compound requests into multiple tasks
- For image_generation: write a detailed visual description as the prompt
- Output ONLY the JSON, nothing else

Output format (JSON only, no extra text, no markdown):
{"tasks":[{"id":0,"type":"TYPE","prompt":"PROMPT","depends_on":[]}]}

Examples:
User: "generate a logo and write an ad post for a bakery"
Output: {"tasks":[{"id":0,"type":"image_generation","prompt":"Logo for a bakery, minimalist style, warm colors"},{"id":1,"type":"text","prompt":"Write an ad post for a bakery","depends_on":[]}]}

User: "найди в интернете последние новости про AI"
Output: {"tasks":[{"id":0,"type":"web_search","prompt":"последние новости про AI 2024","depends_on":[]},{"id":1,"type":"text","prompt":"Summarize the search results about AI news","depends_on":[0]}]}

User: "что написано на этом сайте https://example.com"
Output: {"tasks":[{"id":0,"type":"web_parse","prompt":"https://example.com","depends_on":[]},{"id":1,"type":"text","prompt":"Summarize the content of the page","depends_on":[0]}]}"""


def _extract_text(message: dict) -> str:
    content = message.get('content', '')
    if isinstance(content, list):
        return ' '.join(p.get('text', '') for p in content if p.get('type') == 'text')
    return str(content)


def _has_image(messages: list) -> bool:
    for msg in messages:
        if msg.get('role') == 'user':
            content = msg.get('content', [])
            if isinstance(content, list) and any(p.get('type') == 'image_url' for p in content):
                return True
    return False


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
                'max_tokens': 512,
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

    result = json.loads(cleaned)
    tasks = result.get('tasks', [])

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
        content = await _run_web_search(prompt)
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


def _plan_lines(tasks: list) -> list[str]:
    """Return plan as a list of lines to stream one by one."""
    lines = ['**Задачи:**\n']
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

    # Internal OpenWebUI system tasks (title gen, follow-ups, tags) — route directly
    if user_text.startswith('### Task:'):
        log.debug('[smart_router] internal system task, skipping routing')
        model = MODEL_REGISTRY['text']
        content = await _run_chat(messages, model, api_key)
        yield _sse_chunk(content)
        yield _sse_done()
        return

    # Decompose
    try:
        tasks = await _decompose(user_text, has_image, api_key)
    except Exception as exc:
        log.error(f'[smart_router] decompose FAILED: {exc!r}, falling back to text')
        tasks = []

    if not tasks:
        tasks = [{'id': 0, 'type': 'text', 'prompt': user_text, 'depends_on': []}]

    log.info(f'[smart_router] task plan: {[t["type"] for t in tasks]}')

    # Stream plan line by line so it appears smoothly before tasks run
    for line in _plan_lines(tasks):
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

    # Show only leaf results — tasks whose output is not consumed by other tasks
    dep_ids = {d for t in tasks for d in t.get('depends_on', [])}
    leaf_results = [r for r in results if r['id'] not in dep_ids]
    body = '\n\n---\n\n'.join(r['content'] for r in leaf_results)
    yield _sse_chunk(body)
    yield _sse_done()
