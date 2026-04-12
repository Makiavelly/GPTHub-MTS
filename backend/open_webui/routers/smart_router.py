import asyncio
import json
import logging
import re

import aiohttp

log = logging.getLogger(__name__)

# Model used for decomposition — fastest available
ROUTER_MODEL = "mws-gpt-alpha"

# MWS API base URL — used for all smart router calls directly, bypassing OpenWebUI proxy
MWS_BASE_URL = "https://api.gpt.mws.ru/v1"

# Maps task type → actual model ID on MWS
MODEL_REGISTRY = {
    "text":             "qwen2.5-72b-instruct",
    "reasoning":        "QwQ-32B",
    "coding":           "qwen3-coder-480b-a35b",
    "vlm":              "qwen2.5-vl",
    "image_generation": "qwen-image-lightning",
}

DECOMPOSE_SYSTEM_PROMPT = """\
You are a task router. Analyze the user message and output a JSON task list.

Task types:
- "text" — writing text, answering questions, creating descriptions, posts, articles
- "reasoning" — math, logic, deep analysis
- "coding" — writing or fixing code
- "vlm" — analyzing an attached image
- "image_generation" — creating/drawing/generating a new image, logo, illustration, picture

IMPORTANT: If the user asks to generate, draw, or create an image/logo/picture/illustration — use "image_generation".

Output format (JSON only, no extra text, no markdown):
{"tasks":[{"id":0,"type":"TYPE","prompt":"PROMPT","depends_on":[]}]}

Examples:
User: "generate a logo and write an ad post for a bakery"
Output: {"tasks":[{"id":0,"type":"image_generation","prompt":"Logo for a bakery, minimalist style, warm colors, bread and wheat motif"},{"id":1,"type":"text","prompt":"Write an advertising post for a bakery","depends_on":[]}]}

User: "write a python function to sort a list"
Output: {"tasks":[{"id":0,"type":"coding","prompt":"Write a Python function to sort a list","depends_on":[]}]}

Rules:
- Split compound requests into multiple tasks
- For image_generation: write a detailed visual description as the prompt
- Output ONLY the JSON, nothing else"""


def _extract_text(message: dict) -> str:
    content = message.get("content", "")
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if p.get("type") == "text")
    return str(content)


def _has_image(messages: list) -> bool:
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", [])
            if isinstance(content, list) and any(p.get("type") == "image_url" for p in content):
                return True
    return False


async def _decompose(user_text: str, has_image: bool, api_key: str) -> list[dict]:
    note = "[Note: the user has attached an image to this message]\n\n" if has_image else ""
    messages = [
        {"role": "system", "content": DECOMPOSE_SYSTEM_PROMPT},
        {"role": "user", "content": f"{note}{user_text}"},
    ]

    log.info(f"[smart_router] _decompose → calling {ROUTER_MODEL}")
    log.info(f"[smart_router] user_text = {user_text!r}")

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{MWS_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": ROUTER_MODEL,
                "messages": messages,
                "temperature": 0,
                "max_tokens": 512,
            },
        ) as resp:
            data = await resp.json()

    if "choices" not in data:
        log.error(f"[smart_router] decomposer returned no 'choices': {data}")
        raise KeyError("choices")

    raw = data["choices"][0]["message"]["content"]
    log.info(f"[smart_router] decomposer raw response:\n{raw}")

    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
    # Some models wrap output in <think>...</think> — strip it
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()

    result = json.loads(cleaned)
    tasks = result.get("tasks", [])

    log.info(f"[smart_router] parsed tasks: {json.dumps(tasks, ensure_ascii=False, indent=2)}")
    return tasks


async def _run_image_generation(prompt: str, api_key: str) -> str:
    log.info(f"[smart_router] image_generation → model=qwen-image-lightning")
    log.info(f"[smart_router] image prompt: {prompt!r}")
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{MWS_BASE_URL}/images/generations",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": "qwen-image-lightning", "prompt": prompt, "n": 1},
        ) as resp:
            data = await resp.json()
    log.info(f"[smart_router] image_generation response keys: {list(data.keys())}")

    item = data["data"][0]
    url = item.get("url")
    if url:
        return f"![Generated image]({url})"
    b64 = item.get("b64_json", "")
    return f"![Generated image](data:image/png;base64,{b64})"


async def _run_chat(messages: list, model: str, api_key: str) -> str:
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{MWS_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "messages": messages, "temperature": 0.7},
        ) as resp:
            data = await resp.json()
    return data["choices"][0]["message"]["content"]


async def _execute_task(
    task: dict,
    context: dict,
    original_messages: list,
    api_key: str,
) -> dict:
    task_type = task["type"]
    model = MODEL_REGISTRY.get(task_type, MODEL_REGISTRY["text"])
    log.info(f"[smart_router] executing task id={task['id']} type={task_type} → model={model}")

    # Build prompt, appending output of dependency tasks as context
    prompt = task["prompt"]
    for dep_id in task.get("depends_on", []):
        if dep_id in context:
            prompt += f"\n\nContext from previous step:\n{context[dep_id]}"

    log.info(f"[smart_router] task id={task['id']} final prompt: {prompt!r}")

    if task_type == "image_generation":
        content = await _run_image_generation(prompt, api_key)
        return {"id": task["id"], "type": "image", "content": content}

    # For all chat-based tasks (text / reasoning / coding / vlm)
    # Use conversation history up to (not including) the last user message
    history = [m for m in original_messages if m.get("role") in ("system", "assistant")]
    user_msg: dict = {"role": "user", "content": prompt}

    # For vlm: carry the image parts from the original user message
    if task_type == "vlm":
        last_user = next((m for m in reversed(original_messages) if m.get("role") == "user"), None)
        if last_user:
            orig_content = last_user.get("content", [])
            if isinstance(orig_content, list):
                image_parts = [p for p in orig_content if p.get("type") == "image_url"]
                if image_parts:
                    user_msg["content"] = [{"type": "text", "text": prompt}] + image_parts

    messages = history + [user_msg]
    content = await _run_chat(messages, model, api_key)
    return {"id": task["id"], "type": "text", "content": content}


def _combine(results: list) -> str:
    if len(results) == 1:
        return results[0]["content"]
    return "\n\n---\n\n".join(r["content"] for r in results)


async def route(payload: dict, api_key: str, base_url: str = "") -> dict:
    """
    Analyze the request and decide how to handle it.

    Returns one of:
      {"action": "redirect", "model": "<model_id>"}
          Single task that maps to a chat-completion model.
          Caller should update payload["model"] and continue normal flow.

      {"action": "respond", "content": "<markdown>"}
          Multiple tasks or image generation executed internally.
          Caller should return this content directly as a chat completion response.
    """
    # Normalize key: strip any existing "Bearer " prefix so we don't double it
    api_key = api_key.removeprefix("Bearer ").strip()
    log.info(f"[smart_router] api_key preview: {api_key[:8]!r}... (len={len(api_key)})")

    messages = payload.get("messages", [])
    has_image = _has_image(messages)

    last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
    user_text = _extract_text(last_user) if last_user else ""

    # Skip routing for OpenWebUI's internal system tasks (title gen, follow-ups, tags, etc.)
    if user_text.startswith("### Task:"):
        log.debug("[smart_router] internal system task detected, skipping routing → text")
        return {"action": "redirect", "model": MODEL_REGISTRY["text"]}

    try:
        tasks = await _decompose(user_text, has_image, api_key)
    except Exception as exc:
        log.error(f"[smart_router] decompose FAILED: {exc!r}, falling back to plain text")
        tasks = []

    if not tasks:
        log.warning("[smart_router] task list is empty after decompose, using text fallback")
        tasks = [{"id": 0, "type": "text", "prompt": user_text, "depends_on": []}]

    log.info(f"[smart_router] final task plan: {[t['type'] for t in tasks]}")

    # Single non-image task → just redirect to the right model, no extra LLM call
    if len(tasks) == 1 and tasks[0]["type"] != "image_generation":
        model = MODEL_REGISTRY.get(tasks[0]["type"], MODEL_REGISTRY["text"])
        log.info(f"[smart_router] action=redirect → {model}")
        return {"action": "redirect", "model": model}

    # Multiple tasks or image generation → execute and combine
    context: dict[int, str] = {}
    results: list[dict] = []
    remaining = list(tasks)

    while remaining:
        ready = [t for t in remaining if all(d in context for d in t.get("depends_on", []))]
        if not ready:
            log.error("smart_router: dependency cycle or unsatisfied deps, aborting")
            break

        batch = await asyncio.gather(
            *[_execute_task(t, context, messages, api_key) for t in ready],
            return_exceptions=True,
        )

        for task, res in zip(ready, batch):
            if isinstance(res, Exception):
                log.error(f"smart_router: task {task['id']} ({task['type']}) failed: {res}")
                fallback = f"[Task '{task['type']}' failed: {res}]"
                context[task["id"]] = fallback
                results.append({"id": task["id"], "type": "text", "content": fallback})
            else:
                context[res["id"]] = res["content"]
                results.append(res)

        remaining = [t for t in remaining if t not in ready]

    return {"action": "respond", "content": _combine(results)}
