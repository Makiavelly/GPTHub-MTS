# batch_processor.py
import json
import os
import asyncio
import aiohttp
from collections import deque
from datetime import datetime
from typing import List, Dict
import time

PROFILES_DIR = 'D:/MTS/open-webui/backend/open_webui/data/profiles'
PENDING_DIR = 'D:/MTS/open-webui/backend/open_webui/data/pending_messages'


class BatchProfileUpdater:
    def __init__(self, user_id: str, batch_size: int = 10):
        self.user_id = user_id
        self.batch_size = batch_size
        self.pending_file = os.path.join(PENDING_DIR, f'{user_id}.json')
        self.profile_path = os.path.join(PROFILES_DIR, f'{user_id}.json')
        os.makedirs(PROFILES_DIR, exist_ok=True)
        os.makedirs(PENDING_DIR, exist_ok=True)
        self._load_pending()

    def _load_pending(self):
        """Загружает накопленные сообщения"""
        if os.path.exists(self.pending_file):
            try:
                with open(self.pending_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.pending_messages = deque(data.get('messages', []), maxlen=100)
            except:
                self.pending_messages = deque(maxlen=100)
        else:
            self.pending_messages = deque(maxlen=100)

    def _save_pending(self):
        """Сохраняет накопленные сообщения"""
        with open(self.pending_file, 'w', encoding='utf-8') as f:
            json.dump(
                {'messages': list(self.pending_messages), 'last_updated': time.time()}, f, ensure_ascii=False, indent=2
            )

    def add_message(self, message: str):
        """Добавляет сообщение в очередь"""
        self.pending_messages.append({'content': message, 'timestamp': time.time()})
        self._save_pending()

        # Если накопилось достаточно - запускаем обработку
        if len(self.pending_messages) >= self.batch_size:
            # Запускаем в фоне, не блокируя
            asyncio.create_task(self.process_batch())

    def get_current_profile(self) -> Dict:
        """Получает текущий профиль"""
        if os.path.exists(self.profile_path):
            try:
                with open(self.profile_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {'tech_stack': {'languages': [], 'frameworks': [], 'databases': [], 'tools': []}}

    def save_profile(self, profile: Dict):
        """Сохраняет профиль"""
        profile['last_updated'] = time.time()
        with open(self.profile_path, 'w', encoding='utf-8') as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)
        print(f'✅ Профиль обновлён для {self.user_id}')

    async def process_batch(self):
        """Обрабатывает накопленные сообщения через нейронку"""
        if not self.pending_messages:
            return

        print(f'\n🧠 Обрабатываем батч из {len(self.pending_messages)} сообщений для {self.user_id}')

        # Берём сообщения и очищаем очередь
        messages_to_process = list(self.pending_messages)
        self.pending_messages.clear()
        self._save_pending()

        # Получаем текущий профиль
        current_profile = self.get_current_profile()

        # Отправляем в нейронку
        updated_profile = await self._call_llm_for_profile(current_profile, messages_to_process)

        if updated_profile and updated_profile != current_profile:
            self.save_profile(updated_profile)
            print(f'   Профиль обновлён: {updated_profile["tech_stack"]}')
        else:
            print(f'   Профиль не изменился')

    async def _call_llm_for_profile(self, current_profile: Dict, messages: List[Dict]) -> Dict:
        """Вызывает нейронку для анализа сообщений"""

        # Собираем текст сообщений
        conversation = '\n'.join([f'Сообщение {i + 1}: {msg["content"]}' for i, msg in enumerate(messages)])

        system_prompt = """Ты — модуль обновления IT-профиля пользователя.
Проанализируй историю сообщений пользователя и обнови его tech stack.
Верни ТОЛЬКО JSON, без пояснений.

Формат ответа:
{
    "tech_stack": {
        "languages": ["python", "javascript"],
        "frameworks": ["django", "react"],
        "databases": ["postgresql"],
        "tools": ["docker", "git"]
        "project": ["application", "webservice"]
    }
}

Правила:
1. Добавляй новые технологии, НЕ удаляй существующие
2. Используй lowercase
3. Убирай дубликаты
4. Если технология уже есть в текущем профиле - не добавляй повторно
5. Если пользователь говорит, что он больше не использует технологию, то удали ее
"""

        user_prompt = f"""Текущий профиль пользователя:
{json.dumps(current_profile, ensure_ascii=False)}

История сообщений пользователя (последние {len(messages)}):
{conversation}

Обнови профиль на основе этих сообщений. Верни ТОЛЬКО JSON."""

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    'https://api.gpt.mws.ru/v1/chat/completions',
                    headers={'Authorization': 'Bearer sk-ewgiaPC3A6pPDYHwR8siVA', 'Content-Type': 'application/json'},
                    json={
                        'model': 'qwen2.5-72b-instruct',
                        'messages': [
                            {'role': 'system', 'content': system_prompt},
                            {'role': 'user', 'content': user_prompt},
                        ],
                        'temperature': 0.1,
                    },
                    timeout=aiohttp.ClientTimeout(total=60),  # Большой таймаут для батча
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data['choices'][0]['message']['content']

                        # Очищаем от markdown
                        import re

                        content = re.sub(r'```json\s*', '', content)
                        content = re.sub(r'```\s*$', '', content)
                        content = content.strip()

                        result = json.loads(content)

                        return result
                    else:
                        print(f'   API error: {response.status}')
                        return current_profile

        except Exception as e:
            print(f'   Ошибка вызова LLM: {e}')
            return current_profile


# Глобальный словарь для хранения экземпляров по user_id
_updaters = {}


def get_updater(user_id: str) -> BatchProfileUpdater:
    """Получает или создаёт экземпляр BatchProfileUpdater для пользователя"""
    if user_id not in _updaters:
        _updaters[user_id] = BatchProfileUpdater(user_id, batch_size=5)  # Каждые 5 сообщений
    return _updaters[user_id]


async def add_message_to_profile(user_id: str, message: str):
    """Добавляет сообщение в очередь для обработки"""
    updater = get_updater(user_id)
    updater.add_message(message)
