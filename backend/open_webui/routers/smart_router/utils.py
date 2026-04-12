"""
Вспомогательные функции для роутера
"""

import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class RouterLogger:
    """Логгер решений роутера для анализа и отладки"""
    
    def __init__(self, log_file: str = "logs/router_decisions.log"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def log_decision(
        self,
        request_id: str,
        decision: Dict[str, Any],
        user_feedback: Optional[str] = None
    ):
        """
        Логирование решения роутера
        
        Args:
            request_id: Уникальный ID запроса
            decision: Решение роутера
            user_feedback: Фидбек пользователя (если есть)
        """
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'request_id': request_id,
            'decision': decision,
            'user_feedback': user_feedback
        }
        
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"Failed to write router log: {e}")
    
    def get_recent_decisions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Получить последние решения роутера"""
        
        if not self.log_file.exists():
            return []
        
        decisions = []
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines[-limit:]:
                    decisions.append(json.loads(line.strip()))
        except Exception as e:
            logger.error(f"Failed to read router log: {e}")
        
        return decisions
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Анализ производительности роутера"""
        
        decisions = self.get_recent_decisions(limit=1000)
        
        if not decisions:
            return {}
        
        # Статистика по методам роутинга
        routing_methods = {}
        total_time = 0
        model_usage = {}
        confidence_sum = 0
        
        for decision in decisions:
            dec_data = decision.get('decision', {})
            
            # Подсчет методов
            method = dec_data.get('routing_method', 'unknown')
            routing_methods[method] = routing_methods.get(method, 0) + 1
            
            # Время роутинга
            total_time += dec_data.get('routing_time_ms', 0)
            
            # Использование моделей
            model = dec_data.get('model_name', 'unknown')
            model_usage[model] = model_usage.get(model, 0) + 1
            
            # Средняя уверенность
            confidence_sum += dec_data.get('confidence', 0)
        
        avg_time = total_time / len(decisions) if decisions else 0
        avg_confidence = confidence_sum / len(decisions) if decisions else 0
        
        return {
            'total_decisions': len(decisions),
            'routing_methods': routing_methods,
            'avg_routing_time_ms': round(avg_time, 2),
            'avg_confidence': round(avg_confidence, 3),
            'model_usage': model_usage,
            'most_used_model': max(model_usage.items(), key=lambda x: x[1])[0] if model_usage else None
        }


class FileTypeDetector:
    """Определение типов файлов"""
    
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg'}
    AUDIO_EXTENSIONS = {'.mp3', '.wav', '.ogg', '.m4a', '.flac', '.aac'}
    DOCUMENT_EXTENSIONS = {'.pdf', '.doc', '.docx', '.txt', '.md', '.csv', '.xlsx', '.xls', '.pptx'}
    VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    
    @classmethod
    def detect_file_type(cls, filename: str, mime_type: Optional[str] = None) -> str:
        """
        Определить тип файла
        
        Returns:
            'image', 'audio', 'document', 'video' или 'unknown'
        """
        
        extension = Path(filename).suffix.lower()
        
        if extension in cls.IMAGE_EXTENSIONS:
            return 'image'
        elif extension in cls.AUDIO_EXTENSIONS:
            return 'audio'
        elif extension in cls.DOCUMENT_EXTENSIONS:
            return 'document'
        elif extension in cls.VIDEO_EXTENSIONS:
            return 'video'
        
        # Fallback на mime_type
        if mime_type:
            if mime_type.startswith('image/'):
                return 'image'
            elif mime_type.startswith('audio/'):
                return 'audio'
            elif mime_type.startswith('video/'):
                return 'video'
            elif any(x in mime_type for x in ['pdf', 'document', 'text']):
                return 'document'
        
        return 'unknown'
    
    @classmethod
    def is_supported(cls, filename: str) -> bool:
        """Проверить, поддерживается ли файл"""
        
        extension = Path(filename).suffix.lower()
        all_supported = (
            cls.IMAGE_EXTENSIONS | 
            cls.AUDIO_EXTENSIONS | 
            cls.DOCUMENT_EXTENSIONS | 
            cls.VIDEO_EXTENSIONS
        )
        
        return extension in all_supported


class ContextManager:
    """Управление контекстом диалога для роутера"""
    
    def __init__(self, max_context_messages: int = 10):
        self.max_context_messages = max_context_messages
    
    def extract_context(
        self,
        messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Извлечь релевантный контекст из истории сообщений
        
        Args:
            messages: История сообщений диалога
            
        Returns:
            Структурированный контекст
        """
        
        # Берем только последние N сообщений
        recent_messages = messages[-self.max_context_messages:] if messages else []
        
        context = {
            'messages': recent_messages,
            'conversation_length': len(messages),
            'has_images_in_history': False,
            'last_model_used': None,
            'topics': []
        }
        
        # Анализируем историю
        for msg in recent_messages:
            # Проверка наличия изображений
            if msg.get('images') or msg.get('attachments'):
                context['has_images_in_history'] = True
            
            # Последняя использованная модель
            if msg.get('model'):
                context['last_model_used'] = msg['model']
        
        return context
    
    def should_continue_with_model(
        self,
        context: Dict[str, Any],
        current_decision: str
    ) -> bool:
        """
        Определить, стоит ли продолжить использовать ту же модель
        
        Полезно для сохранения контекста при работе с Vision или специализированными моделями
        """
        
        last_model = context.get('last_model_used')
        
        if not last_model:
            return False
        
        # Если последняя модель была Vision и в истории есть изображения
        # И новый запрос тоже про изображения
        if 'vision' in last_model.lower() and context.get('has_images_in_history'):
            return True
        
        return False


class PromptBuilder:
    """Построение промптов для различных сценариев"""
    
    @staticmethod
    def build_vision_prompt(user_text: str, image_count: int) -> str:
        """Промпт для Vision модели"""
        
        if not user_text:
            return f"Пожалуйста, опиши подробно, что изображено на {'этом изображении' if image_count == 1 else f'этих {image_count} изображениях'}."
        
        return user_text
    
    @staticmethod
    def build_rag_prompt(user_text: str, document_names: List[str]) -> str:
        """Промпт для RAG системы"""
        
        docs_list = ', '.join(document_names)
        
        prompt = f"""Пользователь загрузил следующие документы: {docs_list}

Вопрос пользователя: {user_text}

Пожалуйста, ответь на вопрос, используя информацию из загруженных документов. 
Если ответа нет в документах, так и скажи."""

        return prompt
    
    @staticmethod
    def build_web_search_prompt(user_text: str, urls: Optional[List[str]] = None) -> str:
        """Промпт для веб-поиска"""
        
        if urls:
            urls_list = '\n'.join(f"- {url}" for url in urls)
            prompt = f"""Пользователь предоставил следующие ссылки:
{urls_list}

Вопрос: {user_text}

Пожалуйста, проанализируй содержимое этих страниц и ответь на вопрос."""
        else:
            prompt = f"""Пользователь просит найти информацию в интернете.

Запрос: {user_text}

Пожалуйста, найди актуальную информацию и предоставь развернутый ответ."""
        
        return prompt
    
    @staticmethod
    def build_research_prompt(user_text: str) -> str:
        """Промпт для исследовательского режима"""
        
        prompt = f"""Это запрос на глубокий анализ темы.

Тема: {user_text}

Пожалуйста, проведи многоэтапное исследование:
1. Найди актуальную информацию из разных источников
2. Проанализируй и структурируй данные
3. Выдели ключевые моменты и инсайты
4. Предоставь развернутый, хорошо структурированный ответ

Используй критическое мышление и проверяй факты."""

        return prompt


class ResponseFormatter:
    """Форматирование ответов от различных моделей"""
    
    @staticmethod
    def format_vision_response(
        response: str,
        image_urls: List[str],
        model_name: str
    ) -> Dict[str, Any]:
        """Форматирование ответа Vision модели"""
        
        return {
            'type': 'vision',
            'content': response,
            'images_analyzed': len(image_urls),
            'model': model_name,
            'metadata': {
                'image_urls': image_urls
            }
        }
    
    @staticmethod
    def format_asr_response(
        transcription: str,
        audio_duration: Optional[float] = None,
        model_name: str = 'whisper'
    ) -> Dict[str, Any]:
        """Форматирование ответа ASR модели"""
        
        return {
            'type': 'asr',
            'content': transcription,
            'model': model_name,
            'metadata': {
                'audio_duration_seconds': audio_duration,
                'transcription_length': len(transcription)
            }
        }
    
    @staticmethod
    def format_image_gen_response(
        image_url: str,
        prompt_used: str,
        model_name: str = 'dall-e-3'
    ) -> Dict[str, Any]:
        """Форматирование ответа генератора изображений"""
        
        return {
            'type': 'image_generation',
            'content': f"Изображение создано по запросу: {prompt_used}",
            'image_url': image_url,
            'model': model_name,
            'metadata': {
                'prompt': prompt_used
            }
        }
    
    @staticmethod
    def format_text_response(
        response: str,
        model_name: str,
        tokens_used: Optional[int] = None
    ) -> Dict[str, Any]:
        """Форматирование текстового ответа"""
        
        return {
            'type': 'text',
            'content': response,
            'model': model_name,
            'metadata': {
                'tokens_used': tokens_used,
                'response_length': len(response)
            }
        }


def generate_request_id() -> str:
    """Генерация уникального ID запроса"""
    import uuid
    return str(uuid.uuid4())


def validate_routing_decision(decision: Dict[str, Any]) -> bool:
    """
    Валидация решения роутера
    
    Проверяет, что все необходимые поля присутствуют
    """
    
    required_fields = ['model_name', 'model_type', 'confidence', 'reasoning']
    
    for field in required_fields:
        if field not in decision:
            logger.error(f"Missing required field in routing decision: {field}")
            return False
    
    # Проверка корректности значений
    if not isinstance(decision['confidence'], (int, float)):
        logger.error("Confidence must be a number")
        return False
    
    if not 0 <= decision['confidence'] <= 1:
        logger.error("Confidence must be between 0 and 1")
        return False
    
    return True


def sanitize_user_input(text: str, max_length: int = 10000) -> str:
    """
    Очистка пользовательского ввода
    
    Args:
        text: Входной текст
        max_length: Максимальная длина
        
    Returns:
        Очищенный текст
    """
    
    # Обрезаем слишком длинный текст
    if len(text) > max_length:
        text = text[:max_length]
        logger.warning(f"User input truncated to {max_length} characters")
    
    # Удаляем потенциально опасные символы (базовая санитизация)
    # В продакшне используйте более продвинутые методы
    text = text.strip()
    
    return text