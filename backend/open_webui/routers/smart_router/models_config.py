"""
Конфигурация всех доступных моделей и инструментов
"""

from typing import Dict, List
from enum import Enum

class ModelType(Enum):
    """Типы моделей"""
    VISION = "vision"
    ASR = "asr"
    TEXT_LLM = "text_llm"
    IMAGE_GEN = "image_gen"
    WEB_SEARCH = "web_search"
    WEB_PARSER = "web_parser"
    RAG = "rag"
    RESEARCH = "research"

class ModelConfig:
    """Конфигурация конкретной модели"""
    def __init__(
        self,
        name: str,
        model_type: ModelType,
        endpoint: str,
        description: str,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        supports_streaming: bool = True
    ):
        self.name = name
        self.model_type = model_type
        self.endpoint = endpoint
        self.description = description
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.supports_streaming = supports_streaming

# Маппинг доступных моделей
AVAILABLE_MODELS: Dict[str, ModelConfig] = {
    # Vision модели
    "gpt-4o-vision": ModelConfig(
        name="gpt-4o-vision",
        model_type=ModelType.VISION,
        endpoint="/v1/chat/completions",
        description="Анализ изображений, описание, OCR, поиск объектов",
        max_tokens=4096
    ),
    
    # ASR модели
    "whisper": ModelConfig(
        name="whisper",
        model_type=ModelType.ASR,
        endpoint="/v1/audio/transcriptions",
        description="Распознавание речи из аудио файлов",
        supports_streaming=False
    ),
    
    # Текстовые LLM
    "gpt-4o": ModelConfig(
        name="gpt-4o",
        model_type=ModelType.TEXT_LLM,
        endpoint="/v1/chat/completions",
        description="Основная модель для текстовых диалогов, сложных рассуждений",
        max_tokens=8192
    ),
    
    "gpt-4o-mini": ModelConfig(
        name="gpt-4o-mini",
        model_type=ModelType.TEXT_LLM,
        endpoint="/v1/chat/completions",
        description="Быстрая модель для простых запросов",
        max_tokens=4096,
        temperature=0.5
    ),
    
    # Генерация изображений
    "dall-e-3": ModelConfig(
        name="dall-e-3",
        model_type=ModelType.IMAGE_GEN,
        endpoint="/v1/images/generations",
        description="Создание изображений по текстовому описанию",
        supports_streaming=False
    ),
    
    # Инструменты
    "web-search": ModelConfig(
        name="web-search",
        model_type=ModelType.WEB_SEARCH,
        endpoint="/tools/search",
        description="Поиск актуальной информации в интернете"
    ),
    
    "web-parser": ModelConfig(
        name="web-parser",
        model_type=ModelType.WEB_PARSER,
        endpoint="/tools/parse",
        description="Парсинг содержимого веб-страниц"
    ),
}

# Ключевые слова для определения намерений
INTENT_KEYWORDS = {
    ModelType.IMAGE_GEN: [
        "нарисуй", "создай изображение", "сгенерируй картинку", 
        "визуализируй", "покажи как выглядит", "создай иллюстрацию",
        "нарисовать", "изобрази", "картинка", "рисунок"
    ],
    
    ModelType.WEB_SEARCH: [
        "найди в интернете", "поищи информацию", "что известно о",
        "последние новости", "актуальная информация", "погода",
        "курс", "что говорят о", "search", "гугл"
    ],
    
    ModelType.RESEARCH: [
        "глубокий анализ", "исследуй тему", "подробный отчет",
        "проанализируй со всех сторон", "deep dive", "research"
    ]
}

# Приоритеты правил (чем выше, тем важнее)
RULE_PRIORITIES = {
    "has_image": 100,
    "has_audio": 99,
    "has_document": 98,
    "has_url": 90,
    "intent_image_gen": 85,
    "intent_web_search": 80,
    "intent_research": 75,
    "default": 0
}

def get_model_by_type(model_type: ModelType) -> ModelConfig:
    """Получить дефолтную модель для типа"""
    type_mapping = {
        ModelType.VISION: "gpt-4o-vision",
        ModelType.ASR: "whisper",
        ModelType.TEXT_LLM: "gpt-4o",
        ModelType.IMAGE_GEN: "dall-e-3",
        ModelType.WEB_SEARCH: "web-search",
        ModelType.WEB_PARSER: "web-parser",
    }
    model_name = type_mapping.get(model_type)
    return AVAILABLE_MODELS.get(model_name)