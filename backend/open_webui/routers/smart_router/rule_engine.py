"""
Rule-based маршрутизация
Быстрый путь для очевидных случаев
"""

from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from .analyzer import RequestFeatures
from .models_config import ModelType, RULE_PRIORITIES, get_model_by_type

@dataclass
class RouteDecision:
    """Решение о маршрутизации"""
    
    model_type: ModelType
    model_name: str
    confidence: float  # 0.0 - 1.0
    reasoning: str  # Объяснение, почему выбрана эта модель
    rule_matched: Optional[str] = None  # Какое правило сработало
    
    def to_dict(self) -> Dict[str, Any]:
        """Сериализация для логов"""
        return {
            'model_type': self.model_type.value,
            'model_name': self.model_name,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'rule_matched': self.rule_matched
        }

class RuleEngine:
    """Движок правил для маршрутизации"""
    
    def __init__(self):
        self.priorities = RULE_PRIORITIES
    
    def evaluate(self, features: RequestFeatures) -> Optional[RouteDecision]:
        """
        Проверка всех правил и выбор наиболее подходящего
        
        Returns:
            RouteDecision если правило сработало, None если нужен LLM
        """
        
        # Список всех совпавших правил с приоритетами
        matched_rules = []
        
        # ПРАВИЛО 1: Есть изображение → Vision модель
        if features.has_image:
            matched_rules.append((
                self.priorities['has_image'],
                self._create_vision_decision(features)
            ))
        
        # ПРАВИЛО 2: Есть аудио → ASR модель
        if features.has_audio:
            matched_rules.append((
                self.priorities['has_audio'],
                self._create_asr_decision(features)
            ))
        
        # ПРАВИЛО 3: Есть документы → RAG система
        if features.has_document:
            matched_rules.append((
                self.priorities['has_document'],
                self._create_rag_decision(features)
            ))
        
        # ПРАВИЛО 4: Есть URL → Web Parser
        if features.has_url:
            matched_rules.append((
                self.priorities['has_url'],
                self._create_web_parser_decision(features)
            ))
        
        # ПРАВИЛО 5: Явное намерение генерации изображения
        if features.detected_intent == ModelType.IMAGE_GEN:
            matched_rules.append((
                self.priorities['intent_image_gen'],
                self._create_image_gen_decision(features)
            ))
        
        # ПРАВИЛО 6: Явное намерение поиска
        if features.detected_intent == ModelType.WEB_SEARCH:
            matched_rules.append((
                self.priorities['intent_web_search'],
                self._create_web_search_decision(features)
            ))
        
        # ПРАВИЛО 7: Явное намерение исследования
        if features.detected_intent == ModelType.RESEARCH:
            matched_rules.append((
                self.priorities['intent_research'],
                self._create_research_decision(features)
            ))
        
        # Выбираем правило с максимальным приоритетом
        if matched_rules:
            matched_rules.sort(key=lambda x: x[0], reverse=True)
            return matched_rules[0][1]
        
        # Если никакое правило не сработало → отдаем в LLM роутер
        return None
    
    def _create_vision_decision(self, features: RequestFeatures) -> RouteDecision:
        """Решение для Vision модели"""
        model_config = get_model_by_type(ModelType.VISION)
        
        reasoning = f"Обнаружено {features.image_count} изображение(й). "
        if features.text:
            reasoning += f"Вопрос пользователя: '{features.text[:100]}...'"
        else:
            reasoning += "Требуется анализ изображения."
        
        return RouteDecision(
            model_type=ModelType.VISION,
            model_name=model_config.name,
            confidence=1.0,  # 100% уверенность для правил
            reasoning=reasoning,
            rule_matched="has_image"
        )
    
    def _create_asr_decision(self, features: RequestFeatures) -> RouteDecision:
        """Решение для ASR модели"""
        model_config = get_model_by_type(ModelType.ASR)
        
        return RouteDecision(
            model_type=ModelType.ASR,
            model_name=model_config.name,
            confidence=1.0,
            reasoning=f"Обнаружено {features.audio_count} аудио файл(ов). Требуется распознавание речи.",
            rule_matched="has_audio"
        )
    
    def _create_rag_decision(self, features: RequestFeatures) -> RouteDecision:
        """Решение для RAG системы"""
        
        return RouteDecision(
            model_type=ModelType.RAG,
            model_name="gpt-4o-rag",  # Специальная конфигурация с RAG
            confidence=1.0,
            reasoning=f"Обнаружено {features.document_count} документ(ов). Используется RAG для ответа по содержимому файлов.",
            rule_matched="has_document"
        )
    
    def _create_web_parser_decision(self, features: RequestFeatures) -> RouteDecision:
        """Решение для парсинга веб-страниц"""
        model_config = get_model_by_type(ModelType.WEB_PARSER)
        
        urls_str = ', '.join(features.urls[:3])
        if len(features.urls) > 3:
            urls_str += f" и еще {len(features.urls) - 3}"
        
        return RouteDecision(
            model_type=ModelType.WEB_PARSER,
            model_name=model_config.name,
            confidence=1.0,
            reasoning=f"Обнаружены URL: {urls_str}. Требуется парсинг содержимого.",
            rule_matched="has_url"
        )
    
    def _create_image_gen_decision(self, features: RequestFeatures) -> RouteDecision:
        """Решение для генерации изображений"""
        model_config = get_model_by_type(ModelType.IMAGE_GEN)
        
        keywords = ', '.join(features.keywords_matched[:3])
        
        return RouteDecision(
            model_type=ModelType.IMAGE_GEN,
            model_name=model_config.name,
            confidence=0.95,  # Высокая, но не 100% (могут быть исключения)
            reasoning=f"Обнаружены ключевые слова генерации: {keywords}. Запрос на создание изображения.",
            rule_matched="intent_image_gen"
        )
    
    def _create_web_search_decision(self, features: RequestFeatures) -> RouteDecision:
        """Решение для веб-поиска"""
        model_config = get_model_by_type(ModelType.WEB_SEARCH)
        
        keywords = ', '.join(features.keywords_matched[:3])
        
        return RouteDecision(
            model_type=ModelType.WEB_SEARCH,
            model_name=model_config.name,
            confidence=0.9,
            reasoning=f"Обнаружены ключевые слова поиска: {keywords}. Требуется актуальная информация из интернета.",
            rule_matched="intent_web_search"
        )
    
    def _create_research_decision(self, features: RequestFeatures) -> RouteDecision:
        """Решение для исследовательского режима"""
        
        return RouteDecision(
            model_type=ModelType.RESEARCH,
            model_name="research-pipeline",
            confidence=0.85,
            reasoning="Обнаружен запрос на глубокий анализ. Используется multi-step research pipeline.",
            rule_matched="intent_research"
        )