"""
Гибридный роутер
Комбинирует rule-based и LLM-based подходы
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .analyzer import RequestAnalyzer, RequestFeatures
from .rule_engine import RuleEngine, RouteDecision
from .llm_router import LLMRouter
from .models_config import ModelType

logger = logging.getLogger(__name__)

class HybridRouter:
    """Главный роутер с гибридной логикой"""
    
    def __init__(self, mws_client, enable_llm_fallback: bool = True):
        """
        Args:
            mws_client: Клиент для MWS GPT API
            enable_llm_fallback: Использовать ли LLM когда правила не срабатывают
        """
        self.analyzer = RequestAnalyzer()
        self.rule_engine = RuleEngine()
        self.llm_router = LLMRouter(mws_client) if enable_llm_fallback else None
        
        self.enable_llm_fallback = enable_llm_fallback
        
        # Статистика для мониторинга
        self.stats = {
            'total_requests': 0,
            'rule_based': 0,
            'llm_based': 0,
            'manual_override': 0
        }
    
    def route(
        self,
        text: str,
        files: Optional[list] = None,
        context: Optional[Dict[str, Any]] = None,
        manual_model: Optional[str] = None,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Главный метод маршрутизации
        
        Args:
            text: Текст запроса
            files: Прикрепленные файлы
            context: Контекст диалога
            manual_model: Ручной выбор модели (если пользователь переключил)
            user_preferences: Предпочтения пользователя (память)
            
        Returns:
            Dict с решением о маршрутизации и метаданными
        """
        
        start_time = datetime.now()
        self.stats['total_requests'] += 1
        
        # Если пользователь вручную выбрал модель
        if manual_model:
            return self._handle_manual_override(manual_model, text)
        
        # ШАГ 1: Анализ запроса
        features = self.analyzer.analyze(text, files, context)
        
        logger.info(f"Request analysis: {self.analyzer.get_analysis_summary(features)}")
        
        # ШАГ 2: Проверка правил (быстрый путь)
        rule_decision = self.rule_engine.evaluate(features)
        
        if rule_decision:
            # Правило сработало — используем его
            self.stats['rule_based'] += 1
            routing_method = "rule_based"
            decision = rule_decision
            
            logger.info(f"Rule matched: {decision.rule_matched} → {decision.model_name}")
        
        else:
            # Правила не сработали — используем LLM
            if self.enable_llm_fallback and self.llm_router:
                self.stats['llm_based'] += 1
                routing_method = "llm_based"
                decision = self.llm_router.route(features, context)
                
                logger.info(f"LLM routing → {decision.model_name} (confidence: {decision.confidence})")
            
            else:
                # Fallback на текстовую модель
                routing_method = "fallback"
                decision = RouteDecision(
                    model_type=ModelType.TEXT_LLM,
                    model_name="gpt-4o",
                    confidence=0.5,
                    reasoning="No rules matched, LLM router disabled. Using default text model."
                )
        
        # ШАГ 3: Применение пользовательских предпочтений
        decision = self._apply_user_preferences(decision, user_preferences)
        
        # Вычисляем время роутинга
        routing_time = (datetime.now() - start_time).total_seconds()
        
        # Формируем результат
        result = {
            'model_name': decision.model_name,
            'model_type': decision.model_type.value,
            'confidence': decision.confidence,
            'reasoning': decision.reasoning,
            'routing_method': routing_method,
            'routing_time_ms': int(routing_time * 1000),
            'features': {
                'has_image': features.has_image,
                'has_audio': features.has_audio,
                'has_document': features.has_document,
                'has_url': features.has_url,
                'is_complex': features.is_complex,
                'is_multimodal': features.is_multimodal,
                'detected_intent': features.detected_intent.value if features.detected_intent else None
            },
            'metadata': {
                'rule_matched': decision.rule_matched,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        # Логируем решение
        self._log_decision(result)
        
        return result
    
    def _handle_manual_override(self, manual_model: str, text: str) -> Dict[str, Any]:
        """Обработка ручного выбора модели"""
        
        self.stats['manual_override'] += 1
        
        logger.info(f"Manual model override: {manual_model}")
        
        return {
            'model_name': manual_model,
            'model_type': 'manual',
            'confidence': 1.0,
            'reasoning': f"Пользователь вручную выбрал модель {manual_model}",
            'routing_method': 'manual',
            'routing_time_ms': 0,
            'features': {},
            'metadata': {
                'manual_override': True,
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def _apply_user_preferences(
        self,
        decision: RouteDecision,
        preferences: Optional[Dict[str, Any]]
    ) -> RouteDecision:
        """Применение пользовательских предпочтений"""
        
        if not preferences:
            return decision
        
        # Пример: пользователь предпочитает более быструю модель
        if preferences.get('prefer_speed', False):
            if decision.model_type == ModelType.TEXT_LLM:
                decision.model_name = "gpt-4o-mini"
                decision.reasoning += " (используется быстрая модель по предпочтению пользователя)"
        
        # Пример: пользователь предпочитает более качественные ответы
        if preferences.get('prefer_quality', False):
            if decision.model_type == ModelType.TEXT_LLM:
                decision.model_name = "gpt-4o"
                decision.reasoning += " (используется качественная модель по предпочтению пользователя)"
        
        return decision
    
    def _log_decision(self, result: Dict[str, Any]):
        """Логирование решения о маршрутизации"""
        
        log_entry = {
            'timestamp': result['metadata']['timestamp'],
            'model': result['model_name'],
            'method': result['routing_method'],
            'confidence': result['confidence'],
            'time_ms': result['routing_time_ms'],
            'reasoning': result['reasoning']
        }
        
        # Логируем в файл для анализа
        logger.info(f"Routing decision: {log_entry}")
        
        # Можно также сохранять в БД или отдельный файл для аналитики
    
    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику работы роутера"""
        
        total = self.stats['total_requests']
        
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            'rule_based_percent': round(self.stats['rule_based'] / total * 100, 2),
            'llm_based_percent': round(self.stats['llm_based'] / total * 100, 2),
            'manual_percent': round(self.stats['manual_override'] / total * 100, 2)
        }