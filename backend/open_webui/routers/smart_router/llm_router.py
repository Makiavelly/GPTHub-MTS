"""
LLM-based маршрутизация
Умный путь для сложных случаев
"""

import json
import logging
from typing import Dict, Any, Optional
from .analyzer import RequestFeatures
from .rule_engine import RouteDecision
from .models_config import ModelType, AVAILABLE_MODELS

logger = logging.getLogger(__name__)

class LLMRouter:
    """Роутер на основе LLM для сложных случаев"""
    
    def __init__(self, mws_client):
        """
        Args:
            mws_client: Клиент для обращения к MWS GPT
        """
        self.client = mws_client
        self.router_model = "gpt-4o-mini"  # Быстрая модель для роутинга
        
    def route(
        self,
        features: RequestFeatures,
        context: Optional[Dict[str, Any]] = None
    ) -> RouteDecision:
        """
        Определение маршрута через LLM
        
        Args:
            features: Признаки запроса
            context: Контекст диалога
            
        Returns:
            RouteDecision с выбранной моделью
        """
        
        try:
            # Формируем промпт для роутера
            prompt = self._build_routing_prompt(features, context)
            
            # Вызываем LLM
            response = self._call_llm(prompt)
            
            # Парсим ответ
            decision = self._parse_llm_response(response, features)
            
            logger.info(f"LLM Router decision: {decision.model_name} (confidence: {decision.confidence})")
            
            return decision
            
        except Exception as e:
            logger.error(f"LLM Router error: {e}. Falling back to default.")
            return self._get_fallback_decision(features)
    
    def _build_routing_prompt(
        self,
        features: RequestFeatures,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Построение промпта для роутера"""
        
        # Описание доступных инструментов
        tools_description = self._get_tools_description()
        
        # Формируем контекст запроса
        request_context = self._format_request_context(features, context)
        
        prompt = f"""Ты — система интеллектуальной маршрутизации запросов к AI-моделям.

ДОСТУПНЫЕ ИНСТРУМЕНТЫ:
{tools_description}

ЗАПРОС ПОЛЬЗОВАТЕЛЯ:
{request_context}

ЗАДАЧА:
Проанализируй запрос и определи, какой инструмент лучше всего подходит для ответа.
Учитывай контекст диалога, если он есть.

ФОРМАТ ОТВЕТА (строго JSON):
{{
  "tool": "название_инструмента",
  "confidence": 0.85,
  "reasoning": "краткое объяснение выбора"
}}

Возможные значения tool: vision, text_llm, image_gen, web_search, web_parser, research

Твой ответ (только JSON, без дополнительного текста):"""

        return prompt
    
    def _get_tools_description(self) -> str:
        """Формирование описания доступных инструментов"""
        
        descriptions = []
        
        # Группируем по типам
        tools = {
            "vision": "Vision Model - для анализа изображений, описания картинок, OCR, поиска объектов на фото",
            "text_llm": "Text LLM - для обычных текстовых диалогов, ответов на вопросы, рассуждений, генерации текста",
            "image_gen": "Image Generator - для создания изображений по текстовому описанию",
            "web_search": "Web Search - для поиска актуальной информации в интернете (новости, факты, погода, курсы)",
            "web_parser": "Web Parser - для извлечения содержимого конкретных веб-страниц по URL",
            "research": "Research Pipeline - для глубокого анализа темы с несколькими шагами (поиск → анализ → синтез)"
        }
        
        for i, (tool, desc) in enumerate(tools.items(), 1):
            descriptions.append(f"{i}. {desc}")
        
        return "\n".join(descriptions)
    
    def _format_request_context(
        self,
        features: RequestFeatures,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Форматирование контекста запроса для промпта"""
        
        parts = []
        
        # Текст запроса
        parts.append(f"Текст: \"{features.text}\"")
        
        # Вложения
        attachments = []
        if features.has_image:
            attachments.append(f"{features.image_count} изображение(й)")
        if features.has_audio:
            attachments.append(f"{features.audio_count} аудио файл(ов)")
        if features.has_document:
            attachments.append(f"{features.document_count} документ(ов)")
        
        if attachments:
            parts.append(f"Вложения: {', '.join(attachments)}")
        
        # URL
        if features.has_url:
            urls_preview = features.urls[:2]
            urls_str = ', '.join(urls_preview)
            if len(features.urls) > 2:
                urls_str += f" (и еще {len(features.urls) - 2})"
            parts.append(f"URL в запросе: {urls_str}")
        
        # Контекст диалога
        if context and context.get('messages'):
            messages_count = len(context['messages'])
            parts.append(f"Предыдущих сообщений в диалоге: {messages_count}")
            
            # Последнее сообщение для контекста
            if messages_count > 0:
                last_msg = context['messages'][-1]
                last_content = last_msg.get('content', '')[:100]
                parts.append(f"Последнее сообщение: \"{last_content}...\"")
        
        # Признаки сложности
        if features.is_complex:
            parts.append("Признаки: сложный запрос")
        if features.is_multimodal:
            parts.append("Признаки: мультимодальный запрос")
        
        return "\n".join(parts)
    
    def _call_llm(self, prompt: str) -> str:
        """Вызов LLM для роутинга"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.router_model,
                messages=[
                    {
                        "role": "system",
                        "content": "Ты — эксперт по маршрутизации запросов к AI-инструментам. Отвечай строго в формате JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Низкая температура для консистентности
                max_tokens=200
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error calling LLM router: {e}")
            raise
    
    def _parse_llm_response(
        self,
        response: str,
        features: RequestFeatures
    ) -> RouteDecision:
        """Парсинг ответа LLM в RouteDecision"""
        
        try:
            # Извлекаем JSON из ответа
            # Иногда LLM добавляет текст до/после JSON
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response[json_start:json_end]
            data = json.loads(json_str)
            
            # Извлекаем поля
            tool = data.get('tool', 'text_llm')
            confidence = float(data.get('confidence', 0.7))
            reasoning = data.get('reasoning', 'LLM routing decision')
            
            # Маппинг tool → ModelType
            tool_mapping = {
                'vision': ModelType.VISION,
                'text_llm': ModelType.TEXT_LLM,
                'image_gen': ModelType.IMAGE_GEN,
                'web_search': ModelType.WEB_SEARCH,
                'web_parser': ModelType.WEB_PARSER,
                'research': ModelType.RESEARCH
            }
            
            model_type = tool_mapping.get(tool, ModelType.TEXT_LLM)
            
            # Получаем конфигурацию модели
            model_name = self._get_model_name_for_type(model_type)
            
            return RouteDecision(
                model_type=model_type,
                model_name=model_name,
                confidence=confidence,
                reasoning=reasoning,
                rule_matched=None  # LLM, а не правило
            )
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}. Response: {response}")
            return self._get_fallback_decision(features)
    
    def _get_model_name_for_type(self, model_type: ModelType) -> str:
        """Получить название модели для типа"""
        
        type_to_model = {
            ModelType.VISION: "gpt-4o-vision",
            ModelType.TEXT_LLM: "gpt-4o",
            ModelType.IMAGE_GEN: "dall-e-3",
            ModelType.WEB_SEARCH: "web-search",
            ModelType.WEB_PARSER: "web-parser",
            ModelType.RESEARCH: "research-pipeline"
        }
        
        return type_to_model.get(model_type, "gpt-4o")
    
    def _get_fallback_decision(self, features: RequestFeatures) -> RouteDecision:
        """Запасное решение при ошибке LLM"""
        
        # Простая эвристика: если есть изображение → vision, иначе text_llm
        if features.has_image:
            model_type = ModelType.VISION
            model_name = "gpt-4o-vision"
            reasoning = "Fallback: обнаружено изображение"
        else:
            model_type = ModelType.TEXT_LLM
            model_name = "gpt-4o"
            reasoning = "Fallback: стандартный текстовый запрос"
        
        return RouteDecision(
            model_type=model_type,
            model_name=model_name,
            confidence=0.5,
            reasoning=reasoning,
            rule_matched="fallback"
        )