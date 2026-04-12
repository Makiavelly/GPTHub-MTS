"""
Анализатор входящих запросов
Извлекает признаки для принятия решения о маршрутизации
"""

import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from .models_config import ModelType, INTENT_KEYWORDS

@dataclass
class RequestFeatures:
    """Признаки запроса для маршрутизации"""
    
    # Входные данные
    text: str
    has_image: bool = False
    has_audio: bool = False
    has_document: bool = False
    has_url: bool = False
    
    # Метаданные вложений
    image_count: int = 0
    audio_count: int = 0
    document_count: int = 0
    urls: List[str] = None
    
    # Анализ текста
    detected_intent: Optional[ModelType] = None
    keywords_matched: List[str] = None
    
    # Сложность
    is_complex: bool = False
    is_multimodal: bool = False
    
    # Контекст
    has_previous_context: bool = False
    conversation_length: int = 0
    
    def __post_init__(self):
        if self.urls is None:
            self.urls = []
        if self.keywords_matched is None:
            self.keywords_matched = []

class RequestAnalyzer:
    """Анализатор запросов пользователя"""
    
    def __init__(self):
        self.intent_keywords = INTENT_KEYWORDS
    
    def analyze(
        self,
        text: str,
        files: List[Dict[str, Any]] = None,
        context: Dict[str, Any] = None
    ) -> RequestFeatures:
        """
        Полный анализ запроса
        
        Args:
            text: Текст запроса пользователя
            files: Список прикрепленных файлов
            context: Контекст диалога
            
        Returns:
            RequestFeatures с извлеченными признаками
        """
        features = RequestFeatures(text=text)
        
        # Анализ вложений
        if files:
            features = self._analyze_files(files, features)
        
        # Анализ текста
        features = self._analyze_text(text, features)
        
        # Анализ контекста
        if context:
            features = self._analyze_context(context, features)
        
        # Определение сложности
        features.is_complex = self._is_complex_query(features)
        features.is_multimodal = self._is_multimodal(features)
        
        return features
    
    def _analyze_files(
        self,
        files: List[Dict[str, Any]],
        features: RequestFeatures
    ) -> RequestFeatures:
        """Анализ прикрепленных файлов"""
        
        for file in files:
            file_type = file.get('type', '').lower()
            mime_type = file.get('mime_type', '').lower()
            
            # Изображения
            if any(x in file_type for x in ['image', 'png', 'jpg', 'jpeg', 'gif', 'webp']):
                features.has_image = True
                features.image_count += 1
            
            # Аудио
            elif any(x in file_type for x in ['audio', 'mp3', 'wav', 'ogg', 'm4a']):
                features.has_audio = True
                features.audio_count += 1
            
            # Документы
            elif any(x in file_type for x in ['pdf', 'doc', 'docx', 'txt', 'csv', 'xlsx']):
                features.has_document = True
                features.document_count += 1
        
        return features
    
    def _analyze_text(
        self,
        text: str,
        features: RequestFeatures
    ) -> RequestFeatures:
        """Анализ текста запроса"""
        
        text_lower = text.lower()
        
        # Поиск URL
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, text)
        if urls:
            features.has_url = True
            features.urls = urls
        
        # Определение намерения по ключевым словам
        for intent_type, keywords in self.intent_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    features.detected_intent = intent_type
                    features.keywords_matched.append(keyword)
                    break
            if features.detected_intent:
                break
        
        return features
    
    def _analyze_context(
        self,
        context: Dict[str, Any],
        features: RequestFeatures
    ) -> RequestFeatures:
        """Анализ контекста диалога"""
        
        messages = context.get('messages', [])
        features.conversation_length = len(messages)
        features.has_previous_context = len(messages) > 0
        
        return features
    
    def _is_complex_query(self, features: RequestFeatures) -> bool:
        """Определение сложности запроса"""
        
        complexity_indicators = [
            len(features.text) > 500,  # Длинный текст
            features.text.count('?') > 2,  # Много вопросов
            features.text.count('.') > 5,  # Много предложений
            features.image_count > 1,  # Несколько изображений
            features.document_count > 0,  # Есть документы
            features.detected_intent == ModelType.RESEARCH,  # Исследовательский запрос
            any(word in features.text.lower() for word in [
                'проанализируй', 'сравни', 'исследуй', 'подробно',
                'со всех сторон', 'глубокий анализ'
            ])
        ]
        
        return sum(complexity_indicators) >= 2
    
    def _is_multimodal(self, features: RequestFeatures) -> bool:
        """Проверка на мультимодальность"""
        
        modalities = sum([
            features.has_image,
            features.has_audio,
            features.has_document,
            bool(features.text)
        ])
        
        return modalities > 1
    
    def get_analysis_summary(self, features: RequestFeatures) -> str:
        """Получить текстовое описание анализа (для логов/дебага)"""
        
        summary_parts = [
            f"Text length: {len(features.text)} chars",
        ]
        
        if features.has_image:
            summary_parts.append(f"Images: {features.image_count}")
        if features.has_audio:
            summary_parts.append(f"Audio: {features.audio_count}")
        if features.has_document:
            summary_parts.append(f"Documents: {features.document_count}")
        if features.has_url:
            summary_parts.append(f"URLs: {len(features.urls)}")
        
        if features.detected_intent:
            summary_parts.append(f"Intent: {features.detected_intent.value}")
            summary_parts.append(f"Keywords: {', '.join(features.keywords_matched[:3])}")
        
        summary_parts.append(f"Complex: {features.is_complex}")
        summary_parts.append(f"Multimodal: {features.is_multimodal}")
        
        return " | ".join(summary_parts)