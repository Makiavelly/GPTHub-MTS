"""
Новый эндпоинт для чатов с Smart Router
Работает параллельно с оригинальным chats.py
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import json

# Импорты OpenWebUI (используем существующие)
from apps.webui.models.users import Users
from utils.utils import get_current_user, get_admin_user

# Импорты нашего роутера
from .smart_router.hybrid_router import HybridRouter
from .smart_router.utils import (
    RouterLogger,
    PromptBuilder,
    ResponseFormatter,
    generate_request_id,
    validate_routing_decision,
    sanitize_user_input
)

router = APIRouter()
logger = logging.getLogger(__name__)

# Инициализация компонентов
router_logger = RouterLogger()
prompt_builder = PromptBuilder()
response_formatter = ResponseFormatter()

# Глобальный инстанс роутера
smart_router: Optional[HybridRouter] = None


def init_smart_router(mws_client):
    """Инициализация Smart Router"""
    global smart_router
    smart_router = HybridRouter(mws_client, enable_llm_fallback=True)
    logger.info("Smart Router initialized successfully")


# ===== Pydantic модели =====

class ChatMessage(BaseModel):
    role: str
    content: str
    images: Optional[List[str]] = None


class SmartChatRequest(BaseModel):
    """Запрос к Smart Chat (расширенная версия)"""
    messages: List[ChatMessage]
    model: Optional[str] = None  # Ручной выбор
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 4096
    stream: Optional[bool] = True
    
    # Расширенные параметры
    files: Optional[List[Dict[str, Any]]] = None
    enable_smart_routing: Optional[bool] = True
    user_preferences: Optional[Dict[str, Any]] = None


# ===== ОСНОВНОЙ ЭНДПОИНТ =====

@router.post("/v1/smart-chat/completions")
async def smart_chat_completion(
    request: SmartChatRequest,
    user=Depends(get_current_user)
):
    """
    Умный чат с автоматическим роутингом
    Эндпоинт: POST /api/v1/smart-chat/completions
    """
    
    if not smart_router:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Smart Router not initialized"
        )
    
    try:
        request_id = generate_request_id()
        
        if not request.messages:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No messages provided"
            )
        
        last_message = request.messages[-1]
        user_text = sanitize_user_input(last_message.content)
        
        # Контекст диалога
        context = {
            'messages': [msg.dict() for msg in request.messages[:-1]],
            'user_id': user.id
        }
        
        # Определяем маршрут
        if request.enable_smart_routing and not request.model:
            routing_decision = smart_router.route(
                text=user_text,
                files=request.files,
                context=context,
                manual_model=None,
                user_preferences=request.user_preferences
            )
            
            if not validate_routing_decision(routing_decision):
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Invalid routing decision"
                )
            
            selected_model = routing_decision['model_name']
            logger.info(f"Smart routing: {selected_model} via {routing_decision['routing_method']}")
        
        else:
            selected_model = request.model or "gpt-4o"
            routing_decision = {
                'model_name': selected_model,
                'routing_method': 'manual',
                'confidence': 1.0,
                'reasoning': 'User-selected model'
            }
        
        # Логируем решение
        router_logger.log_decision(request_id, routing_decision)
        
        # Здесь вызов к вашей существующей логике OpenWebUI
        # Используем существующий механизм вызова моделей
        response_data = await call_existing_model_handler(
            model_name=selected_model,
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=request.stream,
            user=user
        )
        
        # Добавляем информацию о роутинге в ответ
        if isinstance(response_data, dict):
            response_data['routing_info'] = routing_decision
        
        return response_data
    
    except Exception as e:
        logger.error(f"Smart chat error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


async def call_existing_model_handler(
    model_name: str,
    messages: List[ChatMessage],
    temperature: float,
    max_tokens: int,
    stream: bool,
    user
):
    """
    Вызов СУЩЕСТВУЮЩЕГО обработчика моделей из OpenWebUI
    
    Находим и используем то, что уже есть в chats.py
    """
    
    # ВАРИАНТ 1: Импортируем существующую функцию
    try:
        from apps.webui.routers.chats import generate_chat_completions
        
        # Формируем запрос в формате существующего API
        existing_request = {
            "model": model_name,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        # Вызываем существующий обработчик
        return await generate_chat_completions(existing_request, user)
    
    except ImportError:
        # ВАРИАНТ 2: Прямой вызов к API (если функция не экспортируется)
        logger.warning("Using direct API call fallback")
        
        # Здесь делаем то же, что делает оригинальный chats.py
        # Скопируйте логику из существующего файла
        pass


# ===== ВСПОМОГАТЕЛЬНЫЕ ЭНДПОИНТЫ =====

@router.get("/v1/smart-chat/stats")
async def get_smart_router_stats(user=Depends(get_current_user)):
    """Статистика роутера"""
    
    if not smart_router:
        return {"error": "Smart Router not initialized"}
    
    return {
        'router_stats': smart_router.get_stats(),
        'performance': router_logger.analyze_performance()
    }


@router.get("/v1/smart-chat/models")
async def get_available_models_with_routing(user=Depends(get_current_user)):
    """Список моделей с информацией о роутинге"""
    
    from .smart_router.models_config import AVAILABLE_MODELS
    
    models = []
    for name, config in AVAILABLE_MODELS.items():
        models.append({
            'name': name,
            'type': config.model_type.value,
            'description': config.description,
            'supports_streaming': config.supports_streaming,
            'max_tokens': config.max_tokens
        })
    
    return {'models': models}


@router.post("/v1/smart-chat/feedback")
async def submit_routing_feedback(
    request_id: str,
    feedback: str,
    rating: Optional[int] = None,
    user=Depends(get_current_user)
):
    """Фидбек о качестве роутинга"""
    
    router_logger.log_decision(
        request_id, 
        {}, 
        user_feedback=f"{feedback} (rating: {rating})"
    )
    
    return {"status": "ok"}


@router.get("/v1/smart-chat/recent-decisions")
async def get_recent_routing_decisions(
    limit: int = 50,
    user=Depends(get_current_user)
):
    """История решений роутера"""
    
    decisions = router_logger.get_recent_decisions(limit=limit)
    return {'decisions': decisions}