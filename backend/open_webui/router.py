from typing import Dict, List, Optional
import json
import re
from open_webui.config import MWS_API_KEY, MWS_API_URL
from open_webui.utils.mws_client import MWSClient

class TaskRouter:
    def init(self):
        self.client = MWSClient(api_key=MWS_API_KEY, base_url=MWS_API_URL)
        self.model_map = {
            "text": "gpt-4o",                # обычный текст
            "image_analysis": "gpt-4o",      # VLM для изображений
            "image_generation": "dall-e-3",  # генерация картинок
            "audio_transcription": "whisper-1",
            "text_to_speech": "tts-1",
            "web_search": "gpt-4o" + function_calling,
            "document_qa": "gpt-4o" + rag
        }
    
    async def classify_task(self, message: str, files: List[dict], history: List[dict]) -> str:
        """Используем небольшую LLM для определения типа задачи"""
        prompt = f"""
        Определи тип задачи по сообщению и вложениям.
        Варианты: text, image_analysis, image_generation, audio_transcription, web_search, document_qa.
        Сообщение: {message}
        Типы файлов: {[f['type'] for f in files]}
        Ответь только одним словом.
        """
        response = await self.client.chat_completion(
            model="gpt-3.5-turbo",  # дешёвая модель для классификации
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip().lower()
    
    async def route(self, request: dict) -> dict:
        # Если ручной режим – просто используем выбранную пользователем модель
        if request.get("manual_mode"):
            return {"model": request["selected_model"], "tools": []}
        
        task_type = await self.classify_task(
            request["message"], 
            request.get("files", []),
            request.get("history", [])
        )
        
        if task_type == "image_generation":
            return {"model": self.model_map["image_generation"], "tools": ["generate_image"]}
        elif task_type == "audio_transcription":
            return {"model": self.model_map["audio_transcription"], "tools": ["transcribe"]}
        elif task_type == "web_search":
            return {"model": self.model_map["web_search"], "tools": ["search_web"]}
        # ... остальные случаи
        else:
            return {"model": self.model_map["text"], "tools": []}