# Copyright 2024
# Directory: yt-rag/app/services/chat.py

"""
Chat completion service for generating RAG responses.
Supports both OpenAI and Anthropic with configurable models.
"""

import base64
import logging
import mimetypes
from pathlib import Path
from typing import List, Dict, Any
import openai
import anthropic
from ..core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ChatService:
    """Service for chat completions supporting multiple AI providers."""
    
    def __init__(self):
        """Initialize chat clients based on configuration."""
        self.provider = settings.ai_provider
        
        if self.provider == "openai":
            self.client = openai.OpenAI(api_key=settings.openai_api_key)
            self.model = settings.openai_chat_model
        elif self.provider == "anthropic":
            self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            self.model = settings.anthropic_chat_model
        else:
            raise ValueError(f"Unsupported AI provider: {self.provider}")
        
        logger.info(f"Initialized chat service with {self.provider} ({self.model})")
    
    async def generate_answer(self, query: str, context_blocks: List[Dict[str, Any]]) -> str:
        """
        Generate RAG answer using context blocks.
        
        Args:
            query: User's question
            context_blocks: Retrieved chunks with metadata
            
        Returns:
            Generated answer with citations
        """
        # Build context string with citations and collect image payloads for vision models.
        context_parts = []
        image_payloads = []
        for block in context_blocks:
            chunk_id = block.get('chunk_id', 'unknown')
            document_id = block.get('document_id', 'unknown')
            item_type = block.get('item_type', 'text')
            page_number = block.get('page_number')
            text = block.get('text', '')
            caption = block.get('caption') or ''
            ocr_text = block.get('ocr_text') or ''

            header = [f"[{chunk_id}]", f"doc={document_id}", f"type={item_type}"]
            if page_number is not None:
                header.append(f"page={page_number}")
            context_parts.append(f"{' '.join(header)}\n{text}")

            image_path = block.get('image_path')
            if image_path:
                payload = self._load_image_payload(image_path)
                if payload:
                    payload['chunk_id'] = chunk_id
                    payload['caption'] = caption
                    payload['ocr_text'] = ocr_text
                    payload['page_number'] = page_number
                    image_payloads.append(payload)

        context = "\n\n".join(context_parts)

        system_prompt = """You are a helpful multimodal AI assistant for customer support.

IMPORTANT RULES:
1. Use only the provided context and any attached images.
2. For policy, returns, shipping, sizing, or support questions: answer from context and include citations.
3. When an attached image is relevant, inspect it directly and cite its chunk ID.
4. For general greetings or casual conversation, respond naturally.
5. For questions outside the knowledge base, politely redirect to relevant policies or support.
6. Always include citations in the form [chunk_id] when using retrieved information.
7. Be concise but comprehensive."""

        user_prompt = f"""Context:
{context}

Question: {query}

Answer using the context above. If an attached image is useful, reason over it directly and cite the corresponding chunk ID."""

        try:
            if self.provider == "openai":
                user_content = [{"type": "text", "text": user_prompt}]
                for image_payload in image_payloads[:2]:
                    user_content.append({
                        "type": "text",
                        "text": (
                            f"Attached image for [{image_payload['chunk_id']}]"
                            + (f" on page {image_payload['page_number']}" if image_payload.get('page_number') is not None else "")
                            + (f". Caption: {image_payload['caption']}" if image_payload.get('caption') else "")
                        ),
                    })
                    user_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": image_payload['data_uri'],
                            "detail": "high",
                        },
                    })

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=settings.temperature,
                    max_tokens=1000
                )
                answer = response.choices[0].message.content
                
            elif self.provider == "anthropic":
                user_content = [{"type": "text", "text": user_prompt}]
                for image_payload in image_payloads[:2]:
                    user_content.append({
                        "type": "text",
                        "text": (
                            f"Attached image for [{image_payload['chunk_id']}]"
                            + (f" on page {image_payload['page_number']}" if image_payload.get('page_number') is not None else "")
                            + (f". Caption: {image_payload['caption']}" if image_payload.get('caption') else "")
                        ),
                    })
                    user_content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image_payload['mime_type'],
                            "data": image_payload['b64'],
                        },
                    })

                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1000,
                    temperature=settings.temperature,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_content}]
                )
                answer = response.content[0].text
            
            logger.info(f"Generated answer using {self.provider}")
            return answer or "I couldn't generate an answer."
            
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            return f"I encountered an error while processing your question: {str(e)}"

    def _load_image_payload(self, image_path: str) -> Dict[str, str]:
        """Load an image and return provider-ready payload fields."""

        path = Path(image_path)
        if not path.exists():
            return {}

        mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return {
            'mime_type': mime_type,
            'b64': encoded,
            'data_uri': f"data:{mime_type};base64,{encoded}",
        }


# Global service instance
chat_service = ChatService()
