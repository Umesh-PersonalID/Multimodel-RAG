# Copyright 2024
# Directory: yt-rag/app/services/multimodal.py

"""
Multimodal document ingestion pipeline.

This service saves uploaded files, extracts text and images from PDFs and DOCX
documents, performs OCR when needed, and creates chunk-ready multimodal records
that can be embedded and stored in Supabase.
"""

from __future__ import annotations

import base64
import io
import logging
import mimetypes
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from fastapi import UploadFile

from ..core.config import get_settings
from ..core.database import db
from ..models.multimodal import ImageReference, SourceReference
from .chunker import chunker
from .embedding import embedding_service

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class ExtractionResult:
    """Internal extraction result for a document."""

    records: List[Dict[str, Any]]
    pages_processed: int
    images_extracted: int
    ocr_applied: bool
    file_type: str


class OCRService:
    """Optional OCR wrapper with graceful fallback when OCR tooling is unavailable."""

    def extract_text(self, image_bytes: bytes) -> str:
        try:
            import pytesseract
            from PIL import Image

            image = Image.open(io.BytesIO(image_bytes))
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as exc:
            logger.debug("OCR unavailable or failed: %s", exc)
            return ""


class VisionCaptionService:
    """Generate concise image captions using the configured chat provider."""

    def __init__(self) -> None:
        from ..core.config import get_settings

        self.settings = get_settings()
        self.provider = self.settings.ai_provider

        if self.provider == "openai":
            import openai

            self.client = openai.OpenAI(api_key=self.settings.openai_api_key)
            self.model = self.settings.openai_chat_model
        elif self.provider == "anthropic":
            import anthropic

            self.client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
            self.model = self.settings.anthropic_chat_model
        else:
            self.client = None
            self.model = ""

    async def caption_image(
        self,
        image_bytes: bytes,
        mime_type: str,
        filename: str,
        page_number: Optional[int] = None,
        image_index: Optional[int] = None,
        ocr_text: str = "",
    ) -> str:
        """Create a caption for an image using the configured model."""

        if not self.client:
            return self._fallback_caption(filename, page_number, image_index, ocr_text)

        encoded_image = base64.b64encode(image_bytes).decode("ascii")
        page_suffix = f" on page {page_number}" if page_number else ""
        image_suffix = f" image {image_index}" if image_index else ""
        prompt = (
            "Write a concise search-friendly caption for this document image. "
            "Describe diagrams, charts, tables, UI screenshots, and labels if present. "
            "Return only the caption text."
        )
        if ocr_text:
            prompt += f" OCR text extracted from the image: {ocr_text[:500]}"

        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You caption document images for multimodal retrieval."
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt + page_suffix + image_suffix},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime_type};base64,{encoded_image}",
                                        "detail": "high",
                                    },
                                },
                            ],
                        },
                    ],
                    temperature=0,
                    max_tokens=200,
                )
                caption = response.choices[0].message.content or ""
                return caption.strip() or self._fallback_caption(filename, page_number, image_index, ocr_text)

            if self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=200,
                    temperature=0,
                    system="You caption document images for multimodal retrieval.",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt + page_suffix + image_suffix},
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": mime_type,
                                        "data": encoded_image,
                                    },
                                },
                            ],
                        }
                    ],
                )
                caption = response.content[0].text if response.content else ""
                return caption.strip() or self._fallback_caption(filename, page_number, image_index, ocr_text)

        except Exception as exc:
            logger.warning("Vision captioning failed, using fallback: %s", exc)

        return self._fallback_caption(filename, page_number, image_index, ocr_text)

    def _fallback_caption(
        self,
        filename: str,
        page_number: Optional[int] = None,
        image_index: Optional[int] = None,
        ocr_text: str = "",
    ) -> str:
        page_suffix = f" page {page_number}" if page_number else ""
        image_suffix = f" image {image_index}" if image_index else ""
        caption = f"Document image from {filename}{page_suffix}{image_suffix}."
        if ocr_text:
            caption += f" OCR text: {ocr_text[:240]}"
        return caption.strip()


class MultimodalIngestionService:
    """End-to-end multimodal document ingestion service."""

    def __init__(self) -> None:
        self.settings = settings
        self.upload_root = Path(self.settings.uploaded_docs_dir)
        self.image_root = Path(self.settings.extracted_images_dir)
        self.upload_root.mkdir(parents=True, exist_ok=True)
        self.image_root.mkdir(parents=True, exist_ok=True)
        self.ocr_service = OCRService()
        self.vision_service = VisionCaptionService()
        self._vision_captions_used = 0
        self._images_stored = 0

    def _reset_ingestion_budget(self) -> None:
        self._vision_captions_used = 0
        self._images_stored = 0

    def _image_large_enough(self, width: Optional[int], height: Optional[int]) -> bool:
        min_dim = self.settings.upload_min_image_dimension
        if not width or not height:
            return True
        return width >= min_dim and height >= min_dim

    def _can_store_more_images(self) -> bool:
        return self._images_stored < self.settings.upload_max_images_per_document

    def _can_use_vision_caption(self) -> bool:
        if not self.settings.upload_vision_caption_enabled:
            return False
        return self._vision_captions_used < self.settings.upload_max_vision_captions

    async def _caption_image(
        self,
        image_bytes: bytes,
        mime_type: str,
        filename: str,
        page_number: Optional[int] = None,
        image_index: Optional[int] = None,
        ocr_text: str = "",
    ) -> str:
        if self._can_use_vision_caption():
            self._vision_captions_used += 1
            return await self.vision_service.caption_image(
                image_bytes=image_bytes,
                mime_type=mime_type,
                filename=filename,
                page_number=page_number,
                image_index=image_index,
                ocr_text=ocr_text,
            )
        return self.vision_service._fallback_caption(filename, page_number, image_index, ocr_text)

    async def ingest_upload(self, upload: UploadFile) -> Dict[str, Any]:
        """Persist an upload, extract multimodal records, and store them in Supabase."""

        self._reset_ingestion_budget()

        file_bytes = await upload.read()
        filename = upload.filename or "upload.bin"
        file_type = self._detect_file_type(filename, upload.content_type)
        document_id = uuid4().hex
        saved_path = self._save_upload(document_id, filename, file_bytes)

        extraction = await self._extract_records(
            document_id=document_id,
            filename=filename,
            file_type=file_type,
            saved_path=saved_path,
            file_bytes=file_bytes,
            content_type=upload.content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream",
        )

        chunks = chunker.chunk_documents(extraction.records)
        if chunks:
            embeddings = await embedding_service.embed_texts([chunk["text"] for chunk in chunks])
            for chunk, embedding in zip(chunks, embeddings):
                chunk["embedding"] = embedding

        inserted = await db.upsert_chunks(chunks)
        sources = [self._chunk_to_source_reference(chunk) for chunk in chunks]

        return {
            "document_id": document_id,
            "filename": filename,
            "file_type": file_type,
            "chunks_inserted": inserted,
            "pages_processed": extraction.pages_processed,
            "images_extracted": extraction.images_extracted,
            "ocr_applied": extraction.ocr_applied,
            "sources": sources,
        }

    async def _extract_records(
        self,
        document_id: str,
        filename: str,
        file_type: str,
        saved_path: Path,
        file_bytes: bytes,
        content_type: str,
    ) -> ExtractionResult:
        if file_type == "pdf":
            return await self._extract_pdf(document_id, filename, saved_path, file_bytes, content_type)
        if file_type == "image":
            return await self._extract_image(document_id, filename, saved_path, file_bytes, content_type)
        if file_type == "docx":
            return await self._extract_docx(document_id, filename, saved_path, file_bytes, content_type)
        return await self._extract_plain_text(document_id, filename, saved_path, file_bytes, content_type)

    async def _extract_pdf(
        self,
        document_id: str,
        filename: str,
        saved_path: Path,
        file_bytes: bytes,
        content_type: str,
    ) -> ExtractionResult:
        try:
            import fitz
        except Exception as exc:
            raise RuntimeError("PyMuPDF is required for PDF processing") from exc

        records: List[Dict[str, Any]] = []
        pages_processed = 0
        images_extracted = 0
        ocr_applied = False

        document = fitz.open(stream=file_bytes, filetype="pdf")
        for page_index in range(document.page_count):
            page = document[page_index]
            page_number = page_index + 1
            pages_processed += 1

            page_text = (page.get_text("text") or "").strip()
            metadata = {
                "filename": filename,
                "content_type": content_type,
                "saved_path": str(saved_path),
                "page_number": page_number,
            }

            if page_text:
                records.append({
                    "document_id": document_id,
                    "chunk_id": f"{document_id}_p{page_number}_text",
                    "source": filename,
                    "page_number": page_number,
                    "item_type": "text",
                    "text": page_text,
                    "metadata": metadata,
                })

            page_image_bytes = None
            if not page_text:
                try:
                    pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                    page_image_bytes = pixmap.tobytes("png")
                    ocr_text = self.ocr_service.extract_text(page_image_bytes)
                    ocr_applied = True
                    if ocr_text:
                        records.append({
                            "document_id": document_id,
                            "chunk_id": f"{document_id}_p{page_number}_ocr",
                            "source": filename,
                            "page_number": page_number,
                            "item_type": "ocr",
                            "text": ocr_text,
                            "ocr_text": ocr_text,
                            "metadata": metadata,
                        })
                except Exception as exc:
                    logger.debug("Page OCR failed for %s page %s: %s", filename, page_number, exc)

            images = page.get_images(full=True)
            for image_index, image_info in enumerate(images, start=1):
                xref = image_info[0]
                extracted = document.extract_image(xref)
                image_bytes = extracted.get("image")
                if not image_bytes:
                    continue

                image_ext = extracted.get("ext", "png")
                image_path = self._save_extracted_image(document_id, filename, page_number, image_index, image_bytes, image_ext)
                images_extracted += 1
                image_ocr = self.ocr_service.extract_text(image_bytes)
                if image_ocr:
                    ocr_applied = True

                caption = await self.vision_service.caption_image(
                    image_bytes=image_bytes,
                    mime_type=mimetypes.types_map.get(f".{image_ext}", "image/png"),
                    filename=filename,
                    page_number=page_number,
                    image_index=image_index,
                    ocr_text=image_ocr,
                )

                text_parts = [caption]
                if image_ocr:
                    text_parts.append(f"OCR: {image_ocr}")

                records.append({
                    "document_id": document_id,
                    "chunk_id": f"{document_id}_p{page_number}_img{image_index}",
                    "source": filename,
                    "page_number": page_number,
                    "item_type": "image",
                    "text": "\n\n".join(text_parts),
                    "image_path": str(image_path),
                    "caption": caption,
                    "ocr_text": image_ocr or None,
                    "width": extracted.get("width"),
                    "height": extracted.get("height"),
                    "metadata": {
                        **metadata,
                        "image_xref": xref,
                        "image_extension": image_ext,
                    },
                })

        return ExtractionResult(records, pages_processed, images_extracted, ocr_applied, "pdf")

    async def _extract_image(
        self,
        document_id: str,
        filename: str,
        saved_path: Path,
        file_bytes: bytes,
        content_type: str,
    ) -> ExtractionResult:
        from PIL import Image

        image = Image.open(io.BytesIO(file_bytes))
        width, height = image.size
        mime_type = content_type or mimetypes.guess_type(filename)[0] or "image/png"
        ocr_text = self.ocr_service.extract_text(file_bytes)
        caption = await self.vision_service.caption_image(
            image_bytes=file_bytes,
            mime_type=mime_type,
            filename=filename,
            ocr_text=ocr_text,
        )

        image_path = self._save_extracted_image(document_id, filename, None, 1, file_bytes, Path(filename).suffix.lstrip(".") or "png")
        record = {
            "document_id": document_id,
            "chunk_id": f"{document_id}_image_1",
            "source": filename,
            "page_number": None,
            "item_type": "image",
            "text": "\n\n".join(part for part in [caption, f"OCR: {ocr_text}" if ocr_text else ""] if part),
            "image_path": str(image_path),
            "caption": caption,
            "ocr_text": ocr_text or None,
            "width": width,
            "height": height,
            "metadata": {
                "filename": filename,
                "content_type": mime_type,
                "saved_path": str(saved_path),
            },
        }
        return ExtractionResult([record], 1, 1, bool(ocr_text), "image")

    async def _extract_docx(
        self,
        document_id: str,
        filename: str,
        saved_path: Path,
        file_bytes: bytes,
        content_type: str,
    ) -> ExtractionResult:
        try:
            from docx import Document
        except Exception as exc:
            raise RuntimeError("python-docx is required for DOCX processing") from exc

        records: List[Dict[str, Any]] = []
        paragraphs: List[str] = []
        document = Document(io.BytesIO(file_bytes))
        for paragraph in document.paragraphs:
            text = paragraph.text.strip()
            if text:
                paragraphs.append(text)

        combined_text = "\n".join(paragraphs).strip()
        if combined_text:
            records.append({
                "document_id": document_id,
                "chunk_id": f"{document_id}_docx_text",
                "source": filename,
                "page_number": None,
                "item_type": "text",
                "text": combined_text,
                "metadata": {
                    "filename": filename,
                    "content_type": content_type,
                    "saved_path": str(saved_path),
                },
            })

        images_extracted = 0
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as archive:
            for member in archive.namelist():
                if not member.startswith("word/media/"):
                    continue

                image_bytes = archive.read(member)
                image_name = Path(member).name
                image_ext = Path(image_name).suffix.lstrip(".") or "png"
                images_extracted += 1
                image_path = self._save_extracted_image(document_id, filename, None, images_extracted, image_bytes, image_ext)
                image_ocr = self.ocr_service.extract_text(image_bytes)
                caption = await self.vision_service.caption_image(
                    image_bytes=image_bytes,
                    mime_type=mimetypes.types_map.get(f".{image_ext}", "image/png"),
                    filename=filename,
                    ocr_text=image_ocr,
                )

                records.append({
                    "document_id": document_id,
                    "chunk_id": f"{document_id}_docx_img{images_extracted}",
                    "source": filename,
                    "page_number": None,
                    "item_type": "image",
                    "text": "\n\n".join(part for part in [caption, f"OCR: {image_ocr}" if image_ocr else ""] if part),
                    "image_path": str(image_path),
                    "caption": caption,
                    "ocr_text": image_ocr or None,
                    "metadata": {
                        "filename": filename,
                        "content_type": content_type,
                        "saved_path": str(saved_path),
                        "zip_member": member,
                    },
                })

        return ExtractionResult(records, 0, images_extracted, bool(records), "docx")

    async def _extract_plain_text(
        self,
        document_id: str,
        filename: str,
        saved_path: Path,
        file_bytes: bytes,
        content_type: str,
    ) -> ExtractionResult:
        text = file_bytes.decode("utf-8", errors="ignore").strip()
        records: List[Dict[str, Any]] = []
        if text:
            records.append({
                "document_id": document_id,
                "chunk_id": f"{document_id}_text",
                "source": filename,
                "page_number": None,
                "item_type": "text",
                "text": text,
                "metadata": {
                    "filename": filename,
                    "content_type": content_type,
                    "saved_path": str(saved_path),
                },
            })

        return ExtractionResult(records, 0, 0, False, "text")

    def _detect_file_type(self, filename: str, content_type: Optional[str]) -> str:
        suffix = Path(filename).suffix.lower()
        content_type = (content_type or "").lower()

        if suffix == ".pdf" or content_type == "application/pdf":
            return "pdf"
        if suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"} or content_type.startswith("image/"):
            return "image"
        if suffix == ".docx" or content_type in {
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword",
        }:
            return "docx"
        return "text"

    def _save_upload(self, document_id: str, filename: str, file_bytes: bytes) -> Path:
        safe_name = Path(filename).name
        destination = self.upload_root / document_id
        destination.mkdir(parents=True, exist_ok=True)
        saved_path = destination / safe_name
        saved_path.write_bytes(file_bytes)
        return saved_path

    def _save_extracted_image(
        self,
        document_id: str,
        filename: str,
        page_number: Optional[int],
        image_index: int,
        image_bytes: bytes,
        image_ext: str,
    ) -> Path:
        destination = self.image_root / document_id
        if page_number is not None:
            destination = destination / f"page_{page_number}"
        destination.mkdir(parents=True, exist_ok=True)

        image_name = f"image_{image_index}.{image_ext.lstrip('.') or 'png'}"
        saved_path = destination / image_name
        saved_path.write_bytes(image_bytes)
        return saved_path

    def _chunk_to_source_reference(self, chunk: Dict[str, Any]) -> SourceReference:
        return SourceReference(
            chunk_id=chunk["chunk_id"],
            document_id=chunk.get("document_id", chunk["chunk_id"].split("#")[0]),
            source=chunk.get("source", "unknown"),
            item_type=chunk.get("item_type", "text"),
            page_number=chunk.get("page_number"),
            image_path=chunk.get("image_path"),
            caption=chunk.get("caption") or chunk.get("ocr_text"),
        )


multimodal_ingestion_service = MultimodalIngestionService()