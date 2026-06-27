# Copyright 2024
# Directory: yt-rag/app/models/multimodal.py

"""
Multimodal ingestion and retrieval models.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class MultimodalDocument(BaseModel):
    """Unified document representation for text, OCR, and image-derived content."""

    document_id: str = Field(..., description="Stable identifier for the uploaded document")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    source: str = Field(..., description="Original filename, URL, or source label")
    text: str = Field(..., description="Chunk text used for embeddings")
    item_type: str = Field(..., description="Content type such as text, ocr, image, or caption")
    page_number: Optional[int] = Field(None, description="Page number when available")
    image_path: Optional[str] = Field(None, description="Stored image path when the chunk came from an image")
    caption: Optional[str] = Field(None, description="Vision caption for an image chunk")
    ocr_text: Optional[str] = Field(None, description="OCR extracted text")
    width: Optional[int] = Field(None, description="Image width in pixels")
    height: Optional[int] = Field(None, description="Image height in pixels")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata")


class ImageReference(BaseModel):
    """Image reference returned to the client."""

    image_path: str = Field(..., description="Stored image path")
    chunk_id: str = Field(..., description="Chunk associated with the image")
    page_number: Optional[int] = Field(None, description="Page number when available")
    caption: Optional[str] = Field(None, description="Caption or OCR summary")


class SourceReference(BaseModel):
    """Source reference exposed in answer metadata."""

    chunk_id: str = Field(..., description="Chunk identifier")
    document_id: str = Field(..., description="Document identifier")
    source: str = Field(..., description="Original source label")
    item_type: str = Field(..., description="Content type")
    page_number: Optional[int] = Field(None, description="Page number when available")
    image_path: Optional[str] = Field(None, description="Stored image path when available")
    caption: Optional[str] = Field(None, description="Caption or OCR summary")


class UploadResponse(BaseModel):
    """Response returned after a document upload."""

    document_id: str = Field(..., description="Assigned document identifier")
    filename: str = Field(..., description="Uploaded filename")
    file_type: str = Field(..., description="Detected file type")
    chunks_inserted: int = Field(..., description="Number of chunks stored")
    pages_processed: int = Field(..., description="Number of pages processed")
    images_extracted: int = Field(..., description="Number of images extracted")
    ocr_applied: bool = Field(..., description="Whether OCR was attempted")
    sources: List[SourceReference] = Field(default_factory=list, description="Stored chunk references")