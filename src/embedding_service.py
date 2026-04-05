# src/embedding_service.py

import asyncio
import logging
import struct
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract base for embedding providers."""

    @abstractmethod
    async def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts into embedding vectors (normalized)."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        ...

    @property
    def max_input_tokens(self) -> Optional[int]:
        """Max input token limit for the provider. None means no limit."""
        return None


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Uses Google's gemini-embedding-001 via the google-genai SDK."""

    _model_name = "gemini-embedding-001"
    _dimensions = 768
    _max_input_tokens = 2048

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def max_input_tokens(self) -> Optional[int]:
        return self._max_input_tokens

    async def encode(self, texts: List[str]) -> List[List[float]]:
        import os
        from google import genai

        client = genai.Client(
            api_key=os.environ.get("GOOGLE_GENERATIVEAI_API_KEY")
        )

        result = await asyncio.to_thread(
            client.models.embed_content,
            model=self._model_name,
            contents=texts,
        )

        vectors = []
        for embedding in result.embeddings or []:
            vec = list(embedding.values or [])
            # Normalize to unit vector
            norm = sum(v * v for v in vec) ** 0.5
            if norm > 0:
                vec = [v / norm for v in vec]
            vectors.append(vec)

        return vectors


class EmbeddingService:
    """High-level service: provider + similarity math + BLOB serialization."""

    def __init__(self, provider: Optional[EmbeddingProvider] = None) -> None:
        self._provider = provider or GeminiEmbeddingProvider()

    @property
    def model_name(self) -> str:
        return self._provider.model_name

    @property
    def dimensions(self) -> int:
        return self._provider.dimensions

    def _truncate_texts(self, texts: List[str]) -> List[str]:
        """Truncate texts exceeding the provider's max input token limit.

        Uses a rough char-based estimate (4 chars per token) since we don't
        have a tokenizer. Conservative — slightly over-truncates rather than
        risking API errors.
        """
        max_tokens = self._provider.max_input_tokens
        if max_tokens is None:
            return texts
        # Rough estimate: 4 chars per token
        max_chars = max_tokens * 4
        result = []
        for text in texts:
            if len(text) > max_chars:
                result.append(text[:max_chars])
            else:
                result.append(text)
        return result

    async def encode(self, texts: List[str]) -> List[bytes]:
        """Encode texts into float32 BLOBs."""
        truncated = self._truncate_texts(texts)
        vectors = await self._provider.encode(truncated)
        return [self._vector_to_blob(v) for v in vectors]

    async def encode_single(self, text: str) -> bytes:
        """Encode a single text into a float32 BLOB."""
        results = await self.encode([text])
        return results[0]

    @staticmethod
    def _vector_to_blob(vector: List[float]) -> bytes:
        """Convert a float vector to raw float32 bytes."""
        return struct.pack(f'{len(vector)}f', *vector)

    @staticmethod
    def _blob_to_vector(blob: bytes) -> np.ndarray:
        """Convert raw float32 bytes to a numpy array."""
        return np.frombuffer(blob, dtype=np.float32)

    @staticmethod
    def cosine_similarity(blob_a: bytes, blob_b: bytes) -> float:
        """Cosine similarity between two BLOB embeddings.

        Vectors are pre-normalized, so this is just a dot product.
        """
        a = np.frombuffer(blob_a, dtype=np.float32)
        b = np.frombuffer(blob_b, dtype=np.float32)
        return float(np.dot(a, b))

    @staticmethod
    def cosine_similarities(query_blob: bytes, candidate_blobs: List[bytes]) -> List[float]:
        """Cosine similarity of a query against multiple candidates.

        Vectors are pre-normalized, so this is just dot products.
        """
        query = np.frombuffer(query_blob, dtype=np.float32)
        scores = []
        for blob in candidate_blobs:
            candidate = np.frombuffer(blob, dtype=np.float32)
            scores.append(float(np.dot(query, candidate)))
        return scores
