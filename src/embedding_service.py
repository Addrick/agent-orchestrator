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

    _MAX_BATCH_SIZE = 100  # Gemini API limit per batch request
    _MAX_RETRIES = 5
    # Free-tier quota: 100 items/min (each item in a batch counts as 1 request).
    # After a successful chunk we must wait for those slots to age out before
    # sending the next chunk, or the next call hits 429 immediately.
    _ITEMS_PER_MINUTE = 90  # stay just under the hard limit

    @staticmethod
    def _parse_retry_delay(exc: Exception) -> Optional[float]:
        """Extract the suggested retry delay (seconds) from a 429 error response.

        The Gemini API returns a RetryInfo detail with retryDelay like '24s'.
        Falls back to None if unparseable so the caller can use a default.
        """
        import re
        # exc.details is the raw response JSON dict; RetryInfo is in the
        # 'details' list inside the 'error' key.
        try:
            details = getattr(exc, "details", None) or {}
            details_list = details.get("error", {}).get("details", [])
            for item in details_list:
                delay_str = item.get("retryDelay", "")
                if delay_str:
                    match = re.match(r"([0-9.]+)s", delay_str)
                    if match:
                        return float(match.group(1))
        except Exception:
            pass
        # Fallback: parse the string representation
        match = re.search(r"'retryDelay':\s*'([0-9.]+)s'", str(exc))
        if match:
            return float(match.group(1))
        return None

    @staticmethod
    def _parse_quota_kind(exc: Exception) -> str:
        """Identify which quota was exceeded on a 429.

        Gemini's 429 carries a QuotaFailure detail listing violated quotaIds
        like 'GenerateContentInputTokensPerModelPerMinute' (TPM) or
        'GenerateRequestsPerModelPerMinute' (RPM). Returns a short label —
        'TPM', 'RPM', 'TPM+RPM', or 'unknown' — for log output.
        """
        quota_ids: List[str] = []
        try:
            details = getattr(exc, "details", None) or {}
            details_list = details.get("error", {}).get("details", [])
            for item in details_list:
                for violation in item.get("violations", []) or []:
                    qid = violation.get("quotaId", "")
                    if qid:
                        quota_ids.append(qid)
        except Exception:
            pass
        if not quota_ids:
            # Fallback: scan string representation
            import re
            quota_ids = re.findall(r"'quotaId':\s*'([^']+)'", str(exc))

        has_tokens = any("Token" in q for q in quota_ids)
        has_requests = any("Request" in q for q in quota_ids)
        if has_tokens and has_requests:
            return "TPM+RPM"
        if has_tokens:
            return "TPM"
        if has_requests:
            return "RPM"
        return "unknown"

    async def encode(self, texts: List[str]) -> List[List[float]]:
        import os
        from google import genai
        from google.genai.errors import ClientError

        api_key = os.environ.get("GOOGLE_GENERATIVEAI_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_GENERATIVEAI_API_KEY not set — skipping Gemini embedding provider.")
        client = genai.Client(api_key=api_key)

        vectors: List[List[float]] = []
        for i in range(0, len(texts), self._MAX_BATCH_SIZE):
            chunk = texts[i:i + self._MAX_BATCH_SIZE]

            for attempt in range(self._MAX_RETRIES):
                try:
                    result = await asyncio.to_thread(
                        client.models.embed_content,
                        model=self._model_name,
                        contents=chunk,
                    )
                    break
                except ClientError as exc:
                    if exc.code != 429 or attempt == self._MAX_RETRIES - 1:
                        raise
                    delay = self._parse_retry_delay(exc) or (30 * (attempt + 1))
                    quota_kind = self._parse_quota_kind(exc)
                    logger.warning(
                        f"GeminiEmbeddingProvider: 429 rate-limited ({quota_kind}) "
                        f"on chunk {i // self._MAX_BATCH_SIZE + 1} "
                        f"(attempt {attempt + 1}/{self._MAX_RETRIES}), "
                        f"retrying in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)

            for embedding in result.embeddings or []:
                vec = list(embedding.values or [])
                # Normalize to unit vector
                norm = sum(v * v for v in vec) ** 0.5
                if norm > 0:
                    vec = [v / norm for v in vec]
                vectors.append(vec)

            # Rate-pace: each item in a batch counts as 1 RPM request.
            # Sleep long enough for this chunk's slots to age out of the
            # rolling window before we send the next chunk.
            remaining = len(texts) - (i + self._MAX_BATCH_SIZE)
            if remaining > 0:
                pace_delay = len(chunk) / self._ITEMS_PER_MINUTE * 60
                logger.debug(
                    f"GeminiEmbeddingProvider: pacing {pace_delay:.1f}s "
                    f"before next chunk ({remaining} items remaining)"
                )
                await asyncio.sleep(pace_delay)

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
