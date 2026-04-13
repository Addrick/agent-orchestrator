# src/embedding_service.py

import asyncio
import logging
import struct
import time
import re
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np

# Import global configurations
from config.global_config import (
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    GEMINI_EMBEDDING_001_RPM,
    GEMINI_EMBEDDING_001_TPM,
    GEMINI_EMBEDDING_001_RPD
)

logger = logging.getLogger(__name__)


class EmbeddingRateLimiter:
    """Proactively tracks item-based Google API quota limits to prevent 429s."""

    def __init__(self, rpm: int, tpm: int, rpd: int):
        self.rpm = rpm
        self.tpm = tpm
        self.rpd = rpd
        self.minute_req_history: List[Tuple[float, int]] = []
        self.minute_tok_history: List[Tuple[float, int]] = []
        self.day_req_history: List[Tuple[float, int]] = []
        self._lock = asyncio.Lock()

    async def acquire(self, item_count: int, token_count: int) -> None:
        """Awaits until the payload can safely be sent without hitting a 429."""
        async with self._lock:
            while True:
                now = time.time()

                # Prune out-of-window timestamps
                self.minute_req_history = [(t, c) for t, c in self.minute_req_history if now - t < 60.0]
                self.minute_tok_history = [(t, tok) for t, tok in self.minute_tok_history if now - t < 60.0]
                self.day_req_history = [(t, c) for t, c in self.day_req_history if now - t < 86400.0]

                # 1. Enforce Daily Limit (Hard stop)
                current_day_reqs = sum(c for _, c in self.day_req_history)
                if current_day_reqs + item_count > self.rpd:
                    raise RuntimeError(f"Daily Google API Quota Exhausted ({self.rpd} items).")

                # 2. Check Minute Limits
                current_min_reqs = sum(c for _, c in self.minute_req_history)
                current_min_toks = sum(tok for _, tok in self.minute_tok_history)

                if current_min_reqs + item_count <= self.rpm and current_min_toks + token_count <= self.tpm:
                    # Safe to proceed!
                    self.minute_req_history.append((now, item_count))
                    self.minute_tok_history.append((now, token_count))
                    self.day_req_history.append((now, item_count))
                    break

                # 3. Throttle if we don't have capacity
                sleep_time = 0.0
                if current_min_reqs + item_count > self.rpm and self.minute_req_history:
                    sleep_time = max(sleep_time, 60.0 - (now - self.minute_req_history[0][0]))
                if current_min_toks + token_count > self.tpm and self.minute_tok_history:
                    sleep_time = max(sleep_time, 60.0 - (now - self.minute_tok_history[0][0]))

                if sleep_time > 0:
                    logger.info(f"Embedding API throttle: pausing {sleep_time:.1f}s for rate limit reset "
                                f"(TPM: {current_min_toks}/{self.tpm}, RPM: {current_min_reqs}/{self.rpm})")
                    await asyncio.sleep(sleep_time + 0.1)


# Global singleton so all instances and agents share the same rate-limit history
GLOBAL_EMBEDDING_LIMITER = EmbeddingRateLimiter(
    rpm=GEMINI_EMBEDDING_001_RPM,
    tpm=GEMINI_EMBEDDING_001_TPM,
    rpd=GEMINI_EMBEDDING_001_RPD
)


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

    @model_name.setter
    @abstractmethod
    def model_name(self, value: str) -> None:
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
    """Uses Google's API via the google-genai SDK."""

    _dimensions = EMBEDDING_DIMENSION
    _max_input_tokens = 2048
    _MAX_BATCH_SIZE = 10  # Kept low to bypass generic payload size errors
    _MAX_RETRIES = 5

    def __init__(self) -> None:
        self._model_name = EMBEDDING_MODEL
        self._limiter = GLOBAL_EMBEDDING_LIMITER

    @property
    def model_name(self) -> str:
        return self._model_name

    @model_name.setter
    def model_name(self, value: str) -> None:
        self._model_name = value

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def max_input_tokens(self) -> Optional[int]:
        return self._max_input_tokens

    @staticmethod
    def _parse_error_details(exc: Exception) -> Tuple[str, float]:
        """Extracts the quota metric and enforces a strict 60s minimum delay."""
        exc_str = str(exc)

        # If we hit a 429, our local limiter is out of sync with Google's API.
        # We MUST wait a full 60 seconds to guarantee Google's sliding window drains.
        delay = 60.0

        # Parse metric name out of Google's new error format
        metric = "Unknown_Quota"
        match_metric = re.search(r"metric:\s*[a-zA-Z0-9.-]+/([^,\s]+)", exc_str)
        if match_metric:
            metric = match_metric.group(1)
            if "limit: 1000" in exc_str:
                metric += "_DAILY_LIMIT"
        elif "quotaId" in exc_str:
            match_quota = re.search(r"'quotaId':\s*'([^']+)'", exc_str)
            if match_quota:
                metric = match_quota.group(1)

        return metric, delay

    async def encode(self, texts: List[str]) -> List[List[float]]:  # noqa: C901
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

            # --- EXACT TOKEN COUNTING ---
            # Instead of guessing with chars // 4, we query the exact token size.
            try:
                count_response = await asyncio.to_thread(
                    client.models.count_tokens,
                    model=self._model_name,
                    contents=chunk,
                )
                exact_tokens = count_response.total_tokens or int(sum(len(t) for t in chunk) / 2.0)
            except Exception as e:
                logger.debug(f"Token count API failed, using conservative heuristic: {e}")
                # Highly conservative fallback if the count API fails (2.0 chars per token)
                exact_tokens = int(sum(len(t) for t in chunk) / 2.0)

            # 1. Proactively wait for API capacity using EXACT token counts
            await self._limiter.acquire(item_count=len(chunk), token_count=exact_tokens)

            # 2. Execute with safety net
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

                    metric, delay = self._parse_error_details(exc)
                    raw_err = str(exc).replace('\n', ' ')

                    logger.warning(
                        f"Gemini API 429 Exception ({metric}) "
                        f"on chunk {i // self._MAX_BATCH_SIZE + 1} "
                        f"(attempt {attempt + 1}/{self._MAX_RETRIES}). "
                        f"API bucket full. Backing off for {delay:.1f}s. "
                        f"[Raw Error: {raw_err}]"
                    )
                    await asyncio.sleep(delay)

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

    @model_name.setter
    def model_name(self, value: str) -> None:
        self._provider.model_name = value

    @property
    def dimensions(self) -> int:
        return self._provider.dimensions

    def _truncate_texts(self, texts: List[str]) -> List[str]:
        """Truncate texts exceeding the provider's max input token limit."""
        max_tokens = self._provider.max_input_tokens
        if max_tokens is None:
            return texts

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
        """Cosine similarity between two BLOB embeddings."""
        a = np.frombuffer(blob_a, dtype=np.float32)
        b = np.frombuffer(blob_b, dtype=np.float32)
        return float(np.dot(a, b))

    @staticmethod
    def cosine_similarities(query_blob: bytes, candidate_blobs: List[bytes]) -> List[float]:
        """Cosine similarity of a query against multiple candidates."""
        query = np.frombuffer(query_blob, dtype=np.float32)
        scores = []
        for blob in candidate_blobs:
            candidate = np.frombuffer(blob, dtype=np.float32)
            scores.append(float(np.dot(query, candidate)))
        return scores
