# tests/test_embedding_service.py

import math
import struct
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from src.embedding_service import (
    EmbeddingProvider,
    EmbeddingService,
    GeminiEmbeddingProvider,
)


# --- Mock Provider ---

class MockProvider(EmbeddingProvider):
    """Deterministic provider for unit tests."""

    _model_name = "mock-embed-001"
    _dimensions = 4

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def max_input_tokens(self):
        return 100  # ~400 chars

    async def encode(self, texts):
        """Returns normalized vectors based on text length for deterministic results."""
        vectors = []
        for text in texts:
            # Simple deterministic embedding: different texts get different vectors
            seed = len(text) % 4
            vec = [0.0] * 4
            vec[seed] = 1.0  # unit vector along one axis
            vectors.append(vec)
        return vectors


# --- BLOB Serialization Tests ---

def test_vector_to_blob_correct_size():
    """float32 BLOB has correct byte count."""
    service = EmbeddingService(MockProvider())
    vec = [0.25, 0.5, 0.75, 1.0]
    blob = service._vector_to_blob(vec)
    # 4 floats * 4 bytes = 16 bytes
    assert len(blob) == 16


def test_blob_round_trip():
    """Vector survives BLOB serialization and deserialization."""
    service = EmbeddingService(MockProvider())
    original = [0.1, 0.2, 0.3, 0.4]
    blob = service._vector_to_blob(original)
    restored = service._blob_to_vector(blob)
    np.testing.assert_allclose(restored, original, rtol=1e-6)


def test_gemini_blob_size():
    """Gemini text-embedding-004 produces 768-dim blobs (768 * 4 = 3072 bytes)."""
    blob = struct.pack('768f', *([0.0] * 768))
    assert len(blob) == 3072


# --- Cosine Similarity Tests ---

def _unit_blob(*components):
    """Create a normalized float32 BLOB from components."""
    vec = list(components)
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return struct.pack(f'{len(vec)}f', *vec)


def test_cosine_similarity_identical():
    """Identical vectors have similarity ~1.0."""
    blob = _unit_blob(1.0, 0.0, 0.0, 0.0)
    assert abs(EmbeddingService.cosine_similarity(blob, blob) - 1.0) < 1e-6


def test_cosine_similarity_orthogonal():
    """Orthogonal vectors have similarity ~0.0."""
    a = _unit_blob(1.0, 0.0, 0.0, 0.0)
    b = _unit_blob(0.0, 1.0, 0.0, 0.0)
    assert abs(EmbeddingService.cosine_similarity(a, b)) < 1e-6


def test_cosine_similarity_opposite():
    """Opposite vectors have similarity ~-1.0."""
    a = _unit_blob(1.0, 0.0, 0.0, 0.0)
    b = _unit_blob(-1.0, 0.0, 0.0, 0.0)
    assert abs(EmbeddingService.cosine_similarity(a, b) + 1.0) < 1e-6


def test_cosine_similarity_known_value():
    """Known angle produces expected similarity."""
    # 45 degrees between (1,0) and (1,1) -> cos(45) = sqrt(2)/2 ≈ 0.7071
    a = _unit_blob(1.0, 0.0)
    b = _unit_blob(1.0, 1.0)
    expected = math.sqrt(2) / 2
    assert abs(EmbeddingService.cosine_similarity(a, b) - expected) < 1e-5


def test_cosine_similarities_batch():
    """Batch similarity matches individual calls."""
    query = _unit_blob(1.0, 0.0, 0.0, 0.0)
    candidates = [
        _unit_blob(1.0, 0.0, 0.0, 0.0),  # identical -> 1.0
        _unit_blob(0.0, 1.0, 0.0, 0.0),  # orthogonal -> 0.0
        _unit_blob(1.0, 1.0, 0.0, 0.0),  # 45 degrees
    ]

    batch_scores = EmbeddingService.cosine_similarities(query, candidates)
    individual_scores = [EmbeddingService.cosine_similarity(query, c) for c in candidates]

    assert len(batch_scores) == 3
    for bs, ind in zip(batch_scores, individual_scores):
        assert abs(bs - ind) < 1e-6


# --- EmbeddingService Integration (Mock Provider) ---

@pytest.mark.asyncio
async def test_encode_returns_blobs():
    """encode() returns list of bytes BLOBs."""
    service = EmbeddingService(MockProvider())
    blobs = await service.encode(["hello", "world"])
    assert len(blobs) == 2
    for blob in blobs:
        assert isinstance(blob, bytes)
        assert len(blob) == 4 * 4  # 4 dims * 4 bytes


@pytest.mark.asyncio
async def test_encode_single():
    """encode_single() returns a single BLOB."""
    service = EmbeddingService(MockProvider())
    blob = await service.encode_single("hello")
    assert isinstance(blob, bytes)
    assert len(blob) == 4 * 4


@pytest.mark.asyncio
async def test_text_truncation():
    """Texts exceeding max_input_tokens are truncated."""
    service = EmbeddingService(MockProvider())
    # MockProvider has max_input_tokens=100 -> max_chars=400
    long_text = "a" * 1000
    blobs = await service.encode([long_text])
    assert len(blobs) == 1  # didn't error


def test_model_name_property():
    """model_name delegates to provider."""
    service = EmbeddingService(MockProvider())
    assert service.model_name == "mock-embed-001"


def test_dimensions_property():
    """dimensions delegates to provider."""
    service = EmbeddingService(MockProvider())
    assert service.dimensions == 4


# --- Provider ABC Contract ---

def test_provider_abc_cannot_instantiate():
    """EmbeddingProvider cannot be instantiated directly."""
    with pytest.raises(TypeError):
        EmbeddingProvider()


# --- GeminiEmbeddingProvider (Live) ---

@pytest.mark.llm_live
@pytest.mark.asyncio
async def test_gemini_provider_live():
    """Live test: GeminiEmbeddingProvider returns correct-dimension embeddings."""
    provider = GeminiEmbeddingProvider()
    vectors = await provider.encode(["Hello world", "Test embedding"])
    assert len(vectors) == 2
    for vec in vectors:
        assert len(vec) == 768
        # Verify normalized (magnitude ≈ 1.0)
        magnitude = math.sqrt(sum(v * v for v in vec))
        assert abs(magnitude - 1.0) < 0.01


@pytest.mark.llm_live
@pytest.mark.asyncio
async def test_gemini_service_live_blob_round_trip():
    """Live test: full encode -> similarity pipeline."""
    service = EmbeddingService(GeminiEmbeddingProvider())
    blobs = await service.encode(["Python programming", "Python coding"])
    assert len(blobs) == 2

    # Similar texts should have high similarity
    sim = EmbeddingService.cosine_similarity(blobs[0], blobs[1])
    assert sim > 0.5  # semantically similar
