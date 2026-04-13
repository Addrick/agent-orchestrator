# tests/database/test_memory_consolidation.py

import pytest
import math
import struct
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from src.database.memory_manager import (
    MemoryManager, LEVEL_EPISODIC, LEVEL_CORE
)
from src.database.memory_consolidation import MemoryConsolidator
from src.persona import Persona
from src.embedding_service import EmbeddingService

def _unit_blob(*components):
    """Create a normalized float32 BLOB of dimension 3072."""
    vec = [0.0] * 3072
    for i, val in enumerate(components):
        if i < 3072:
            vec[i] = val
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return struct.pack('3072f', *vec)

@pytest.fixture
def mem_manager():
    manager = MemoryManager(db_path=":memory:")
    manager.create_schema()
    yield manager
    manager.close()

@pytest.fixture
def consolidator(mem_manager):
    te = MagicMock()
    es = MagicMock(spec=EmbeddingService)
    es.model_name = "test-embedding-model"
    return MemoryConsolidator(mem_manager, te, es)

@pytest.mark.asyncio
async def test_consolidate_clusters_related_summaries(mem_manager, consolidator):
    # 1. Seed L1 summaries (LEVEL_EPISODIC)
    # They need to be in the same channel/persona
    ts = datetime.now()
    seg_id = mem_manager.store_segment("chan1", "srv1", "persona1", 1, 10, 10, ts)
    
    # Create two very similar embeddings (1.0, 0.0, ...)
    emb1 = _unit_blob(1.0, 0.0)
    emb2 = _unit_blob(0.99, 0.1) # similarity ~0.99
    
    mem_manager.store_summary(seg_id, "Fact A", emb1, "m1", ts, summary_level=LEVEL_EPISODIC)
    mem_manager.store_summary(seg_id, "Fact B", emb2, "m1", ts, summary_level=LEVEL_EPISODIC)
    
    # 2. Mock LLM response for consolidation
    consolidator.text_engine.generate_response = AsyncMock(return_value=(
        {'type': 'text', 'content': 'Condensed Fact A and B'}, None
    ))
    consolidator.embedding_service.encode_single = AsyncMock(return_value=emb1)
    
    # 3. Run consolidation
    persona = Persona(persona_name="persona1", model_name="m1", prompt="p")
    await consolidator.consolidate_memory(persona, "persona1", "chan1", "srv1")
    
    # 4. Verify results
    with mem_manager.transaction() as conn:
        # Should have 1 LEVEL_CORE profile
        core = conn.execute("SELECT * FROM Memory_Summaries WHERE summary_level = ?", (LEVEL_CORE,)).fetchall()
        assert len(core) == 1
        assert core[0]['content'] == 'Condensed Fact A and B'
        
        # Original episodics should stay LEVEL_EPISODIC (1)
        # but have a parent_summary_id
        episodics = conn.execute("SELECT * FROM Memory_Summaries WHERE summary_level = ? AND parent_summary_id IS NOT NULL", (LEVEL_EPISODIC,)).fetchall()
        assert len(episodics) == 2
        for row in episodics:
            assert row['parent_summary_id'] == core[0]['summary_id']

@pytest.mark.asyncio
async def test_consolidate_skips_unrelated_summaries(mem_manager, consolidator):
    ts = datetime.now()
    seg_id = mem_manager.store_segment("chan1", "srv1", "persona1", 1, 10, 10, ts)
    
    # Orthogonal embeddings
    emb1 = _unit_blob(1.0, 0.0)
    emb2 = _unit_blob(0.0, 1.0)
    
    mem_manager.store_summary(seg_id, "Fact A", emb1, "m1", ts, summary_level=LEVEL_EPISODIC)
    mem_manager.store_summary(seg_id, "Fact B", emb2, "m1", ts, summary_level=LEVEL_EPISODIC)
    
    consolidator.similarity_threshold = 0.9
    persona = Persona(persona_name="persona1", model_name="m1", prompt="p")
    await consolidator.consolidate_memory(persona, "persona1", "chan1", "srv1")
    
    with mem_manager.transaction() as conn:
        # No core profiles should be created
        core = conn.execute("SELECT * FROM Memory_Summaries WHERE summary_level = ?", (LEVEL_CORE,)).fetchall()
        assert len(core) == 0
        # Summaries remain LEVEL_EPISODIC
        episodic = conn.execute("SELECT * FROM Memory_Summaries WHERE summary_level = ?", (LEVEL_EPISODIC,)).fetchall()
        assert len(episodic) == 2

@pytest.mark.asyncio
async def test_retrieval_excludes_archived_and_prefers_core(mem_manager, consolidator):
    # This test verifies the MemoryManager.retrieve_relevant_summaries logic 
    # that we just updated.
    
    ts = datetime.now()
    seg_id = mem_manager.store_segment("chan1", "srv1", "persona1", 1, 10, 10, ts)
    
    # L2 Core Profile
    emb_core = _unit_blob(1.0, 0.0)
    core_id = mem_manager.store_summary(seg_id, "Core Fact", emb_core, "m1", ts, summary_level=LEVEL_CORE)
    
    # Subsumed L1 (No longer tagged L3, stays L1 but with a parent)
    mem_manager.store_summary(seg_id, "Archived Fact", emb_core, "m1", ts, 
                               summary_level=LEVEL_EPISODIC, parent_summary_id=core_id)
    
    # Loose L1 (Not yet subsumed)
    emb_loose = _unit_blob(0.0, 1.0)
    mem_manager.store_summary(seg_id, "Loose Fact", emb_loose, "m1", ts, summary_level=LEVEL_EPISODIC)
    
    # Retrieve
    results = mem_manager.retrieve_relevant_summaries(
        "persona1", "chan1", server_id="srv1", memory_mode="channel",
        query_embeddings=[emb_core, emb_loose]
    )
    
    # Should get "Core Fact" and "Loose Fact", but NOT "Archived Fact"
    contents = {r['content'] for r in results}
    assert "Core Fact" in contents
    assert "Loose Fact" in contents
    assert "Archived Fact" not in contents
    assert len(results) == 2
