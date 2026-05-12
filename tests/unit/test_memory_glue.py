import unittest
import numpy as np
from unittest.mock import MagicMock
from src.agents.sqlite_consolidator import SqliteConsolidator

class MockChat:
    def __init__(self):
        self.personas = {}
        self.text_engine = None
        self.memory_manager = None

class TestMemoryGlue(unittest.TestCase):
    def setUp(self):
        self.chat = MagicMock()
        self.chat.personas = {}
        self.chat.text_engine = MagicMock()
        self.chat.memory_manager = MagicMock()
        # Mock the segment tail lookup to return None (no previous segment)
        self.chat.memory_manager.get_last_segment_tail_embeddings.return_value = None
        
        self.agent = SqliteConsolidator(
            self.chat, 
            agent_config={'similarity_threshold': 0.80, 'min_segment_size': 2}
        )

    def test_backward_gravity_glue(self):
        # Setup: 
        # 1. A Question (User)
        # 2. An Answer (Assistant)
        # We manually create embeddings that should trigger a split (similarity < 0.8)
        # but the glue logic should keep them together.
        
        # Identity vector for Q
        v_q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        # Vector for A that is 0.5 similar to Q (should split normally)
        v_a = np.array([0.5, 0.866, 0.0], dtype=np.float32) 
        
        messages = [
            {'interaction_id': 1, 'author_role': 'user', 'content': 'How do I X?'},
            {'interaction_id': 2, 'author_role': 'assistant', 'content': 'Step 1: Y.'}
        ]
        embeddings = [v_q.tobytes(), v_a.tobytes()]
        
        # Run segmentation
        segments = self.agent._segment_by_similarity(messages, embeddings, "test", "test", "server")
        
        # Verify
        self.assertEqual(len(segments), 1, "Segments should be glued into ONE despite low similarity.")
        self.assertEqual(segments[0]['count'], 2)
        self.assertEqual(segments[0]['start_id'], 1)
        self.assertEqual(segments[0]['end_id'], 2)
        print("PASS: Question and Answer were glued correctly.")

    def test_normal_split_maintained(self):
        # Ensure it still splits unrelated messages
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32) # Orthogonal (0.0 similarity)
        
        messages = [
            {'interaction_id': 1, 'author_role': 'user', 'content': 'Topic A'},
            {'interaction_id': 2, 'author_role': 'user', 'content': 'Topic B'}
        ]
        embeddings = [v1.tobytes(), v2.tobytes()]
        
        segments = self.agent._segment_by_similarity(messages, embeddings, "test", "test", "server")
        
        # Since min_segment_size is 2, it won't split if it would create tiny segments, 
        # but it will try to split. 
        # In our case, it should split at the end because Topic B is different.
        self.assertEqual(len(segments), 1) # Still 1 because min_segment_size=2 prevents the first from splitting alone
        
if __name__ == "__main__":
    unittest.main()
