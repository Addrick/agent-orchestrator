# scripts/backfill_hindsight.py
import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.memory.memory_manager import MemoryManager
from src.memory.backend.hindsight import HindsightBackend
from src.utils.save_utils import load_personas_from_file, load_system_personas_from_file
from config.global_config import HINDSIGHT_URL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("backfill")

async def backfill():
    # 1. Initialize
    mm = MemoryManager()
    hindsight = HindsightBackend(url=HINDSIGHT_URL)
    
    # 2. Get all personas to ensure banks exist — but skip system personas
    # (triage_*, *_selector, etc.). They have no conversational LTM and a bank
    # for them stays empty forever. Matches ChatSystem.startup() gate.
    loaded = {**(load_personas_from_file() or {}), **(load_system_personas_from_file() or {})}
    ltm_personas = {n for n, p in loaded.items() if p.get_long_term_memory()}
    with mm.transaction() as conn:
        all_personas = [row["persona_name"] for row in conn.execute("SELECT DISTINCT persona_name FROM User_Interactions")]
    personas = [p for p in all_personas if p in ltm_personas]
    skipped = sorted(set(all_personas) - set(personas))
    if skipped:
        logger.info(f"Skipping non-LTM personas (no bank): {skipped}")

    logger.info(f"Ensuring banks exist for personas: {personas}")
    for persona in personas:
        await hindsight.ensure_bank(
            bank_id=persona,
            enable_observations=True,
            retain_mission=f"Extract facts and experiences for the persona '{persona}'.",
            reflect_mission=f"Reason over the memories of '{persona}' to provide thoughtful insights."
        )

    # 3. Fetch all interactions
    # We order by interaction_id to ensure we process in the order they were recorded
    logger.info("Fetching all interactions from SQLite...")
    with mm.transaction() as conn:
        placeholders = ",".join("?" * len(personas)) if personas else "''"
        rows = conn.execute(f"""
            SELECT interaction_id, user_identifier, persona_name, channel, author_role,
                   content, timestamp, reasoning_content, tool_context
            FROM User_Interactions
            WHERE persona_name IN ({placeholders})
            ORDER BY timestamp ASC, interaction_id ASC
        """, personas).fetchall()

    total = len(rows)
    logger.info(f"Starting backfill of {total} interactions...")

    # 4. Process in batches to avoid overwhelming the async queue too fast
    # Though HindsightBackend has its own queue, we want to monitor progress
    batch_size = 50
    for i in range(0, total, batch_size):
        batch = rows[i:i+batch_size]
        
        for row in batch:
            # Prepare content: include reasoning if present
            content = row["content"] or ""
            if row["reasoning_content"]:
                content = f"<thought>\n{row['reasoning_content']}\n</thought>\n\n{content}"
            
            # Metadata for traceability
            # Hindsight metadata values must be strings (HTTP 422 otherwise).
            metadata = {
                "legacy_id": str(row["interaction_id"]),
                "user": str(row["user_identifier"]),
            }
            if row["tool_context"]:
                metadata["tool_context"] = str(row["tool_context"])

            # Retain turn
            ts = row["timestamp"]
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts)
                except ValueError:
                    # Fallback for older sqlite formats if needed
                    ts = datetime.now(timezone.utc)

            await hindsight.retain_turn(
                bank_id=row["persona_name"],
                role=row["author_role"],
                content=content,
                timestamp=ts,
                scope_tags=[f"channel:{row['channel']}"],
                source_persona=row["persona_name"],
                metadata=metadata
            )
        
        logger.info(f"Enqueued batch {i//batch_size + 1}/{(total // batch_size) + 1} ({min(i+batch_size, total)}/{total})")
        
        # Give the backend workers a tiny bit of time to breathe, 
        # though they process in the background.
        await asyncio.sleep(0.1)

    logger.info("All turns enqueued. Waiting for Hindsight workers to finish processing...")
    logger.info("Note: This depends on your LLM speed (Kobold/Gemma). Watch Hindsight container logs for progress.")
    
    # We don't necessarily have to wait for every single one in this script 
    # since it's fire-and-forget, but for a one-shot backfill we should 
    # ideally wait for the queue to drain if we want to confirm completion.
    # But HindsightBackend doesn't expose a 'wait_until_idle'. 
    # We'll just close it, which gathers the workers (and waits for current batch).
    
    await hindsight.aclose()
    logger.info("Backfill script finished enqueuing.")

if __name__ == "__main__":
    asyncio.run(backfill())
