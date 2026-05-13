# scripts/backfill_hindsight.py
import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.memory.memory_manager import MemoryManager
from src.memory.backend.hindsight import HindsightBackend
from src.utils.save_utils import load_system_personas_from_file
from config.global_config import HINDSIGHT_URL, MEMORY_DATABASE_FILE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("backfill")

async def backfill(persona_filter: list[str] | None = None, wipe: bool = False):
    # 1. Initialize
    mm = MemoryManager(db_path=MEMORY_DATABASE_FILE)
    hindsight = HindsightBackend(url=HINDSIGHT_URL)

    # 2. Pick personas. Explicit --personas list wins; otherwise auto-include
    # everything in User_Interactions minus system pipeline workers.
    system_persona_names = set((load_system_personas_from_file() or {}).keys())
    with mm.transaction() as conn:
        all_personas = [row["persona_name"] for row in conn.execute("SELECT DISTINCT persona_name FROM User_Interactions")]
    if persona_filter:
        missing = [p for p in persona_filter if p not in all_personas]
        if missing:
            logger.warning(f"Persona(s) not present in User_Interactions: {missing}")
        personas = [p for p in persona_filter if p in all_personas]
    else:
        personas = [p for p in all_personas if p not in system_persona_names]
        skipped = sorted(set(all_personas) - set(personas))
        if skipped:
            logger.info(f"Skipping system personas (no bank): {skipped}")

    logger.info(f"Ensuring banks exist for personas: {personas}")
    for persona in personas:
        if wipe:
            logger.info(f"Wiping bank for persona: {persona}")
            await hindsight.delete_bank(persona)
        
        if persona == "ambient":
            retain_mission = (
                "You are an observer listening to an ambient channel. The messages are from "
                "various users and are not necessarily directed at you. Extract facts about "
                "the users, their interests, and mentioned entities. Attribute actions to "
                "the correct speakers identified in the text (e.g. 'Name: message'). "
                "IMPORTANT: Do NOT attribute these actions to a persona named 'ambient'. "
                "The 'ambient' name refers to the bank itself, not a participant."
            )
            reflect_mission = "Reason over the ambient conversation logs to identify user interests, recurring themes, and entity relationships."
        else:
            retain_mission = f"Extract facts and experiences for the persona '{persona}'."
            reflect_mission = f"Reason over the memories of '{persona}' to provide thoughtful insights."

        await hindsight.ensure_bank(
            bank_id=persona,
            enable_observations=True,
            retain_mission=retain_mission,
            reflect_mission=reflect_mission
        )

    # 3. Fetch all interactions
    # We order by interaction_id to ensure we process in the order they were recorded
    logger.info("Fetching all interactions from SQLite...")
    with mm.transaction() as conn:
        placeholders = ",".join("?" * len(personas)) if personas else "''"
        rows = conn.execute(f"""
            SELECT interaction_id, user_identifier, persona_name, channel, author_role,
                   author_name, content, timestamp, reasoning_content, tool_context
            FROM User_Interactions
            WHERE persona_name IN ({placeholders})
            ORDER BY timestamp ASC, interaction_id ASC
        """, personas).fetchall()

    total = len(rows)
    logger.info(f"Starting backfill of {total} interactions...")

    # 4. Group interactions by (persona, channel) to maximize context window
    # and minimize instruction overhead.
    logger.info("Grouping interactions into context blocks...")
    
    # Structure: {(persona, channel): [list of message dictionaries]}
    groups = {}
    for row in rows:
        key = (row["persona_name"], row["channel"])
        if key not in groups:
            groups[key] = []
        groups[key].append(row)

    MAX_BLOCK_SIZE = 40000  # Characters (approx 10k tokens)
    MAX_BLOCK_MESSAGES = 100
    
    processed_count = 0
    for (persona_name, channel), items in groups.items():
        logger.info(f"Backfilling {len(items)} messages for {persona_name} in #{channel}...")
        
        current_block = []
        current_size = 0
        
        for idx, row in enumerate(items):
            # Prepare the individual message content
            ts = row["timestamp"]
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts)
                except ValueError:
                    ts = datetime.now(timezone.utc)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            msg_content = row["content"] or ""
            if row["reasoning_content"]:
                msg_content = f"<thought>\n{row['reasoning_content']}\n</thought>\n\n{msg_content}"
            
            date_header = ts.strftime("%Y-%m-%d %H:%M:%S")
            speaker = row["author_name"] or row["user_identifier"] or "Unknown"
            formatted_msg = f"Date: {date_header}\nSpeaker: {speaker}\n---\n{msg_content}"
            
            current_block.append((formatted_msg, ts, row["interaction_id"], row["user_identifier"]))
            current_size += len(formatted_msg)
            
            # Ship block if it's full or it's the last message for this group
            if current_size >= MAX_BLOCK_SIZE or len(current_block) >= MAX_BLOCK_MESSAGES or idx == len(items) - 1:
                # Concatenate the messages in the block
                block_content = "\n\n".join([m[0] for m in current_block])
                
                # Use the timestamp of the last message as the anchor
                anchor_ts = current_block[-1][1]
                
                # Metadata from the last message (or collective)
                metadata = {
                    "legacy_ids": ",".join([str(m[2]) for m in current_block]),
                    "users": ",".join(list(set([str(m[3]) for m in current_block]))),
                }

                await hindsight.retain_turn(
                    bank_id=persona_name,
                    role="user", # Grouped chat transcript is treated as input
                    content=block_content,
                    timestamp=anchor_ts,
                    scope_tags=[f"channel:{channel}"],
                    source_persona=persona_name,
                    metadata=metadata
                )
                
                processed_count += len(current_block)
                logger.info(f"Enqueued block of {len(current_block)} messages ({processed_count}/{total})")
                
                current_block = []
                current_size = 0
                
                # Tiny breather for the loop
                await asyncio.sleep(0.05)

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
    parser = argparse.ArgumentParser(description="Backfill User_Interactions into Hindsight banks.")
    parser.add_argument(
        "--personas", nargs="+", default=None,
        help="Persona names to backfill (default: all non-system personas in User_Interactions).",
    )
    parser.add_argument(
        "--wipe", action="store_true",
        help="Delete existing banks for selected personas before backfilling (clears duplicates).",
    )
    args = parser.parse_args()
    asyncio.run(backfill(persona_filter=args.personas, wipe=args.wipe))
