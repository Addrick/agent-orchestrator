"""TEMP dev launcher — serves ONLY the Kobold engine adapter (port 5003) over the
real user-memory DB, with no Discord/Gmail/agents/hindsight startup. Used to wire
up + verify the bespoke DERPR Portal UI against live read/stream endpoints and the
/derpr production mount. Not part of the app; delete after verification.
"""
import asyncio
import logging
import os

os.environ.setdefault("SEMANTIC_BACKEND", "sqlite")  # fully local; read path is sqlite regardless

from dotenv import load_dotenv
load_dotenv(".env")
os.environ["SEMANTIC_BACKEND"] = "sqlite"  # override .env (may be hindsight)

from src.memory.memory_manager import MemoryManager
from src.engine import TextEngine
from src.stream_engine import StreamEngine
from src.chat_system import ChatSystem
from src.interfaces.kobold_engine_adapter import create_kobold_engine_adapter
from config.global_config import MEMORY_DATABASE_FILE

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s:%(lineno)d %(message)s")
log = logging.getLogger("derpr_ui_devserver")


async def main() -> None:
    log.info("DB: %s", MEMORY_DATABASE_FILE)
    mm = MemoryManager(db_path=MEMORY_DATABASE_FILE)
    mm.create_schema()
    se = StreamEngine()
    te = TextEngine(stream_engine=se)
    bot = ChatSystem(memory_manager=mm, text_engine=te, embedding_service=None, stream_engine=se)
    log.info("personas: %s", sorted(bot.visible_personas().keys()))
    adapter = create_kobold_engine_adapter(bot)
    adapter.port = int(os.environ.get("DERPR_ENGINE_PORT", "5003"))
    adapter.host = "127.0.0.1"
    try:
        await adapter.start()
    finally:
        await se.aclose()


if __name__ == "__main__":
    asyncio.run(main())
