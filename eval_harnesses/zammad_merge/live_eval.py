import asyncio
import os
import logging
from unittest.mock import patch
from dotenv import load_dotenv
from src.chat_system import ChatSystem
from src.memory.memory_manager import MemoryManager
from src.engine import TextEngine
from src.clients.zammad_client import ZammadClient
from src.clients.zammad_service import ZammadIntegration
from src.persona import ExecutionMode
from src.utils.save_utils import load_personas_from_file

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_live_merge_eval():
    load_dotenv()
    
    # 1. Initialize components
    db_path = "eval_harnesses/zammad_merge/live_eval_v3.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    memory_manager = MemoryManager(db_path=db_path)
    memory_manager.create_schema()
    
    text_engine = TextEngine()
    
    # Explicitly load from default_personas.json to ensure 'joy' is present
    default_file = os.path.join("config", "default_personas.json")
    personas = load_personas_from_file(file_path_override=default_file)
    
    with patch('src.chat_system.load_personas_from_file', return_value=personas):
        chat_system = ChatSystem(memory_manager=memory_manager, text_engine=text_engine)
    
    zammad_client = ZammadClient()
    chat_system.register_service(ZammadIntegration(zammad_client))
    
    # 2. Configure Joy for live autonomous mode
    joy = chat_system.personas.get("joy")
    if not joy:
        print("Persona 'joy' not found.")
        return
    
    joy.set_execution_mode(ExecutionMode.AUTONOMOUS)
    joy.set_service_bindings(["zammad"])
    # Ensure tool descriptions are clean (no conflicts with grounding)
    joy.set_enabled_tools([
        "search_tickets", 
        "get_ticket_details", 
        "update_ticket", 
        "merge_tickets"
    ])
    joy.set_model_name("gemma-4-31b-it") # Use Gemma 4 as requested

    # 3. Define the prompt with the REAL ticket numbers I just created
    # Based on the seed output:
    # #731435 - Phishing Master Ticket (ID: 1436)
    # #731436 - Coordination ID
    # #731437 - Document Status
    # #731438 - Document Status
    # #731439 - URGENT: Internal Security Test
    # #731441 - Re: Your Google Account password
    
    ticket_summary = """
Daily Ticket Summary:
• #731452 Fwd: Coordination ID: #4/24/2026 — Tina Azarvand
• #731453 Fwd: Document Status: 5d0ec1fc-27ad-4d0d-a9c9-5680519b1270 - Final — Tina Azarvand
• #731454 Fwd: Document Status: 5d0ec1fc-27ad-4d0d-a9c9-5680519b1270 - Final — Azarvand Tax Law
• #731455 URGENT: Internal Security Test (Action Required) — Tina Azarvand
• #731456 Email Accounts — Tina Azarvand
• #731457 Re: Your Google Account password for azarvandtaxlaw.com has changed — Tina Azarvand
• #731458 2025 1099-NEC — Tina Azarvand
"""
    user_request = "Joy, please identify the phishing tickets from that list and merge them into the 'Phishing Master Ticket' (#731451). You must search for each ticket individually to find its internal ID before merging. Do not use the ticket number as the ID."

    print("\n--- Starting LIVE Evaluation Scenario ---")
    print(f"User Request: {user_request}")

    # 4. Execute
    response, r_type, _, _ = await chat_system.generate_response(
        persona_name="joy",
        user_identifier="adam",
        channel="live_eval_channel",
        message=f"{ticket_summary}\n\n{user_request}"
    )

    print("\n--- Final Bot Response ---")
    print(response)
    
    memory_manager.close()

if __name__ == "__main__":
    asyncio.run(run_live_merge_eval())
