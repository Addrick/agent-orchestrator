import asyncio
import logging
import os
import json
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional
from unittest.mock import MagicMock, patch

from src.chat_system import ChatSystem, ResponseType
from src.memory.memory_manager import MemoryManager
from src.engine import TextEngine
from src.clients.zammad_client import ZammadClient
from src.clients.zammad_service import ZammadIntegration
from src.persona import Persona, ExecutionMode
from src.utils.save_utils import load_personas_from_file

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class ZammadMergeHarness:
    def __init__(self, db_path: str = "eval_harnesses/zammad_merge/eval_temp.db"):
        self.db_path = db_path
        self.memory_manager = None
        self.text_engine = TextEngine()
        self.chat_system = None
        self.mock_zammad = None

    async def setup(self, persona_prompt_override: Optional[str] = None, tool_defs_override: Optional[List[Dict[str, Any]]] = None):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        
        self.memory_manager = MemoryManager(db_path=self.db_path)
        self.memory_manager.create_schema()
        
        self.mock_zammad = MagicMock(spec=ZammadClient)
        self.mock_zammad.api_url = "http://zammad.test"
        
        # Default mock returns for safety
        self.mock_zammad.get_ticket_articles.return_value = [{"id": 1000, "body": "Mock content"}]
        self.mock_zammad.get_tags.return_value = []
        self.mock_zammad.get_user.return_value = {"id": 1, "firstname": "Test", "lastname": "User"}
        self.mock_zammad.update_ticket.return_value = {"status": "success"}
        self.mock_zammad.link_tickets.return_value = {"status": "success"}
        self.mock_zammad.merge_tickets.return_value = {"status": "success"}

        with patch.dict(os.environ, {"ZAMMAD_URL": "http://zammad.test", "ZAMMAD_API_KEY": "test"}):
            # Load personas
            default_file = os.path.join("config", "default_personas.json")
            personas = load_personas_from_file(file_path_override=default_file)
            
            # Apply prompt override
            if persona_prompt_override and "joy" in personas:
                personas["joy"].prompt = persona_prompt_override
            
            with patch('src.chat_system.load_personas_from_file', return_value=personas):
                self.chat_system = ChatSystem(memory_manager=self.memory_manager, text_engine=self.text_engine)
            
            self.chat_system.register_service(ZammadIntegration(self.mock_zammad))

        # Configure Joy
        joy = self.chat_system.personas.get("joy")
        if joy:
            joy.set_execution_mode(ExecutionMode.AUTONOMOUS)
            joy.set_service_bindings(["zammad"])
            joy.set_enabled_tools(["search_tickets", "get_ticket_details", "update_ticket", "merge_tickets"])
            # Use gemini-1.5-flash by default as it's balanced
            joy.set_model_name("gemini-1.5-flash")

    def _setup_scenario_mocks(self, scenario: Dict[str, Any]):
        tickets = scenario.get("mock_data", {}).get("tickets", [])
        
        def mock_search(query, **kwargs):
            q = query.lower() if query else ""
            results = []
            for t in tickets:
                # Basic search matching
                if str(t['id']) in q or t['number'] in q or t['title'].lower() in q:
                    results.append(t)
                elif "phishing" in q and "phishing" in t['title'].lower():
                    results.append(t)
            return results

        self.mock_zammad.search_tickets.side_effect = mock_search

    async def run_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        print(f"\n>>> Running Scenario: {scenario['name']}")
        self._setup_scenario_mocks(scenario)
        
        full_message = f"{scenario['context']}\n\n{scenario['user_request']}" if scenario['context'] else scenario['user_request']
        
        start_time = datetime.now()
        try:
            with patch('src.tools.tool_loop.MAX_TOOL_CALLS', 15):
                response, r_type, _, _ = await self.chat_system.generate_response(
                    persona_name="joy",
                    user_identifier="adam",
                    channel="eval_channel",
                    message=full_message
                )
        except Exception as e:
            return {"name": scenario['name'], "success": False, "error": str(e)}
        
        duration = (datetime.now() - start_time).total_seconds()
        
        # Analyze results
        merge_calls = [kwargs if kwargs else args for args, kwargs in self.mock_zammad.merge_tickets.call_args_list]
        
        # Basic validation
        success = True
        missing_merges = []
        expected = scenario.get("expected_merges", [])
        
        for exp in expected:
            found = False
            for call in merge_calls:
                # Check if this call matches the expectation
                source_id = call.get('source_ticket_id')
                target_id = call.get('target_ticket_id')
                
                # Check patterns (numbers or IDs)
                match_source = False
                if 'source_pattern' in exp:
                    match_source = exp['source_pattern'] in str(source_id)
                else:
                    match_source = source_id == exp['source']
                
                match_target = False
                if 'target_pattern' in exp:
                    match_target = exp['target_pattern'] in str(target_id)
                elif 'target' in exp:
                    match_target = target_id == exp['target']
                
                if match_source and (match_target or 'target_pattern' not in exp):
                    found = True
                    break
            
            if not found:
                success = False
                missing_merges.append(exp)

        return {
            "name": scenario['name'],
            "success": success,
            "duration": duration,
            "merge_calls": merge_calls,
            "missing_merges": missing_merges,
            "bot_response": response
        }

    def cleanup(self):
        if self.memory_manager:
            self.memory_manager.close()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

async def main():
    parser = argparse.ArgumentParser(description="Zammad Merge Evaluation Harness")
    parser.add_argument("--scenarios", type=str, default="eval_harnesses/zammad_merge/scenarios.json")
    parser.add_argument("--model", type=str, default="gemini-1.5-flash")
    args = parser.parse_args()

    if not os.path.exists(args.scenarios):
        print(f"Scenarios file not found: {args.scenarios}")
        return

    with open(args.scenarios, 'r') as f:
        scenarios = json.load(f)

    harness = ZammadMergeHarness()
    results = []
    
    try:
        await harness.setup()
        # Override model if specified
        if args.model:
            harness.chat_system.personas["joy"].set_model_name(args.model)

        for scenario in scenarios:
            res = await harness.run_scenario(scenario)
            results.append(res)
            print(f"Result: {'PASS' if res['success'] else 'FAIL'} ({res.get('duration', 0):.2f}s)")
            if not res['success']:
                print(f"  Missing Merges: {res.get('missing_merges')}")
            
            # Reset mocks for next scenario
            harness.mock_zammad.reset_mock()
            # Restore default returns
            harness.mock_zammad.merge_tickets.return_value = {"status": "success"}

    finally:
        harness.cleanup()

    # Final Summary
    passed = len([r for r in results if r['success']])
    print(f"\n=== EVALUATION SUMMARY ===")
    print(f"Passed: {passed}/{len(results)}")
    
    # Save results
    output_file = f"eval_harnesses/zammad_merge/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    asyncio.run(main())
