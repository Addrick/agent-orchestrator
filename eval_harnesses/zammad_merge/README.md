# Zammad Ticket Merging Evaluation Harness

This harness evaluates the autonomous dispatcher bot's ('joy') ability to identify, tag, and merge related tickets (specifically phishing duplicates) in a Zammad ticketing system.

## System Input/Output Details

### 1. User Input (The Trigger)
- **Format**: Natural language request via Discord, Web, or Email.
- **Content**: Instructions to consolidate tickets, often accompanied by a list or summary of current tickets.
- **Example**: *"Joy, please identify the phishing tickets from the list and merge them into the master phishing ticket."*

### 2. LLM Orchestration (The "Brain")
- **Input**: User prompt + Available Tools + Persona Guidelines.
- **Output**: A sequence of tool calls:
    1. `search_tickets(query=...)` to find internal IDs.
    2. `update_ticket(ticket_id=..., payload={'tags': ['phishing']})` to tag.
    3. `merge_tickets(source_ticket_id=..., target_ticket_id=...)` to consolidate.

### 3. Tool Output (`merge_tickets`)
- **Action**: Performs a three-step Zammad API orchestration:
    - `POST /api/v1/links`: Links the source to the target as "merged".
    - `PUT /api/v1/ticket_articles/{id}`: Moves articles from source to target.
    - `PUT /api/v1/tickets/{id}`: Sets source state to "merged" (ID 7).
- **Final Result**: The bot confirms the merge status to the user.

---

## Evaluation Harness Usage

The harness resides in `eval_harnesses/zammad_merge/`.

### Files
- `scenarios.json`: Data-driven test cases including mock tickets, context, and expected tool calls.
- `eval_runner.py`: The execution logic that spins up a mock environment, runs the LLM through the scenarios, and validates the resulting tool calls.

### Running the Evaluation
```powershell
# From the project root
$env:PYTHONPATH="."; python eval_harnesses/zammad_merge/eval_runner.py --model gemini-1.5-flash
```

### Metrics Measured
- **Success Rate**: Did the LLM call `merge_tickets` for all expected source/target pairs?
- **ID Accuracy**: Did it correctly resolve ticket numbers (e.g., #731183) to internal database IDs (e.g., 183)?
- **Latency**: How long did the multi-turn tool loop take?
- **Response Fidelity**: Does the bot's final text accurately reflect the work performed?

## Iteration Support
To test changes to the **tool descriptions** or **persona prompts**, you can:
1. Modify `src/tools/definitions.py` (for descriptions).
2. Modify `config/default_personas.json` or pass an override to the harness (for prompts).
3. Re-run the runner to see the impact on the success metrics.
