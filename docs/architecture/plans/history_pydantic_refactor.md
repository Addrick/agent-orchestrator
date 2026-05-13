# Plan: Pydantic History Refactor

## Objective
Standardize the representation of conversation history using Pydantic models. This ensures data integrity, JSON serializability (especially for Gemini's `bytes` fields), and clear architectural boundaries between LLM providers and internal logic.

## Problem Statement
The current system passes around raw dictionaries for conversation history. LLM providers (specifically Gemini) can return non-JSON-serializable types like `bytes` for fields such as `thought_signature`. Handling these via ad-hoc patches in `engine.py` is brittle and prone to regressions.

## Proposed Solution
Introduce a schema-on-write layer using Pydantic at the edge of the `TextEngine`. All provider responses must be validated into standard models before being appended to the history.

## Implementation Steps

### Phase 1: Foundation
1.  **Create `src/models/history.py`**:
    -   Define `ToolCall` model with a validator to handle `bytes` -> `base64` conversion for `thought_signature`.
    -   Define `ConversationTurn` model to represent a single message (system, user, assistant, tool).
    -   Ensure the models support `model_dump()` to produce the legacy dictionary format required by existing logic.

### Phase 2: TextEngine Boundary
1.  **Refactor `src/engine.py`**:
    -   Import the new models.
    -   Update `_parse_google_response` to instantiate `ToolCall` models, letting Pydantic handle the `bytes` conversion.
    -   Standardize `_parse_openai_tool_calls` and Anthropic parsing to use the same models.
    -   Update `_build_google_history` to handle the reconstruction (decoding b64 back to bytes) using model methods.

### Phase 3: ToolLoop Consolidation
1.  **Refactor `src/tools/tool_loop.py`**:
    -   Update the loop to collect `ConversationTurn` objects.
    -   Use the models to generate `tool_context_json` safely, ensuring no serialization errors can occur.

### Phase 4: Verification
1.  **Add Regression Tests**:
    -   Test that `bytes` in any field are correctly handled by the models.
    -   Test that malformed provider responses are caught early.
    -   Verify that history remains JSON-serializable throughout a multi-turn tool loop.

## Benefits
-   **Robustness**: No more `TypeError: Object of type bytes is not JSON serializable`.
-   **Clarity**: Explicit documentation of what a "Message" looks like in DERPR.
-   **Security**: Prevents unexpected data types from leaking into the database.
-   **Maintainability**: Easier to add support for new providers or complex metadata (e.g., grounding citations).

## Risks
-   **Performance**: Minimal overhead from Pydantic validation (negligible compared to LLM latency).
-   **Compatibility**: Care must be taken to ensure the `model_dump()` output matches the exact structure expected by the various LLM APIs.
