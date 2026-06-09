# src/persona_fields.py
"""Declarative persona-field registry (DP-200 slice D).

One table describing every operator-tunable persona field, consumed by every
surface that previously hand-maintained its own copy:

  - BotLogic `what <field>` / `set <field>` dev-command dispatch
    (formerly ~50 near-identical `_what_X` / `_set_X` methods),
  - the kobold engine adapter's PATCH /persona/{name} route
    (known-key set + per-field apply/reject semantics),
  - the `help` command's field listings.

Adding a persona field = one `PersonaField` entry here (plus its accessor on
`Persona`, which keeps owning validation/coercion — this module never
reimplements that; `set_cli`/`patch_apply` always delegate to Persona setters).

Fields that need ChatSystem state or an LLM call (`set model`/`set tools`
fuzzy matching, `what models/personas/tools/security`) stay as bespoke
BotLogic methods and are merged into the same dispatch tables there.

All user-visible strings are preserved verbatim from the pre-registry
methods — tests and operators depend on them.
"""

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type

from config.global_config import DEFAULT_PERSONA
from src.persona import ExecutionMode, MemoryMode, Persona

# `what <field>` renderer: persona -> response text.
WhatFn = Callable[[Persona], str]
# `set <field> ...` handler: (full args incl. field name, persona) ->
# (response | None to fall through, mutated).
SetFn = Callable[[List[str], Persona], Tuple[Optional[str], bool]]
# PATCH apply: (persona, raw JSON value) -> True if the value was rejected.
PatchFn = Callable[[Persona, Any], bool]


@dataclass(frozen=True)
class PersonaField:
    name: str                          # dev-command key ('temp', 'top_p', …)
    describe: Optional[WhatFn] = None  # None = not queryable via `what`
    set_cli: Optional[SetFn] = None    # None = not settable via `set`
    patch_key: Optional[str] = None    # JSON key on the PATCH route, if patchable
    patch_apply: Optional[PatchFn] = None


# ---------------------------------------------------------------------------
# Factories for the recurring shapes
# ---------------------------------------------------------------------------

def _optional_number_setter(
        *,
        caster: Callable[[str], Any],
        apply: Callable[[Persona, Any], Any],
        ok_msg: str,
        fallback_msg: str,
        lo: Optional[float] = None,
        hi: Optional[float] = None,
        range_msg: Optional[str] = None,
) -> SetFn:
    """Numeric param that clears to provider default on non-numeric input.

    Mirrors the legacy `_set_temp`-family semantics exactly: missing arg falls
    through (None, False); a parse failure *sets None* (a mutation) and says
    the default will be used; an out-of-range value is an error with no
    mutation.
    """
    def set_cli(args: List[str], persona: Persona) -> Tuple[Optional[str], bool]:
        try:
            raw = args[1]
        except IndexError:
            return None, False
        try:
            value = caster(raw)
        except ValueError:
            apply(persona, None)
            return fallback_msg.format(raw=raw, name=persona.get_name()), True
        if range_msg is not None and not (lo <= value <= hi):
            return range_msg, False
        apply(persona, value)
        return ok_msg.format(value=value, name=persona.get_name()), True
    return set_cli


_TRUTHY = ('true', 'on', 'yes', '1')
_FALSY = ('false', 'off', 'no', '0')


def _bool_setter(
        *,
        apply: Callable[[Persona, bool], Any],
        missing_msg: str,
        state_msg: Callable[[bool, str], str],
) -> SetFn:
    """on/off toggle with the legacy truthy/falsy token sets."""
    def set_cli(args: List[str], persona: Persona) -> Tuple[Optional[str], bool]:
        try:
            value_str = args[1].lower()
        except IndexError:
            return missing_msg, False
        if value_str in _TRUTHY:
            new_value = True
        elif value_str in _FALSY:
            new_value = False
        else:
            return f"Error: Invalid value '{value_str}'. Please use 'on' or 'off'.", False
        apply(persona, new_value)
        return state_msg(new_value, persona.get_name()), True
    return set_cli


def _enum_setter(
        *,
        enum_cls: Type[Any],
        apply: Callable[[Persona, str], Any],
        label_title: str,    # "Execution mode"
        label_lower: str,    # "execution mode"
        label_article: str,  # "an execution mode"
) -> SetFn:
    def set_cli(args: List[str], persona: Persona) -> Tuple[Optional[str], bool]:
        valid_modes = ", ".join([e.name.lower() for e in enum_cls])
        try:
            mode_str = args[1].upper()
        except IndexError:
            return f"Error: Please specify {label_article}. Valid modes are: {valid_modes}.", False
        try:
            enum_cls[mode_str]
            apply(persona, mode_str)
            return f"{label_title} for {persona.get_name()} set to '{mode_str}'.", True
        except KeyError:
            return f"Error: Invalid {label_lower} '{args[1]}'. Valid modes are: {valid_modes}.", False
    return set_cli


def _optional_number_patch(apply: Callable[[Persona, Any], Any]) -> PatchFn:
    """PATCH semantics for the clear-to-default numeric setters: the Persona
    setter returns the resolved value, so None back for non-None input means
    the value was rejected (coerced away)."""
    def patch_apply(persona: Persona, value: Any) -> bool:
        return apply(persona, value) is None and value is not None
    return patch_apply


def _plain_patch(apply: Callable[[Persona, Any], Any]) -> PatchFn:
    def patch_apply(persona: Persona, value: Any) -> bool:
        apply(persona, value)
        return False
    return patch_apply


# ---------------------------------------------------------------------------
# Bespoke (but persona-only) handlers that don't fit a factory
# ---------------------------------------------------------------------------

def _what_prompt(p: Persona) -> str:
    return f"Prompt for '{p.get_name()}': {p.get_prompt()}"


def _set_prompt(args: List[str], persona: Persona) -> Tuple[Optional[str], bool]:
    prompt = ' '.join(args[1:])
    if not prompt:
        return None, False
    persona.set_prompt(prompt)
    return 'Prompt saved.', True


def _set_default_prompt(args: List[str], persona: Persona) -> Tuple[Optional[str], bool]:
    persona.set_prompt(DEFAULT_PERSONA)
    return f"Prompt for {persona.get_name()} reset to default.", True


def _set_history(args: List[str], persona: Persona) -> Tuple[Optional[str], bool]:
    if len(args) < 2:
        return "Usage: set history <number|dynamic> [start_value]", False

    mode = args[1].lower()
    if mode == 'dynamic':
        start_value: int
        if len(args) > 2:
            try:
                start_value = int(args[2])
            except ValueError:
                return f"Error: Invalid start value '{args[2]}'. Must be an integer.", False
        else:
            start_value = persona.get_current_effective_history_messages()

        persona.start_new_conversation(start_value)
        return f"Dynamic history mode enabled for {persona.get_name()}, starting at size {start_value}.", True
    else:
        try:
            history_messages = int(mode)
            persona.set_history_messages(history_messages)
            return f"Set static history limit for {persona.get_name()} to '{history_messages}' messages.", True
        except ValueError:
            return f"Error: Invalid history command '{mode}'. Use a number or 'dynamic'.", False


def _set_tokens(args: List[str], persona: Persona) -> Tuple[Optional[str], bool]:
    try:
        limit_str = args[1]
        token_limit = int(limit_str)
        persona.set_response_token_limit(token_limit)
        return f"Set token limit to '{token_limit}' for {persona.get_name()}.", True
    except IndexError:
        return None, False
    except ValueError:
        limit_str = args[1]
        persona.set_response_token_limit(None)
        return f"Non-numeric token limit '{limit_str}' provided. The default token limit will be used for {persona.get_name()}.", True


def _set_service_bindings(args: List[str], persona: Persona) -> Tuple[Optional[str], bool]:
    if len(args) < 2:
        return "Error: Please specify service bindings (comma-separated, or 'none' to clear).", False
    value_str = args[1].lower().strip()
    if value_str in ['none', 'clear', '[]']:
        persona.set_service_bindings([])
        return f"Service bindings for {persona.get_name()} cleared.", True
    bindings = [b.strip() for b in value_str.split(',') if b.strip()]
    persona.set_service_bindings(bindings)
    return f"Service bindings for {persona.get_name()} set to: {', '.join(bindings)}.", True


def _set_chat_template(args: List[str], persona: Persona) -> Tuple[Optional[str], bool]:
    if len(args) < 2:
        return "Error: Please specify a chat template (e.g. 'chatml', 'llama3') or 'none' to clear.", False
    value_str = args[1].lower().strip()
    if value_str in ('none', 'null', 'clear'):
        persona.set_chat_template(None)
        return f"Chat template for {persona.get_name()} cleared (reverting to global default).", True
    persona.set_chat_template(value_str)
    return f"Chat template for {persona.get_name()} set to '{value_str}'.", True


def _set_thinking_level(args: List[str], persona: Persona) -> Tuple[Optional[str], bool]:
    try:
        value = args[1].lower()
    except IndexError:
        return "Error: Please specify a thinking level (e.g. 'minimal') or 'none' to clear.", False

    if value == 'none':
        persona.set_thinking_level(None)
        return f"Thinking level cleared for {persona.get_name()} (will use model default).", True
    persona.set_thinking_level(value)
    return f"Thinking level for {persona.get_name()} set to '{value}'.", True


def _set_max_context_tokens(args: List[str], persona: Persona) -> Tuple[Optional[str], bool]:
    try:
        value = args[1]
    except IndexError:
        return "Error: Please specify an integer max_context_tokens value.", False
    new_val = persona.set_max_context_tokens(value)
    return f"Max context tokens for {persona.get_name()} set to {new_val}.", True


def _set_tool_policy(args: List[str], persona: Persona) -> Tuple[Optional[str], bool]:
    if len(args) < 2:
        return "Usage: set tool_policy <json_string>", False
    json_str = " ".join(args[1:])
    try:
        policy_dict = json.loads(json_str)
        persona.set_tool_policy(policy_dict)
        return f"Tool policy for {persona.get_name()} updated.", True
    except json.JSONDecodeError:
        return "Error: Invalid JSON string for tool policy.", False
    except ValueError as e:
        return f"Error: {e}", False


def _patch_memory_mode(persona: Persona, value: Any) -> bool:
    before = persona.get_memory_mode()
    persona.set_memory_mode(value)
    return persona.get_memory_mode() == before and value not in (None, before.name)


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

PERSONA_FIELDS: List[PersonaField] = [
    PersonaField(
        name='prompt',
        describe=_what_prompt,
        set_cli=_set_prompt,
        patch_key='prompt',
        patch_apply=_plain_patch(lambda p, v: p.set_prompt(v)),
    ),
    PersonaField(
        name='default_prompt',
        set_cli=_set_default_prompt,
    ),
    PersonaField(
        # `set model` is bespoke (async LLM fuzzy match, lives in BotLogic);
        # the PATCH route takes a literal model_name with no fuzzing.
        name='model',
        describe=lambda p: f"{p.get_name()} is using {p.get_model_name()}",
        patch_key='model_name',
        patch_apply=_plain_patch(lambda p, v: p.set_model_name(v)),
    ),
    PersonaField(
        name='history',
        describe=lambda p: f"{p.get_name()} default history message count is {p.get_base_history_messages()}.",
        set_cli=_set_history,
    ),
    PersonaField(
        name='tokens',
        describe=lambda p: f"{p.get_name()} is limited to {p.get_response_token_limit()} response tokens.",
        set_cli=_set_tokens,
        patch_key='max_tokens',
        patch_apply=_plain_patch(lambda p, v: p.set_response_token_limit(v)),
    ),
    PersonaField(
        name='temp',
        describe=lambda p: f"Temperature for {p.get_name()} is set to {p.get_temperature() or 'default'}.",
        set_cli=_optional_number_setter(
            caster=float,
            apply=lambda p, v: p.set_temperature(v),
            ok_msg="Set temperature to {value} for {name}.",
            fallback_msg="Non-numeric temperature '{raw}' provided. The default temperature will be used for {name}.",
            lo=0, hi=2,
            range_msg="Error: Temperature must be between 0 and 2.",
        ),
        patch_key='temperature',
        patch_apply=_optional_number_patch(lambda p, v: p.set_temperature(v)),
    ),
    PersonaField(
        name='top_p',
        describe=lambda p: f"Top P for {p.get_name()} is set to {p.get_top_p() or 'default'}.",
        set_cli=_optional_number_setter(
            caster=float,
            apply=lambda p, v: p.set_top_p(v),
            ok_msg="Set top_p to {value} for {name}.",
            fallback_msg="Non-numeric Top P '{raw}' provided. The default Top P will be used for {name}.",
            lo=0, hi=1,
            range_msg="Error: Top P must be between 0 and 1.",
        ),
        patch_key='top_p',
        patch_apply=_optional_number_patch(lambda p, v: p.set_top_p(v)),
    ),
    PersonaField(
        name='top_k',
        describe=lambda p: f"Top K for {p.get_name()} is set to {p.get_top_k() or 'default'}.",
        set_cli=_optional_number_setter(
            caster=int,
            apply=lambda p, v: p.set_top_k(v),
            ok_msg="Set top_k to {value} for {name}.",
            fallback_msg="Non-numeric Top K '{raw}' provided. The default Top K will be used for {name}.",
        ),
        patch_key='top_k',
        patch_apply=_optional_number_patch(lambda p, v: p.set_top_k(v)),
    ),
    PersonaField(
        name='display_name',
        describe=lambda p: (
            f"Display name in chat for '{p.get_name()}' is "
            f"{'enabled' if p.should_display_name_in_chat() else 'disabled'}."
        ),
        set_cli=_bool_setter(
            apply=lambda p, v: p.set_display_name_in_chat(v),
            missing_msg="Error: Please specify 'on' or 'off' for the display name.",
            state_msg=lambda v, name: (
                f"Displaying name in chat for {name} is now {'enabled' if v else 'disabled'}."
            ),
        ),
    ),
    PersonaField(
        name='execution_mode',
        describe=lambda p: f"Execution mode for '{p.get_name()}' is set to {p.get_execution_mode().name}.",
        set_cli=_enum_setter(
            enum_cls=ExecutionMode,
            apply=lambda p, v: p.set_execution_mode(v),
            label_title="Execution mode",
            label_lower="execution mode",
            label_article="an execution mode",
        ),
    ),
    PersonaField(
        name='memory_mode',
        describe=lambda p: (
            f"Memory mode for '{p.get_name()}' is {p.get_memory_mode().name.lower()}.\n"
            f"Valid modes are: {', '.join([e.name.lower() for e in MemoryMode])}."
        ),
        set_cli=_enum_setter(
            enum_cls=MemoryMode,
            apply=lambda p, v: p.set_memory_mode(v),
            label_title="Memory mode",
            label_lower="memory mode",
            label_article="a memory mode",
        ),
        patch_key='memory_mode',
        patch_apply=_patch_memory_mode,
    ),
    PersonaField(
        name='service_bindings',
        describe=lambda p: (
            f"Service bindings for '{p.get_name()}': "
            f"{', '.join(p.get_service_bindings()) if p.get_service_bindings() else 'none'}."
        ),
        set_cli=_set_service_bindings,
    ),
    PersonaField(
        name='long_term_memory',
        describe=lambda p: (
            f"Long-term memory retrieval for '{p.get_name()}' is "
            f"{'enabled' if p.get_long_term_memory() else 'disabled'}."
        ),
        set_cli=_bool_setter(
            apply=lambda p, v: p.set_long_term_memory(v),
            missing_msg="Error: Please specify 'on' or 'off' for long_term_memory.",
            state_msg=lambda v, name: (
                f"Long-term memory retrieval {'enabled' if v else 'disabled'} for {name}."
            ),
        ),
        patch_key='long_term_memory',
        patch_apply=_plain_patch(lambda p, v: p.set_long_term_memory(bool(v))),
    ),
    PersonaField(
        name='include_ambient_memory',
        describe=lambda p: (
            f"Ambient memory inclusion for '{p.get_name()}' is "
            f"{'enabled' if p.get_include_ambient_memory() else 'disabled'}."
        ),
        set_cli=_bool_setter(
            apply=lambda p, v: p.set_include_ambient_memory(v),
            missing_msg="Error: Please specify 'on' or 'off' for include_ambient_memory.",
            state_msg=lambda v, name: (
                f"Ambient memory inclusion {'enabled' if v else 'disabled'} for {name}."
            ),
        ),
    ),
    PersonaField(
        name='ingest_bank',
        describe=lambda p: (
            f"Ingest target bank for '{p.get_name()}' is "
            + (f"'{p.get_ingest_bank()}'" if p.get_ingest_bank() else f"default ('{p.get_name()}')")
            + "."
        ),
    ),
    PersonaField(
        name='thinking_level',
        describe=lambda p: (
            f"Thinking level for '{p.get_name()}' is "
            + (f"'{p.get_thinking_level()}'" if p.get_thinking_level() else "not set (default)")
            + "."
        ),
        set_cli=_set_thinking_level,
    ),
    PersonaField(
        name='max_context_tokens',
        describe=lambda p: f"Max context tokens for '{p.get_name()}' is {p.get_max_context_tokens()}.",
        set_cli=_set_max_context_tokens,
        patch_key='max_context_tokens',
        patch_apply=_plain_patch(lambda p, v: p.set_max_context_tokens(v)),
    ),
    PersonaField(
        name='chat_template',
        describe=lambda p: (
            f"Chat template for '{p.get_name()}' is "
            + (f"'{p.get_chat_template()}'" if p.get_chat_template() else "not set (default)")
            + "."
        ),
        set_cli=_set_chat_template,
        patch_key='chat_template',
        patch_apply=_plain_patch(lambda p, v: p.set_chat_template(v)),
    ),
    PersonaField(
        name='tool_policy',
        describe=lambda p: (
            f"Tool policy for '{p.get_name()}': {json.dumps(p.get_tool_policy().to_dict(), indent=2)}"
        ),
        set_cli=_set_tool_policy,
    ),
]


def cli_what_handlers() -> Dict[str, SetFn]:
    """`what` dispatch entries for every describable registry field, in the
    (args, persona) -> (response, mutated) shape BotLogic handlers use."""
    table: Dict[str, SetFn] = {}
    for f in PERSONA_FIELDS:
        if f.describe is None:
            continue
        describe = f.describe

        def what(args: List[str], persona: Persona,
                 _describe: WhatFn = describe) -> Tuple[Optional[str], bool]:
            return _describe(persona), False
        table[f.name] = what
    return table


def cli_set_handlers() -> Dict[str, SetFn]:
    """`set` dispatch entries for every registry field with a CLI setter."""
    return {f.name: f.set_cli for f in PERSONA_FIELDS if f.set_cli is not None}


def patchable_fields() -> List[PersonaField]:
    return [f for f in PERSONA_FIELDS if f.patch_key is not None]


def apply_patch_fields(persona: Persona, data: Dict[str, Any], rejected: List[str]) -> None:
    """Apply every registry-managed key present in a PATCH body.

    Rejections (values the Persona setter coerced away or refused) are
    appended to `rejected`. Route-specific keys (the history_messages/
    context_length pair, instruct_tags, kobold sampler extras) are handled
    by the caller.
    """
    for f in patchable_fields():
        if f.patch_key in data and f.patch_apply is not None:
            if f.patch_apply(persona, data[f.patch_key]):
                rejected.append(f.patch_key)


def registry_patch_keys() -> Set[str]:
    """JSON keys on the PATCH route owned by the registry."""
    return {f.patch_key for f in PERSONA_FIELDS if f.patch_key is not None}
