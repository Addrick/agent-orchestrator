# src/memory/backend/base.py
"""MemoryBackend ABC + dataclasses.

The ABC carries two method sets:

1. **Legacy SQLite-shape methods** (`store_segment`, `retrieve_relevant_summaries`, etc.).
   These are the contract every current caller of MemoryManager uses today.
   They will be removed once all callers migrate to the new-shape methods.

2. **New Hindsight-shape methods** (`retain_turn`, `recall`, `reflect`, ...).
   These are placeholders in Sprint 1 (DP-108). Sprint 2 lands the real
   implementations on HindsightBackend; SQLite keeps them as NotImplementedError
   stubs (or noops, where the plan specifies — `reflect`, `list_mental_models`).

Carrying both on the ABC lets MemoryManager.delegate the legacy methods through
`self.backend` without coupling to a concrete subclass, while still defining the
forward-looking surface that Hindsight will implement.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


# ----------------------------- New-shape dataclasses ----------------------------- #


@dataclass
class MemoryHit:
    """A single recall result (new-shape)."""
    id: str
    content: str
    score: float
    untrusted: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    timestamp: Optional[datetime] = None


@dataclass
class Experience:
    """An episodic memory unit (action + outcome)."""
    id: str
    action_type: str
    context: Dict[str, Any]
    outcome: Optional[str]
    score: float = 0.0
    timestamp: Optional[datetime] = None


@dataclass
class MentalModel:
    """A consolidated mental model produced by reflect()."""
    id: str
    content: str
    tags: List[str] = field(default_factory=list)


@dataclass
class ReflectResult:
    """Result of a reflect() call."""
    answer: str
    mental_models: List[MentalModel] = field(default_factory=list)


# ----------------------------- ABC ----------------------------- #


class MemoryBackend(ABC):
    """Pluggable backend for the semantic + episodic memory tier.

    Implementations:
      - `SqliteSemanticBackend` — wraps existing MemoryManager logic (Sprint 1).
      - `HindsightBackend` — vectorize-io/hindsight via hindsight-client (Sprint 2).
    """

    # ===== Legacy SQLite-shape: episodic =====

    @abstractmethod
    def log_agent_action(
        self,
        agent_name: str,
        action_type: str,
        trigger_context: Optional[str] = None,
        action_payload: Optional[str] = None,
        outcome: Optional[str] = None,
        outcome_payload: Optional[str] = None,
        parent_id: Optional[int] = None,
    ) -> int: ...

    @abstractmethod
    def update_agent_action_outcome(
        self,
        action_id: int,
        outcome: str,
        outcome_payload: Optional[str] = None,
    ) -> None: ...

    @abstractmethod
    def add_action_contexts(
        self,
        action_id: int,
        contexts: List[Tuple[str, str]],
    ) -> None: ...

    @abstractmethod
    def get_relevant_agent_actions(
        self,
        agent_name: str,
        match_contexts: Optional[List[Tuple[str, str]]] = None,
        match_types: Optional[List[str]] = None,
        limit: int = 15,
    ) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def get_action_steps(self, parent_id: int) -> List[Dict[str, Any]]: ...

    # ===== Legacy SQLite-shape: semantic =====

    @abstractmethod
    def store_message_embedding(
        self,
        interaction_id: int,
        embedding: bytes,
        model_name: str,
        created_at: datetime,
    ) -> None: ...

    @abstractmethod
    def get_unembedded_messages(
        self,
        persona_name: str,
        channel: str,
        server_id: Optional[str] = None,
        limit: int = 50,
        model_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def store_segment(
        self,
        channel: str,
        server_id: Optional[str],
        persona_name: str,
        start_id: int,
        end_id: int,
        message_count: int,
        created_at: datetime,
    ) -> int: ...

    @abstractmethod
    def store_summary(
        self,
        segment_id: int,
        content: str,
        embedding: bytes,
        model_name: str,
        created_at: datetime,
        summary_level: Optional[int] = None,
        parent_summary_id: Optional[int] = None,
        untrusted: bool = False,
    ) -> int: ...

    @abstractmethod
    def get_summaries_for_channel(
        self,
        channel: str,
        persona_name: str,
        server_id: Optional[str] = None,
        exclude_after_interaction_id: Optional[int] = None,
        model_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def get_unsegmented_embedded_messages(
        self,
        persona_name: str,
        channel: str,
        server_id: Optional[str] = None,
        model_name: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def retrieve_relevant_summaries(
        self,
        persona_name: str,
        channel: str,
        server_id: Optional[str] = None,
        user_identifier: Optional[str] = None,
        memory_mode: str = "channel",
        include_ambient: bool = True,
        exclude_after_interaction_id: Optional[int] = None,
        model_name: Optional[str] = None,
        query_embeddings: Optional[List[bytes]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def record_segment_failure(
        self,
        channel: str,
        server_id: Optional[str],
        persona_name: str,
        start_id: int,
        end_id: int,
        message_count: int,
        error_reason: Optional[str] = None,
    ) -> None: ...

    @abstractmethod
    def get_failed_segment_ranges(
        self,
        channel: str,
        persona_name: str,
        server_id: Optional[str] = None,
        max_attempts: int = 3,
        cooldown_hours: float = 24.0,
    ) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def clear_segment_failure(
        self,
        channel: str,
        persona_name: str,
        server_id: Optional[str],
        start_id: int,
        end_id: int,
    ) -> None: ...

    @abstractmethod
    def get_active_channels(
        self,
        model_name: Optional[str] = None,
    ) -> List[Tuple[str, str, Optional[str]]]: ...

    @abstractmethod
    def get_last_segment_tail_embeddings(
        self,
        channel: str,
        persona_name: str,
        server_id: Optional[str] = None,
        n: int = 3,
        model_name: Optional[str] = None,
    ) -> Optional[List[bytes]]: ...

    # ===== New Hindsight-shape (Sprint 2 lands real impls) =====

    async def retain_turn(
        self,
        bank_id: str,
        role: str,
        content: str,
        *,
        timestamp: datetime,
        scope_tags: List[str],
        source_persona: str,
        untrusted: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Retain a conversation turn in the semantic store (new-shape).

        `untrusted` is part of the security framework contract — backends MUST
        persist it and surface it on subsequent recall hits. See
        plans/tool_security_framework.md +
        decisions/2026-05-03-minimum-viable-tool-security.md.

        Sprint 1 stub — Hindsight backend (Sprint 2) implements; SQLite raises.
        """
        raise NotImplementedError("retain_turn lands in Sprint 2 (HindsightBackend)")

    async def mark_trusted(
        self,
        bank_id: str,
        hit_id: str,
        *,
        operator_id: str,
        reason: str,
    ) -> None:
        """Flip the untrusted bit OFF on a stored unit.

        Operator override path — see security plan. Default raises so backends
        must opt in.
        """
        raise NotImplementedError("mark_trusted not implemented on this backend")

    async def mark_untrusted(
        self,
        bank_id: str,
        hit_id: str,
        *,
        operator_id: str,
        reason: str,
    ) -> None:
        """Flip the untrusted bit ON on a stored unit."""
        raise NotImplementedError("mark_untrusted not implemented on this backend")

    async def retain_experience(
        self,
        bank_id: str,
        action_type: str,
        context: Dict[str, Any],
        outcome: Optional[str],
        *,
        scope_tags: List[str],
        source_persona: str,
        untrusted: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Retain an episodic action+outcome record (new-shape).

        Sprint 1 stub — Hindsight backend (Sprint 2) implements; SQLite raises.
        """
        raise NotImplementedError("retain_experience lands in Sprint 2 (HindsightBackend)")

    async def recall(
        self,
        bank_id: str,
        query: str,
        *,
        k: int = 10,
        types: Optional[List[str]] = None,
        tag_filter: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
        budget: Optional[str] = None,
    ) -> List[MemoryHit]:
        """Semantic recall (new-shape).

        Sprint 1 stub — Hindsight backend (Sprint 2) implements; SQLite raises.
        """
        raise NotImplementedError("recall lands in Sprint 2 (HindsightBackend)")

    async def recall_experiences(
        self,
        bank_id: str,
        query: str,
        *,
        match_contexts: Optional[List[Tuple[str, str]]] = None,
        k: int = 10,
    ) -> List[Experience]:
        """Episodic recall (new-shape).

        Sprint 1 stub — Hindsight backend (Sprint 2) implements; SQLite raises.
        """
        raise NotImplementedError("recall_experiences lands in Sprint 2 (HindsightBackend)")

    async def reflect(
        self,
        bank_id: str,
        query: str,
        *,
        tag_filter: Optional[List[str]] = None,
    ) -> ReflectResult:
        """Run consolidation/reflection (new-shape).

        SQLite is a noop per plan (consolidation runs out-of-band in
        memory_consolidation.py). Hindsight invokes its reflect endpoint.
        """
        return ReflectResult(answer="", mental_models=[])

    async def list_mental_models(
        self,
        bank_id: str,
        *,
        tags: Optional[List[str]] = None,
    ) -> List[MentalModel]:
        """List consolidated mental models (new-shape).

        SQLite is a noop per plan (no mental-model store). Hindsight returns
        the `mental_models` table.
        """
        return []

    async def ensure_bank(
        self,
        bank_id: str,
        *,
        retain_mission: Optional[str] = None,
        reflect_mission: Optional[str] = None,
        enable_observations: Optional[bool] = None,
        observations_mission: Optional[str] = None,
    ) -> None:
        """Ensure a bank exists (new-shape).

        `retain_mission` steers extraction; `reflect_mission` steers reflect
        output. Upstream's deprecated `mission` alias is intentionally not
        accepted — it silently overwrites reflect_mission server-side.
        SQLite noop — banks are implicit in persona/channel scoping.
        """
        return None

    async def delete_bank(self, bank_id: str) -> None:
        """Delete a bank (new-shape).

        Sprint 1 stub — Hindsight backend (Sprint 2) implements; SQLite raises.
        """
        raise NotImplementedError("delete_bank lands in Sprint 2 (HindsightBackend)")
