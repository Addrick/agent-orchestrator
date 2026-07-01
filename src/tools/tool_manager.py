# src/tools/tool_manager.py

import asyncio
import logging
from typing import Any, Coroutine, Dict, List, Callable, Optional, cast

from src.tools.definitions import get_all_tool_definitions

logger = logging.getLogger(__name__)


class ToolManager:
    """
    Generic registry for tool implementations.

    Tool handlers register their async callables via `register()`.
    The manager provides a unified interface for definition listing and execution.
    Only definitions for registered tools (plus non-callable flags like google_grounding)
    are exposed to the LLM.
    """

    def __init__(self) -> None:
        self._handlers: Dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {}
        self._enrichers: Dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {}

    def register(self, name: str, handler: Callable[..., Coroutine[Any, Any, Any]], 
                 enricher: Optional[Callable[..., Coroutine[Any, Any, Any]]] = None) -> None:
        """Register an async handler for a named tool, and optionally an enrichment handler."""
        self._handlers[name] = handler
        if enricher:
            self._enrichers[name] = enricher

    def unregister(self, name: str) -> None:
        """Remove a tool handler (and its enricher). Used by runtime-registered
        tools (DP-268 MCP) when their server is removed; unknown names are a
        no-op."""
        self._handlers.pop(name, None)
        self._enrichers.pop(name, None)

    async def enrich_audit_action(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[str]:
        """Returns a human-readable enrichment string for a tool call if an enricher is registered."""
        if tool_name in self._enrichers:
            try:
                return cast(Optional[str], await self._enrichers[tool_name](**arguments))
            except Exception as e:
                logger.warning(f"Enrichment failed for {tool_name}: {e}")
        return None

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Returns definitions for registered tools plus non-callable tool flags."""
        return [
            t for t in get_all_tool_definitions()
            if t.get('function', {}).get('name') in self._handlers
            or t.get('type') != 'function'
        ]

    async def execute_tool(self, tool_name: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Executes a registered tool by name.

        Returns:
            A dictionary containing either the 'result' or an 'error' message.
        """
        if tool_name not in self._handlers:
            return {"error": f"Tool '{tool_name}' not found."}

        try:
            result = await self._handlers[tool_name](**kwargs)
            return {"result": result}
        except Exception as e:
            logger.error(f"Error executing tool '{tool_name}' with args {kwargs}: {e}", exc_info=True)
            return {"error": f"An unexpected error occurred while executing {tool_name}: {str(e)}"}


class ZammadToolHandler:
    """Registers all Zammad CRUD tools with a ToolManager."""

    def __init__(self, zammad_client: Any) -> None:
        self.zammad_client = zammad_client

    def register(self, manager: ToolManager) -> None:
        manager.register("get_ticket_details", self._get_ticket_details)
        manager.register("update_ticket", self._update_ticket, self._enrich_ticket_action)
        manager.register("add_note_to_ticket", self._add_note_to_ticket, self._enrich_ticket_action)
        manager.register("search_tickets", self._search_tickets)
        manager.register("create_ticket", self._create_ticket)
        manager.register("search_user", self._search_user)
        manager.register("create_user", self._create_user)
        manager.register("update_user", self._update_user)
        manager.register("delete_user", self._delete_user)
        manager.register("merge_tickets", self._merge_tickets, self._enrich_merge_action)

    async def _enrich_ticket_action(self, **kwargs: Any) -> Optional[str]:
        """Fetches ticket number and title for enrichment."""
        ticket_id = kwargs.get("ticket_id")
        if not ticket_id:
            return None
        try:
            ticket = await asyncio.to_thread(self.zammad_client.get_ticket, ticket_id=ticket_id)
            return f"#{ticket.get('number')} ({ticket.get('title')})"
        except Exception:
            return None

    async def _enrich_merge_action(self, **kwargs: Any) -> Optional[str]:
        """Fetches ticket info for merge action enrichment."""
        source_id = kwargs.get("source_ticket_id")
        target_id = kwargs.get("target_ticket_id")
        if not source_id or not target_id:
            return None
        try:
            source = await asyncio.to_thread(self.zammad_client.get_ticket, ticket_id=source_id)
            target = await asyncio.to_thread(self.zammad_client.get_ticket, ticket_id=target_id)
            return f"Merge #{source.get('number')} into #{target.get('number')} ('{target.get('title')}')"
        except Exception:
            return None

    async def _get_ticket_details(self, ticket_number: int) -> Dict[str, Any]:
        """Translates user-facing ticket number to internal ID, then fetches complete details (articles + customer)."""
        logger.info(f"Executing tool: get_ticket_details for ticket_number={ticket_number}")
        search_results = await asyncio.to_thread(
            self.zammad_client.search_tickets, query=f"number:{ticket_number}"
        )
        if not search_results:
            raise ValueError(f"No ticket found with number {ticket_number}.")

        ticket: Dict[str, Any] = search_results[0]
        ticket_id = ticket['id']
        
        # 1. Fetch Articles - Prune to prevent context overflow
        articles = await asyncio.to_thread(self.zammad_client.get_ticket_articles, ticket_id=ticket_id)
        # Only keep necessary fields and limit count
        pruned_articles = []
        for art in articles[-10:]: # Last 10 articles
            pruned_articles.append({
                'id': art.get('id'),
                'from': art.get('from'),
                'to': art.get('to'),
                'subject': art.get('subject'),
                'body': art.get('body', '')[:1000], # Limit body length per article
                'created_at': art.get('created_at'),
                'internal': art.get('internal')
            })
        ticket['articles'] = pruned_articles

        # 2. Fetch Customer Info
        customer_id = ticket.get('customer_id')
        if customer_id:
            customer = await asyncio.to_thread(self.zammad_client.get_user, user_id=customer_id)
            ticket['customer_info'] = customer

        # 3. Fetch Tags
        tags = await asyncio.to_thread(self.zammad_client.get_tags, ticket_id=ticket_id)
        ticket['tags'] = tags

        # 4. Ensure human-readable state
        state_map = {1: 'new', 2: 'open', 3: 'pending reminder', 4: 'closed', 7: 'merged'}
        if 'state_id' in ticket:
            ticket['state'] = state_map.get(ticket['state_id'], 'unknown')

        return ticket

    async def _update_ticket(self, ticket_id: int, **kwargs: Any) -> Dict[str, Any]:
        logger.info(f"Executing tool: update_ticket on ticket_id={ticket_id} with args={kwargs}")
        payload: Dict[str, Any] = {}
        valid_args = ["state", "priority", "owner_id"]
        
        # 1. Handle Tags via Tags API (Delta Update)
        if "tags" in kwargs:
            requested_tags = set(kwargs["tags"]) if isinstance(kwargs["tags"], list) else set()
            current_tags = set(await asyncio.to_thread(self.zammad_client.get_tags, ticket_id=ticket_id))
            
            tags_to_add = requested_tags - current_tags
            tags_to_remove = current_tags - requested_tags
            
            for tag in tags_to_add:
                await asyncio.to_thread(self.zammad_client.add_tag, ticket_id=ticket_id, tag=tag)
            for tag in tags_to_remove:
                await asyncio.to_thread(self.zammad_client.remove_tag, ticket_id=ticket_id, tag=tag)

        # 2. Handle other fields via Ticket API
        for key, value in kwargs.items():
            if key in valid_args:
                payload[key] = value

        if not payload:
            if "tags" in kwargs:
                # Only tags were updated, return the current ticket state
                return await asyncio.to_thread(self.zammad_client.get_ticket, ticket_id=ticket_id)
            raise ValueError("No valid update parameters provided for update_ticket.")

        return await asyncio.to_thread(
            self.zammad_client.update_ticket, ticket_id=ticket_id, payload=payload
        )

    async def _add_note_to_ticket(self, ticket_id: int, body: str, internal: bool = False) -> Dict[str, Any]:
        logger.info(f"Executing tool: add_note_to_ticket on ticket_id={ticket_id}")
        return await asyncio.to_thread(
            self.zammad_client.add_article_to_ticket,
            ticket_id=ticket_id, body=body, internal=internal
        )

    async def _search_tickets(self, query: str) -> List[Dict[str, Any]]:
        logger.info(f"Executing tool: search_tickets with query='{query}'")
        results = await asyncio.to_thread(
            self.zammad_client.search_tickets, query=query
        )
        # Map state_id to human-readable 'state' for the LLM
        state_map = {1: 'new', 2: 'open', 3: 'pending reminder', 4: 'closed', 7: 'merged'}
        for t in results:
            if 'state_id' in t:
                t['state'] = state_map.get(t['state_id'], 'unknown')
        return cast(List[Dict[str, Any]], results)

    async def _create_ticket(self, title: str, body: str, customer_id: Optional[int] = None) -> Dict[str, Any]:
        logger.info(f"Executing tool: create_ticket with title='{title}' for customer_id={customer_id}")
        if not customer_id:
            raise ValueError("create_ticket requires a customer_id, which was not provided by the system.")
        return await asyncio.to_thread(
            self.zammad_client.create_ticket,
            title=title, group='Users', customer_id=customer_id, article_body=body
        )

    async def _search_user(self, query: str) -> List[Dict[str, Any]]:
        logger.info(f"Executing tool: search_user with query='{query}'")
        return await asyncio.to_thread(
            self.zammad_client.search_user, query=query
        )

    async def _create_user(self, firstname: str, lastname: str, email: str,
                           note: Optional[str] = None) -> Dict[str, Any]:
        logger.info(f"Executing tool: create_user with email='{email}'")
        return await asyncio.to_thread(
            self.zammad_client.create_user,
            firstname=firstname, lastname=lastname, email=email, note=note
        )

    async def _update_user(self, user_id: int, **kwargs: Any) -> Dict[str, Any]:
        logger.info(f"Executing tool: update_user on user_id={user_id} with args={kwargs}")
        valid_args = ["firstname", "lastname", "email", "active", "note"]
        payload = {k: v for k, v in kwargs.items() if k in valid_args}
        if not payload:
            raise ValueError("No valid update parameters provided for update_user.")
        return await asyncio.to_thread(
            self.zammad_client.update_user, user_id=user_id, payload=payload
        )

    async def _delete_user(self, user_id: int) -> Dict[str, str]:
        logger.info(f"Executing tool: delete_user on user_id={user_id}")
        await asyncio.to_thread(self.zammad_client.delete_user, user_id=user_id)
        return {"status": "success", "message": f"User {user_id} deleted."}

    async def _merge_tickets(self, source_ticket_id: int, target_ticket_id: int) -> Dict[str, Any]:
        """Merges one ticket into another by moving articles, linking them, and closing the source."""
        logger.info(f"Executing tool: merge_tickets (source={source_ticket_id} -> target={target_ticket_id})")
        return await asyncio.to_thread(
            self.zammad_client.merge_tickets,
            source_ticket_id=source_ticket_id,
            target_ticket_id=target_ticket_id
        )


class WebSearchHandler:
    """Registers web search tools with a ToolManager."""

    def register(self, manager: ToolManager) -> None:
        manager.register("web_search", self._web_search)

    async def _web_search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        logger.info(f"Executing tool: web_search with query='{query}', max_results={max_results}")
        from ddgs import DDGS

        def _sync_search() -> List[Dict[str, Any]]:
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=max_results))

        raw = await asyncio.to_thread(_sync_search)
        return [{"title": r["title"], "url": r["href"], "summary": r["body"]} for r in raw]


class MemoryRecallHandler:
    """Wraps `MemoryBackend.recall` as the model-callable `recall_memory` tool.

    The tool's tag scope is inherited from the active turn via `turn_context`
    — the LLM can't redirect recall to another persona / channel. Returns
    structured hits the engine LLM integrates into its response. The
    `produces_untrusted` capability flag means hits taint the turn per the
    tool-security framework.
    """

    def __init__(self, memory_backend: Any) -> None:
        self.memory_backend = memory_backend

    def register(self, manager: ToolManager) -> None:
        manager.register("recall_memory", self._recall_memory)

    async def _recall_memory(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        from src.tools.turn_context import get_turn_context
        ctx = get_turn_context()
        if ctx is None:
            logger.warning("recall_memory invoked without an active turn context.")
            return []
        tag_filter: List[str] = [
            f"channel:{ctx.channel}",
            f"user:{ctx.user_identifier}",
        ]
        if ctx.server_id:
            tag_filter.append(f"server:{ctx.server_id}")

        logger.info(
            f"Executing tool: recall_memory query='{query}' limit={limit} "
            f"persona={ctx.persona_name}"
        )
        hits = await self.memory_backend.recall(
            bank_id=ctx.persona_name,
            query=query,
            k=limit,
            tag_filter=tag_filter,
        )
        return [
            {
                "id": h.id,
                "content": h.content,
                "score": h.score,
                "untrusted": h.untrusted,
                "tags": h.tags,
                "timestamp": h.timestamp.isoformat() if h.timestamp else None,
            }
            for h in hits
        ]


class MemoryToolHandler:
    def __init__(self, memory_manager: Any, embedding_service: Any = None) -> None:
        self.memory_manager = memory_manager
        self.embedding_service = embedding_service

    def register(self, manager: ToolManager) -> None:
        backend = getattr(self.memory_manager, "backend", None)
        if backend and backend.__class__.__name__ == "SqliteSemanticBackend":
            manager.register("drill_down_memory", self._drill_down_memory)
            manager.register("update_core_memory", self._update_core_memory)

    async def _drill_down_memory(self, parent_summary_id: int) -> List[Dict[str, Any]]:
        logger.info(f"Executing drill_down_memory for parent {parent_summary_id}")
        with self.memory_manager._lock:
            conn = self.memory_manager._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT content, created_at, untrusted FROM Memory_Summaries WHERE parent_summary_id = ? ORDER BY created_at ASC",
                (parent_summary_id,)
            )
            return [dict(row) for row in cursor.fetchall()]

    async def _update_core_memory(self, summary_id: int, new_content: str) -> Dict[str, Any]:
        logger.info(f"Executing update_core_memory for ID {summary_id}")
        embedding = None
        if self.embedding_service:
            embedding = await self.embedding_service.encode_single(new_content)
        
        with self.memory_manager.transaction() as conn:
            conn.execute("UPDATE Memory_Summaries SET content = ? WHERE summary_id = ?", (new_content, summary_id))
            if embedding is not None:
                conn.execute("UPDATE Memory_Summaries SET embedding = ? WHERE summary_id = ?", (embedding, summary_id))
                conn.execute(
                    "INSERT OR REPLACE INTO vec_Memory_Summaries (summary_id, embedding) VALUES (?, ?)", 
                    (summary_id, embedding)
                )
        return {"status": "success", "message": f"Core Profile {summary_id} updated."}

