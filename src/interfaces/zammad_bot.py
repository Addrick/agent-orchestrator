# src/interfaces/zammad_bot.py

import asyncio
import logging
from typing import Optional, List, Dict, Any

from config.global_config import ZAMMAD_POLL_INTERVAL, ZAMMAD_TRIAGE_TAG, TRIAGE_PERSONA_NAME
from src.chat_system import ChatSystem
from src.persona import Persona

logger = logging.getLogger(__name__)


class ZammadBot:
    def __init__(self, chat_system: ChatSystem):
        self.chat_system = chat_system
        self.zammad_client = chat_system.zammad_client
        self._shutdown_event = asyncio.Event()

    async def start(self) -> None:
        """Starts the polling loop."""
        logger.info("Zammad Bot started. Polling for new tickets...")
        while not self._shutdown_event.is_set():
            try:
                await self._poll()
            except Exception as e:
                logger.error(f"Error in Zammad polling loop: {e}", exc_info=True)

            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=ZAMMAD_POLL_INTERVAL)
            except asyncio.TimeoutError:
                continue  # Timeout reached, loop again

    def stop(self) -> None:
        """Signals the bot to stop polling."""
        self._shutdown_event.set()

    async def _poll(self) -> None:
        """Checks for new, untagged tickets."""
        # Search for new tickets that haven't been triaged yet
        query = f"state.name:new AND NOT tags:{ZAMMAD_TRIAGE_TAG}"
        try:
            new_tickets = await asyncio.to_thread(
                self.zammad_client.search_tickets, query=query, limit=10
            )
        except Exception as e:
            logger.error(f"Failed to search Zammad tickets: {e}")
            return

        for ticket in new_tickets:
            if self._shutdown_event.is_set():
                break
            await self._process_ticket(ticket['id'])

    async def _get_search_keywords(self, title: str, body: str) -> str:
        """
        Uses the Local LLM to extract technical search keywords from the ticket content.
        """
        prompt = (
            f"Analyze the following support ticket content. "
            f"Extract 3 distinct, technical search keywords or error codes that would be best for finding similar solved tickets. "
            f"Output ONLY the keywords separated by spaces. Do not include punctuation or filler words.\n\n"
            f"Title: {title}\n"
            f"Body: {body[:500]}"  # Limit body to first 500 chars to save local compute
        )

        try:
            # Use a temporary config for the local model
            local_config = {
                "model_name": "local",
                "max_output_tokens": 50,
                "temperature": 0.1
            }

            response, _ = await self.chat_system.text_engine.generate_response(
                persona_config=local_config,
                context_object={
                    "persona_prompt": "You are a keyword extraction tool.",
                    "history": [],
                    "current_message": {"text": prompt, "image_url": None}
                },
                tools=None
            )

            if response.get('type') == 'text':
                keywords = response.get('content', '').strip()
                # Basic sanitization: remove newlines, keep it simple
                keywords = " ".join(keywords.split())
                logger.info(f"Local LLM extracted keywords: '{keywords}'")
                return keywords

        except Exception as e:
            logger.warning(f"Local LLM keyword extraction failed: {e}. Falling back to title.")

        return title  # Fallback to title if local LLM fails

    async def _process_ticket(self, ticket_id: int) -> None:
        """
        Performs the AI triage on a single ticket using Hybrid Context.
        1. Local LLM -> Keywords
        2. Python -> Global Search (using keywords)
        3. Python -> User History
        4. Main LLM -> Analysis
        """
        logger.info(f"Processing ticket {ticket_id} for AI triage...")
        try:
            # 1. Fetch full ticket details
            ticket = await asyncio.to_thread(self.zammad_client.get_ticket, ticket_id=ticket_id)
            customer_id = ticket.get('customer_id')
            title = ticket.get('title', 'No Title')

            # Fetch the first article to get the body content
            articles = await asyncio.to_thread(self.zammad_client.get_ticket_articles, ticket_id=ticket_id)
            first_article_body = "No content"
            if articles:
                first_article_body = articles[0].get('body', 'No content')

            # 2. Keyword Scout (Local LLM)
            search_keywords = await self._get_search_keywords(title, first_article_body)

            # 3. Global Solution Search (The "What")
            keyword_list = search_keywords.split()
            global_query = ""
            if keyword_list:
                conditions = []
                for kw in keyword_list:
                    clean_kw = kw.replace('"', '').replace("'", "")
                    conditions.append(f'title:"{clean_kw}"')
                    conditions.append(f'body:"{clean_kw}"')

                or_clause = " OR ".join(conditions)
                global_query = f"({or_clause}) AND state.name:closed"
            else:
                global_query = f'title:"{title}" AND state.name:closed'

            global_tickets = await asyncio.to_thread(
                self.zammad_client.search_tickets, query=global_query, limit=3, sort_by='updated_at', order_by='desc'
            )

            global_context_parts = []
            if global_tickets:
                for t in global_tickets:
                    try:
                        t_articles = await asyncio.to_thread(self.zammad_client.get_ticket_articles, ticket_id=t['id'])
                        if t_articles:
                            t_body = t_articles[-1].get('body', 'No content')
                            t_body_snippet = (t_body[:300] + '...') if len(t_body) > 300 else t_body
                            # CHANGED: (ID: {t['id']}) -> (Ticket #{t['id']})
                            global_context_parts.append(
                                f"- [Global Match] {t.get('title', 'No Title')} (Ticket #{t['id']})\n  Latest Note: {t_body_snippet}")
                        else:
                            global_context_parts.append(
                                f"- [Global Match] {t.get('title', 'No Title')} (Ticket #{t['id']})")
                    except Exception as e:
                        logger.warning(f"Failed to fetch articles for global ticket {t['id']}: {e}")
                        global_context_parts.append(
                            f"- [Global Match] {t.get('title', 'No Title')} (Ticket #{t['id']})")

                global_context = "\n".join(global_context_parts)
            else:
                global_context = "No similar closed tickets found globally."

            # 4. User History Search (The "Who")
            history_query = f"customer_id:{customer_id} AND state.name:closed"
            history_tickets = await asyncio.to_thread(
                self.zammad_client.search_tickets, query=history_query, limit=5, sort_by='updated_at', order_by='desc'
            )

            user_history_text = ""
            if history_tickets:
                # CHANGED: Added (Ticket #{t['id']}) to user history as well
                user_history_text = "\n".join(
                    [f"- {t.get('updated_at', 'N/A')}: {t.get('title', 'No Title')} (Ticket #{t['id']})" for t in
                     history_tickets])
            else:
                user_history_text = "No previous closed tickets found for this user."

            # 5. Prepare Main LLM Context
            persona = self.chat_system.personas.get(TRIAGE_PERSONA_NAME)
            if not persona:
                persona = Persona(
                    persona_name=TRIAGE_PERSONA_NAME,
                    model_name="gpt-4o",
                    prompt="You are a support triage assistant. Analyze the ticket, user history, and similar past solutions.",
                    token_limit=600
                )

            context_message = (
                f"NEW TICKET DETAILS:\n"
                f"Title: {title}\n"
                f"Body: {first_article_body}\n\n"
                f"USER HISTORY (Context on the User):\n{user_history_text}\n\n"
                f"SIMILAR SOLVED TICKETS (Potential Solutions):\n{global_context}\n\n"
                f"INSTRUCTIONS:\n"
                f"1. Summarize the user's issue.\n"
                f"2. If the 'Similar Solved Tickets' seem relevant, suggest a solution based on them.\n"
                f"3. If the 'User History' shows a pattern of similar issues, note that.\n"
                f"4. Provide a concise internal note for the agent."
            )

            # 6. Call Main LLM (No Tools)
            response, _ = await self.chat_system.text_engine.generate_response(
                persona_config=persona.get_config_for_engine(),
                context_object={
                    "persona_prompt": persona.get_prompt(),
                    "history": [],
                    "current_message": {"text": context_message, "image_url": None}
                },
                tools=None
            )

            if response.get('type') == 'text':
                content = response.get('content', '')

                # 7. Post Internal Note with Plain Text Debug Dump
                final_note_body = (
                    f"{content}\n\n"
                    f"--------------------------------------------------\n"
                    f"[ AI TRIAGE CONTEXT DUMP ]\n\n"
                    f"EXTRACTED KEYWORDS: {search_keywords}\n"
                    f"GLOBAL SEARCH QUERY: {global_query}\n\n"
                    f"GLOBAL MATCHES FOUND:\n{global_context}\n\n"
                    f"USER HISTORY FOUND:\n{user_history_text}"
                )

                await asyncio.to_thread(
                    self.zammad_client.add_article_to_ticket,
                    ticket_id=ticket_id,
                    body=final_note_body,
                    internal=True
                )

                # 8. Tag Ticket
                current_tags = ticket.get('tags', [])
                if isinstance(current_tags, str):
                    current_tags = current_tags.split(',')

                if ZAMMAD_TRIAGE_TAG not in current_tags:
                    current_tags.append(ZAMMAD_TRIAGE_TAG)

                tags_str = ",".join(current_tags)

                await asyncio.to_thread(
                    self.zammad_client.update_ticket,
                    ticket_id=ticket_id,
                    payload={'tags': tags_str}
                )
                logger.info(f"Ticket {ticket_id} triaged successfully.")

        except Exception as e:
            logger.error(f"Error processing ticket {ticket_id}: {e}", exc_info=True)
