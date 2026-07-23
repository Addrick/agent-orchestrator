# src/agents/zammad_bot.py

import asyncio
import logging
from typing import Optional, Dict, Any

from config.global_config import (
    ZAMMAD_TRIAGE_TAG,
    TRIAGE_SCOUT_NAME,
    TRIAGE_SUMMARIZER_NAME,
    TRIAGE_ANALYST_NAME,
    TRIAGE_FILTER_NAME,
    TRIAGE_GLOBAL_HISTORY_COUNT,
    TRIAGE_USER_HISTORY_COUNT,
    TRIAGE_MAX_CONTEXT_CHARS,
    ZAMMAD_BOT_EMAIL,
    ZAMMAD_BOT_FIRSTNAME,
    ZAMMAD_BOT_LASTNAME,
    SECURITY_REPORT_TAG,
    PHISHING_SUSPECT_TAG,
    QUARANTINE_TAGS,
    PHISHING_SUSPECT_MIN_CONFIDENCE,
)
from src.agents.base import Agent
from src.chat_system import ChatSystem
from src.clients.zammad_client import ZammadClient

logger = logging.getLogger(__name__)


class ZammadBot(Agent):

    agent_name: str = "zammad_bot"

    def __init__(
        self,
        chat_system: ChatSystem,
        zammad_client: ZammadClient,
        content_classifier: Any,
    ) -> None:
        super().__init__(chat_system)
        self.zammad_client = zammad_client
        # Single-shot inference agent, injected by AgentManager convention-DI
        # (registered as CONTENT_CLASSIFIER_NAME). No longer constructed ad-hoc
        # from chat_system — see the DP-292 single-shot-agent decision.
        self.classifier = content_classifier

    async def _on_start(self) -> None:
        """
        Checks if the Zammad user for the bot exists.
        Logs instructions if missing, as manual setup is required for permissions.
        """
        email = ZAMMAD_BOT_EMAIL
        try:
            # Check if user exists
            users = await asyncio.to_thread(self.zammad_client.search_user, f"email:{email}")
            if not users:
                logger.error(
                    f"Zammad Bot User '{email}' NOT FOUND.\n"
                    f"Automatic creation is disabled to ensure correct permissions.\n"
                    f"ACTION REQUIRED: Please create this user manually in Zammad:\n"
                    f"  - Email: {email}\n"
                    f"  - Name: {ZAMMAD_BOT_FIRSTNAME} {ZAMMAD_BOT_LASTNAME}\n"
                    f"  - Roles: Agent (Required to write internal notes)\n"
                    f"  - Group Permissions: Read/Write access to relevant groups (e.g. Users)"
                )
            else:
                logger.info(f"Zammad Bot User found: {email} (ID: {users[0]['id']})")
        except Exception as e:
            logger.error(f"Failed to check bot identity in Zammad: {e}")

    async def deploy(self) -> None:
        """Checks for new, untagged tickets."""
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

    async def _get_search_keywords(self, title: str, body: str) -> Optional[str]:
        """
        Uses the 'triage_scout' persona to extract technical search keywords.
        """
        persona = self.chat_system.personas.get(TRIAGE_SCOUT_NAME)
        if not persona:
            logger.error(f"System persona '{TRIAGE_SCOUT_NAME}' not found. Skipping keyword extraction.")
            return None

        prompt = (
            f"Title: {title}\n"
            f"Body: {body[:1000]}"
        )

        try:
            response, _ = await self.chat_system.text_engine.generate_response(
                persona_config=persona.get_config_for_engine(),
                history_object=self._build_history_object(persona, prompt),
                tools=None
            )

            if response.get('type') == 'text':
                keywords = response.get('content', '').strip()
                keywords = " ".join(keywords.split())
                logger.info(f"Local LLM extracted keywords: '{keywords}'")
                return keywords

        except Exception as e:
            logger.warning(f"Local LLM keyword extraction failed: {e}. Skipping global search.")

        return None

    async def _summarize_text(self, text: str) -> str:
        """
        Uses the 'triage_summarizer' persona to summarize a ticket body.
        """
        if len(text) < 500:
            return text

        persona = self.chat_system.personas.get(TRIAGE_SUMMARIZER_NAME)
        if not persona:
            logger.error(f"System persona '{TRIAGE_SUMMARIZER_NAME}' not found. Returning original text.")
            return text[:500] + "... [Truncated: Summarizer Missing]"

        prompt = f"Content:\n{text[:4000]}"

        try:
            response, _ = await self.chat_system.text_engine.generate_response(
                persona_config=persona.get_config_for_engine(),
                history_object=self._build_history_object(persona, prompt),
                tools=None
            )
            if response.get('type') == 'text':
                return f"[SUMMARIZED]: {response.get('content', '').strip()}"
        except Exception as e:
            logger.warning(f"Local LLM summarization failed: {e}")

        return text[:500] + "... [Truncated due to error]"

    async def _check_relevance(self, new_text: str, history_text: str) -> bool:
        """
        Uses the 'triage_filter' persona to check if a historical ticket is relevant.
        """
        persona = self.chat_system.personas.get(TRIAGE_FILTER_NAME)
        if not persona:
            return True

        prompt = (
            f"New Ticket:\n{new_text[:500]}\n\n"
            f"Historical Ticket:\n{history_text[:500]}\n\n"
            f"Is the Historical Ticket relevant to the New Ticket? Respond with RELEVANT or IRRELEVANT."
        )

        try:
            response, _ = await self.chat_system.text_engine.generate_response(
                persona_config=persona.get_config_for_engine(),
                history_object=self._build_history_object(persona, prompt),
                tools=None
            )

            content = response.get('content', '').strip().upper()
            # FIX: Ensure IRRELEVANT doesn't trigger RELEVANT
            is_relevant = "RELEVANT" in content and "IRRELEVANT" not in content
            logger.debug(f"Relevance Check: {content} -> {is_relevant}")
            return is_relevant
        except Exception as e:
            logger.warning(f"Relevance check failed: {e}. Defaulting to Relevant.")
            return True

    def _smart_truncate(self, text: str, limit: int) -> str:
        """
        Truncates text to limit, preserving the first 20% and last 80% of the budget.
        """
        if len(text) <= limit:
            return text

        head_limit = int(limit * 0.2)
        tail_limit = int(limit * 0.8)

        if head_limit + tail_limit >= len(text):
            return text

        return f"{text[:head_limit]}\n\n... [TRUNCATED INTELLIGENTLY] ...\n\n{text[-tail_limit:]}"

    async def _post_internal_note(self, ticket_id: int, body: str) -> None:
        """Post an internal note as the bot identity, falling back to the
        API-token identity when impersonation is not permitted."""
        try:
            await asyncio.to_thread(
                self.zammad_client.add_article_to_ticket,
                ticket_id=ticket_id,
                body=body,
                internal=True,
                impersonate_email=ZAMMAD_BOT_EMAIL
            )
        except Exception as e:
            logger.warning(
                f"Failed to post note as {ZAMMAD_BOT_EMAIL}: {e}. Falling back to API token identity.")
            await asyncio.to_thread(
                self.zammad_client.add_article_to_ticket,
                ticket_id=ticket_id,
                body=body,
                internal=True
            )

    async def _classification_gate(self, ticket_id: int, title: str, body: str) -> bool:
        """Classify ticket content and quarantine phishing (DP-288 Phase 1).

        Returns True when the ticket was quarantined (caller must stop the
        triage pipeline — the content is bait and must not reach any further
        LLM prompt). Classification failure returns False: the pipeline
        proceeds unclassified, matching pre-DP-288 behavior, while the
        deterministic reporter-marker path inside the classifier keeps
        working even with the LLM down.
        """
        # A quarantine tag already on the ticket (operator-applied, or a
        # re-poll of a previously quarantined ticket) is always respected.
        try:
            existing_tags = await asyncio.to_thread(
                self.zammad_client.get_tags, ticket_id=ticket_id)
        except Exception as e:
            logger.warning(f"Could not fetch tags for ticket {ticket_id}: {e}")
            existing_tags = []
        already_tagged = [t for t in existing_tags if t in QUARANTINE_TAGS]
        if already_tagged:
            logger.info(
                f"Ticket {ticket_id} already quarantined ({already_tagged}); skipping triage.")
            await asyncio.to_thread(
                self.zammad_client.add_tag, ticket_id=ticket_id, tag=ZAMMAD_TRIAGE_TAG)
            return True

        classification = await self.classifier.classify(title, body)
        if classification is None:
            return False

        if classification.label == "phishing_report":
            quarantine_tag = SECURITY_REPORT_TAG
        elif (classification.label == "phishing_suspect"
              and classification.confidence >= PHISHING_SUSPECT_MIN_CONFIDENCE):
            quarantine_tag = PHISHING_SUSPECT_TAG
        else:
            # clean / spam / low-confidence suspect: proceed with normal
            # triage. A low-confidence suspect verdict still gets recorded
            # for the human agent, but does not block the pipeline.
            if classification.label == "phishing_suspect":
                await self._post_internal_note(
                    ticket_id,
                    f"[ AI CONTENT CLASSIFIER ]\n"
                    f"Verdict: phishing_suspect (LOW CONFIDENCE "
                    f"{classification.confidence:.2f} — not quarantined)\n"
                    f"Indicators: {', '.join(classification.indicators) or 'none given'}"
                )
            return False

        note = (
            f"[ AI CONTENT CLASSIFIER — QUARANTINED ]\n"
            f"Verdict: {classification.label} "
            f"(confidence {classification.confidence:.2f}, "
            f"source: {classification.source})\n"
            f"Indicators: {', '.join(classification.indicators) or 'none given'}\n\n"
            f"This ticket's content has been excluded from AI triage and "
            f"board planning. Handle as a security item. To re-run AI triage "
            f"after review: remove BOTH the '{quarantine_tag}' and "
            f"'{ZAMMAD_TRIAGE_TAG}' tags and set the ticket state to new."
        )
        await self._post_internal_note(ticket_id, note)
        await asyncio.to_thread(
            self.zammad_client.add_tag, ticket_id=ticket_id, tag=quarantine_tag)
        await asyncio.to_thread(
            self.zammad_client.add_tag, ticket_id=ticket_id, tag=ZAMMAD_TRIAGE_TAG)
        logger.info(
            f"Ticket {ticket_id} quarantined as {classification.label} "
            f"(tag '{quarantine_tag}'); triage pipeline skipped.")
        return True

    async def _process_ticket(self, ticket_id: int) -> None:
        """
        Adaptive Triage Pipeline using System Personas.
        """
        logger.info(f"Processing ticket {ticket_id} for AI triage...")

        analyst_persona = self.chat_system.personas.get(TRIAGE_ANALYST_NAME)
        if not analyst_persona:
            logger.error(f"System persona '{TRIAGE_ANALYST_NAME}' not found. Aborting triage for ticket {ticket_id}.")
            return

        try:
            # 1. Fetch New Ticket Data
            ticket = await asyncio.to_thread(self.zammad_client.get_ticket, ticket_id=ticket_id)
            customer_id = ticket.get('customer_id')
            title = ticket.get('title', 'No Title')
            articles = await asyncio.to_thread(self.zammad_client.get_ticket_articles, ticket_id=ticket_id)
            new_ticket_body = "\n---\n".join([a.get('body', '') for a in articles]) if articles else "No content"

            # 1b. Content classification gate (DP-288 Phase 1). Runs BEFORE
            # any triage persona sees the body: quarantined content must never
            # reach the scout/analyst prompts. A manually applied quarantine
            # tag is always respected without re-judging it.
            if await self._classification_gate(ticket_id, title, new_ticket_body):
                return

            # 2. Keyword Scout
            search_keywords = await self._get_search_keywords(title, new_ticket_body)

            # 3. Gather History (Global & User)
            global_tickets = []
            user_tickets = []

            if search_keywords:
                keyword_list = search_keywords.split()
                conditions = [f'title:"{k.replace("`", "")}"' for k in keyword_list] + \
                             [f'body:"{k.replace("`", "")}"' for k in keyword_list]
                global_query = f"({' OR '.join(conditions)}) AND state.name:closed"

                global_tickets = await asyncio.to_thread(
                    self.zammad_client.search_tickets, query=global_query, limit=TRIAGE_GLOBAL_HISTORY_COUNT
                )
            else:
                global_query = "N/A (No keywords)"

            if search_keywords:
                user_query = f"customer_id:{customer_id} AND ({' OR '.join(conditions)}) AND state.name:closed"
                user_tickets = await asyncio.to_thread(
                    self.zammad_client.search_tickets, query=user_query, limit=TRIAGE_USER_HISTORY_COUNT
                )
            else:
                user_query = f"customer_id:{customer_id} AND state.name:closed"
                user_tickets = await asyncio.to_thread(
                    self.zammad_client.search_tickets, query=user_query, limit=TRIAGE_USER_HISTORY_COUNT,
                    sort_by='updated_at', order_by='desc'
                )

            # 4. Fetch Full Bodies for History
            history_data = []

            async def fetch_body(t: Dict[str, Any], source_type: str) -> Optional[Dict[str, Any]]:
                try:
                    arts = await asyncio.to_thread(self.zammad_client.get_ticket_articles, ticket_id=t['id'])
                    body = "\n---\n".join([a.get('body', '') for a in arts]) if arts else "No content"
                    return {"id": t['id'], "title": t['title'], "body": body, "type": source_type}
                except Exception as e:
                    logger.warning(f"Failed to fetch articles for ticket {t['id']}: {e}")
                    return None

            tasks = [fetch_body(t, "Global") for t in global_tickets] + [fetch_body(t, "User") for t in user_tickets]
            results = await asyncio.gather(*tasks)
            history_data = [r for r in results if r is not None]

            # 5. Relevance Filtering (Interim Comprehension)
            for item in history_data:
                is_rel = await self._check_relevance(new_ticket_body, item['body'])
                item['is_relevant'] = is_rel
                logger.info(f"Ticket {item['id']} ({item['title']}) relevance: {is_rel}")

            # 6. Adaptive Compression Logic
            relevant_history = [h for h in history_data if h['is_relevant']]
            total_chars = len(new_ticket_body) + sum(len(h['body']) for h in relevant_history)

            if total_chars > TRIAGE_MAX_CONTEXT_CHARS:
                logger.info(
                    f"Context size {total_chars} exceeds limit {TRIAGE_MAX_CONTEXT_CHARS}. Engaging summarizers.")

                summary_tasks = [self._summarize_text(h['body']) for h in relevant_history]
                summaries = await asyncio.gather(*summary_tasks)

                for i, summary in enumerate(summaries):
                    relevant_history[i]['body'] = summary

                total_chars = len(new_ticket_body) + sum(len(h['body']) for h in relevant_history)

                if total_chars > TRIAGE_MAX_CONTEXT_CHARS:
                    history_size = sum(len(h['body']) for h in relevant_history)
                    available_for_new = max(1000, TRIAGE_MAX_CONTEXT_CHARS - history_size - 2000)
                    logger.info(f"Still over limit. Truncating new ticket to {available_for_new} chars.")
                    new_ticket_body = self._smart_truncate(new_ticket_body, available_for_new)

            # 7. Format Context for Main LLM
            global_context_str = ""
            user_context_str = ""

            note_global_list = ""
            note_user_list = ""

            for h in history_data:
                if h['type'] == "Global":
                    if h['is_relevant']:
                        global_context_str += f"- [Global Match] {h['title']} (Ticket #{h['id']})\n  Content: {h['body']}\n"
                        note_global_list += f"- {h['title']} (Ticket #{h['id']})\n"

                elif h['type'] == "User":
                    if h['is_relevant']:
                        user_context_str += f"- [User History] {h['title']} (Ticket #{h['id']})\n  Content: {h['body']}\n"
                        note_user_list += f"- {h['title']} (Ticket #{h['id']})\n"
                    else:
                        user_context_str += f"- [User History] {h['title']} (Ticket #{h['id']}) - [IRRELEVANT TO CURRENT ISSUE]\n"
                        note_user_list += f"- {h['title']} (Ticket #{h['id']}) (Irrelevant)\n"

            if not global_context_str: global_context_str = "No similar closed tickets found."
            if not user_context_str: user_context_str = "No relevant user history found."

            context_message = (
                f"NEW TICKET DETAILS:\n"
                f"Title: {title}\n"
                f"Body: {new_ticket_body}\n\n"
                f"USER HISTORY (Context on the User):\n{user_context_str}\n\n"
                f"SIMILAR SOLVED TICKETS (Potential Solutions):\n{global_context_str}\n\n"
                f"INSTRUCTIONS:\n"
                f"1. Summarize the user's issue.\n"
                f"2. If the 'Similar Solved Tickets' seem relevant, suggest a solution based on them.\n"
                f"3. If the 'User History' shows a pattern of similar issues, note that.\n"
                f"4. Provide a concise internal note for the agent."
            )

            # 8. Call Main LLM (Analyst)
            logger.debug("Sending context to Analyst LLM...")
            response, _ = await self.chat_system.text_engine.generate_response(
                persona_config=analyst_persona.get_config_for_engine(),
                history_object={
                    "persona_prompt": analyst_persona.get_prompt(),
                    "message_history": [{"role": "user", "content": context_message}],
                    "history": [{"role": "user", "content": context_message}],  # Legacy key
                    "current_message": {"text": context_message, "image_url": None}
                },
                tools=None
            )

            if response.get('type') == 'text':
                content = response.get('content', '')

                final_note_body = (
                    f"{content}\n\n"
                    f"--------------------------------------------------\n"
                    f"[ AI TRIAGE CONTEXT DUMP ]\n\n"
                    f"EXTRACTED KEYWORDS: {search_keywords}\n"
                    f"GLOBAL SEARCH QUERY: {global_query}\n\n"
                    f"GLOBAL MATCHES FOUND:\n{note_global_list or 'None'}\n\n"
                    f"USER HISTORY FOUND:\n{note_user_list or 'None'}"
                )

                # 9. Post Internal Note (With Fallback)
                await self._post_internal_note(ticket_id, final_note_body)

                # 10. Tag Ticket
                await asyncio.to_thread(
                    self.zammad_client.add_tag,
                    ticket_id=ticket_id,
                    tag=ZAMMAD_TRIAGE_TAG
                )
                logger.info(f"Ticket {ticket_id} triaged successfully.")

        except Exception as e:
            logger.error(f"Error processing ticket {ticket_id}: {e}", exc_info=True)
