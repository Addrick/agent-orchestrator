import json
import logging
import os
from typing import Dict, Any, Optional, Tuple, List

from dotenv import load_dotenv

# --- Provider-specific imports ---
import aiohttp
import anthropic
from openai import AsyncOpenAI
from google import genai
from google.genai.types import GenerateContentConfig, Tool, GoogleSearch

logger = logging.getLogger(__name__)


class TextEngine:
    """A centralized engine for handling requests to various LLM APIs."""

    def __init__(self) -> None:
        # --- Lazy-loaded clients ---
        self.openai_client: Optional[AsyncOpenAI] = None
        self.anthropic_client: Optional[anthropic.Anthropic] = None

        # --- Google Client (matching original implementation) ---
        self.google_client: Optional[genai.client.AsyncClient] = None
        self.google_search_tool: Optional[Tool] = None
        self.google_tool_config: Optional[Dict[str, Any]] = None
        self.google_safety_settings: Optional[List[Dict[str, str]]] = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        self._initialize_env()

    def _initialize_env(self) -> None:
        """Load API keys from .env file."""
        env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
        if os.path.exists(env_path):
            load_dotenv(env_path)
        else:
            logger.warning(".env file not found, API keys must be in environment.")

    async def _get_openai_client(self) -> AsyncOpenAI:
        """Initializes and returns the OpenAI client."""
        if self.openai_client is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key: raise ValueError("OPENAI_API_KEY not found in environment.")
            self.openai_client = AsyncOpenAI(api_key=api_key)
        return self.openai_client

    def _get_anthropic_client(self) -> anthropic.Anthropic:
        """Initializes and returns the Anthropic client."""
        if self.anthropic_client is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key: raise ValueError("ANTHROPIC_API_KEY not found in environment.")
            self.anthropic_client = anthropic.Anthropic(api_key=api_key)
        return self.anthropic_client

    def _initialize_google_client(self) -> None:
        """Initializes the Google client using the original project's method."""
        if self.google_client is not None:
            return

        api_key = os.environ.get("GOOGLE_GENERATIVEAI_API_KEY")
        if not api_key: raise ValueError("GOOGLE_GENERATIVEAI_API_KEY not found in environment.")

        client = genai.client.BaseApiClient(api_key=api_key)
        self.google_client = genai.client.AsyncClient(client)
        self.google_search_tool = Tool(google_search=GoogleSearch())
        self.google_tool_config = {"function_calling_config": {"mode": "AUTO"}}
        logger.info("Google AI Studio client initialized.")

    async def generate_response(self, persona_config: dict, context_object: Dict[str, Any]) -> Tuple[str, Optional[Dict]]:
        """
        Routes the generation request and returns the text response AND the API payload.
        Returns: (response_string, api_payload_dictionary_or_None)
        """
        model_name = persona_config.get("model_name", "")
        if model_name.startswith("gpt"):
            return await self._generate_openai_response(persona_config, context_object)
        elif "claude" in model_name:
            return await self._generate_anthropic_response(persona_config, context_object)
        elif "gemini" in model_name:
            return await self._generate_google_response(persona_config, context_object)
        elif model_name == 'local':
            return await self._generate_local_response(persona_config, context_object)
        else:
            error_msg = f"Error: Model '{model_name}' is not supported."
            logger.error(f"Unknown model provider for model_name: {model_name}")
            return error_msg, None

    async def _generate_openai_response(self, config: dict, context: Dict[str, Any]) -> Tuple[str, Dict]:
        client = await self._get_openai_client()
        messages = [{"role": "system", "content": context["persona_prompt"]}]
        messages.extend(context["history"])
        content_parts = [{"type": "text", "text": context["current_message"]["text"]}]
        if context["current_message"].get("image_url"):
            content_parts.append({"type": "image_url", "image_url": {"url": context["current_message"]["image_url"]}})
        messages.append({"role": "user", "content": content_parts})
        api_params = {"model": config["model_name"], "messages": messages,
                      "max_tokens": config.get("max_output_tokens"), "temperature": config.get("temperature"),
                      "top_p": config.get("top_p")}
        api_params = {k: v for k, v in api_params.items() if v is not None}
        try:
            completion = await client.chat.completions.create(**api_params)
            return completion.choices[0].message.content, api_params
        except Exception as e:
            logger.error(f"OpenAI API error: {e}", exc_info=True)
            return f"Error communicating with OpenAI: {e}", api_params

    async def _generate_anthropic_response(self, config: dict, context: Dict[str, Any]) -> Tuple[str, Dict]:
        client = self._get_anthropic_client()
        history = [msg for msg in context["history"] if msg["role"] != "system"]
        history.append({"role": "user", "content": context["current_message"]["text"]})
        api_params = {"model": config["model_name"], "system": context["persona_prompt"], "messages": history,
                      "max_tokens": config["max_output_tokens"], "temperature": config.get("temperature"),
                      "top_p": config.get("top_p"), "top_k": config.get("top_k")}
        api_params = {k: v for k, v in api_params.items() if v is not None}
        try:
            response = client.messages.create(**api_params)
            return response.content[0].text, api_params
        except Exception as e:
            logger.error(f"Anthropic API error: {e}", exc_info=True)
            return f"Error communicating with Anthropic: {e}", api_params

    async def _generate_google_response(self, config: dict, context: Dict[str, Any]) -> Tuple[str, Dict]:
        """Builds and sends a request to the Google API using the original client structure."""
        try:
            self._initialize_google_client()
        except ValueError as e:
            return f"Error: Google not configured: {e}", {}

        history_str = "\n".join([f"{item['role']}: {item['content']}" for item in context['history']])
        prompt = context['persona_prompt']
        message = context['current_message']['text']
        request_content = (f"### Instructions: ###\n{prompt}\n\n"
                           f"### Conversation History: ###\n{history_str}\n\n"
                           f"### Current Message to Respond To: ###\n{message}")

        # This dictionary contains the actual Tool OBJECTS and is used for the API call
        content_config_for_api = {
            'tools': [self.google_search_tool],
            'tool_config': self.google_tool_config,
            'response_modalities': ["TEXT"],
            'safety_settings': self.google_safety_settings
        }
        if isinstance(config.get("max_output_tokens"), int):
            if config.get("max_output_tokens") >= 100:
                content_config_for_api['max_output_tokens'] = config.get("max_output_tokens")
        if isinstance(config.get("temperature"), (int, float)):
            content_config_for_api['temperature'] = config.get("temperature")
        if isinstance(config.get("top_p"), (int, float)):
            content_config_for_api['top_p'] = config.get("top_p")
        if isinstance(config.get("top_k"), (int, float)):
            content_config_for_api['top_k'] = config.get("top_k")
        # ... any other params ...

        # +++ THIS IS THE FIX +++
        # Create a deep copy of the config for dumping, then sanitize it.
        # This ensures the original config_for_api with its objects is not modified.
        config_for_dumping = json.loads(json.dumps(content_config_for_api, default=str))

        # The final payload for dumping is now guaranteed to be serializable
        api_params_for_dumping = {
            'model': config["model_name"],
            'contents': request_content,
            'config': config_for_dumping
        }

        try:
            # The API call uses the original dictionary with the real objects
            response_obj = await self.google_client.models.generate_content(
                model=config["model_name"],
                contents=request_content,
                config=GenerateContentConfig(**content_config_for_api)
            )

            if response_obj.prompt_feedback and response_obj.prompt_feedback.block_reason:
                return f"Response blocked by Google: {response_obj.prompt_feedback.block_reason.name}", api_params_for_dumping

            if response_obj.candidates:
                return response_obj.text, api_params_for_dumping
            else:
                return "Google returned an empty response.", api_params_for_dumping
        except Exception as e:
            logger.error(f"Error during Google generation: {e}", exc_info=True)
            return f"An error occurred with Google API: {e}", api_params_for_dumping

    async def _generate_local_response(self, config: dict, context: Dict[str, Any]) -> Tuple[str, Dict]:
        url = 'http://localhost:5001/api/v1/generate'
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in context['history']])
        full_prompt = f"{history_str}\nuser: {context['current_message']['text']}\nassistant:"
        payload = {"max_length": config["max_output_tokens"], "temperature": config.get("temperature", 0.7),
                   "top_p": config.get("top_p", 0.9), "top_k": config.get("top_k", 40),
                   "memory": context["persona_prompt"], "prompt": full_prompt}
        payload = {k: v for k, v in payload.items() if v is not None}
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data['results'][0]['text'].strip(), payload
                    else:
                        error_text = await response.text()
                        logger.error(f"Local model error ({response.status}): {error_text}")
                        return f"Error with local model: Status {response.status}", payload
        except aiohttp.ClientError as e:
            logger.error(f"Local model connection error: {e}", exc_info=True)
            return "Error: Could not connect to local model at http://localhost:5001.", payload
