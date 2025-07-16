from src.chat_system import ChatSystem
import logging
logger = logging.getLogger(__name__)

class WebUI:
    def __init__(self, bot: ChatSystem):
        self.bot = ChatSystem()

        webui = self.create_interface()
        webui.launch()

    def create_interface(self):
        pass


    def setup_event_handlers(self, persona_name, message_in, message_out):
        pass

    async def handle_submission(self, persona_name, message_in):
        # Use your existing function to process the input
        response = await self.bot.generate_response(persona_name=persona_name, message=message_in, context=[])
        return response

    @classmethod
    def launch(cls):
        pass

