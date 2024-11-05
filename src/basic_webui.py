import gradio as gr

from src.chat_system import ChatSystem


class WebUI:
    def __init__(self, bot: ChatSystem):
        self.bot = ChatSystem()

        webui = self.create_interface()
        webui.launch()

    def create_interface(self):
        with gr.Blocks(analytics_enabled=False) as demo:
            # Define input and output components
            persona_name = gr.Textbox(label="Persona Name")
            message_in = gr.Textbox(label="Input")
            message_out = gr.Textbox(label="Output", interactive=False)

            self.setup_event_handlers(persona_name, message_in, message_out)
        return demo


    def setup_event_handlers(self, persona_name, message_in, message_out):
        message_in.submit(self.handle_submission, inputs=[persona_name, message_in], outputs=message_out)


    async def handle_submission(self, persona_name, message_in):
        # Use your existing function to process the input
        response = await self.bot.generate_response(persona_name=persona_name, message=message_in, context=[])
        return response

