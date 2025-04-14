from .cohere_ai.chat import CohereChat
from .openai.chat import OpenAIChat
from .phi_3_mini.chat import Phi3MiniChat

class ChatFactory:
    @staticmethod
    def get_chat(chat_type):
        if chat_type == 'cohere':
            return CohereChat()
        elif chat_type == 'openai':
            return OpenAIChat()
        elif chat_type == 'phi3-mini':
            return Phi3MiniChat()
        else:
            raise ValueError(f"Invalid chat provider: {chat_type}")
