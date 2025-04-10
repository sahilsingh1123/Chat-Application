from .cohere_ai.chat import CohereChat
from .openai.chat import OpenAIChat

class ChatFactory:
    @staticmethod
    def get_chat(chat_type):
        if chat_type == 'cohere':
            return CohereChat()
        elif chat_type == 'openai':
            return OpenAIChat()
        else:
            raise ValueError(f"Invalid chat provider: {chat_type}")
