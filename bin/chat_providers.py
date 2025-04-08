from .cohere_ai.chat import CohereChat

class ChatFactory:
    @staticmethod
    def get_chat(chat_type):
        if chat_type == 'cohere':
            return CohereChat()