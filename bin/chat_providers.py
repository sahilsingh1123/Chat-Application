from .cohere_ai.chat import CohereChat
from .openai.chat import OpenAIChat
from .phi_3_mini_llama.chat import Phi3MiniChatLlama
from .phi_3_mini_transformers.chat import Phi3MiniChatHF

class ChatFactory:
    @staticmethod
    def get_chat(chat_type):
        if chat_type == 'cohere':
            return CohereChat()
        elif chat_type == 'openai':
            return OpenAIChat()
        elif chat_type == 'phi3-mini-llama':
            return Phi3MiniChatLlama()
        elif chat_type == 'phi3-mini-hf':
            return Phi3MiniChatHF()
        else:
            raise ValueError(f"Invalid chat provider: {chat_type}")
