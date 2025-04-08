"""
Interface for the chat class
"""

from abc import ABC, abstractmethod


class Chat(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def chat(self, msg):
        pass

    @abstractmethod
    def _get_client(self):
        pass

    @abstractmethod
    def _get_template(self, msg):
        pass