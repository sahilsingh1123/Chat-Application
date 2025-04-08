import chainlit as cl
from bin.chat_providers import ChatFactory
from dotenv import load_dotenv
import os

load_dotenv()
CHAT_PROVIDER = os.getenv('CHAT_PROVIDER')

@cl.on_message
async def main(message: cl.Message):
    # message is the object which contains data from the user
    # message.content = text provided by the user in the chatbox
    # custom logic go here
    chat = ChatFactory.get_chat(CHAT_PROVIDER)
    response = chat.chat(message.content)

    await cl.Message(
        content=f"Received: {response}",
    ).send()