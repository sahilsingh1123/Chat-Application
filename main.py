import chainlit as cl
from bin.chat_providers import ChatFactory
from dotenv import load_dotenv
import os

load_dotenv()
CHAT_PROVIDER = os.getenv("CHAT_PROVIDER")


@cl.on_message
async def main(message: cl.Message):
    # message is the object which contains data from the user
    # message.content = text provided by the user in the chatbox
    # custom logic go here
    chat = ChatFactory.get_chat(CHAT_PROVIDER)
    response = chat.chat(message.content)
    # Start with an empty message and store a reference to it:
    await cl.Message(content=f"Received: {response}").send()


def stream_msg():
    """stream = chat.sync_stream_chat(message.content)

    accumulated_text = ""
    for event in stream:
        if event and event.type == 'content-delta':
            accumulated_text += event.delta.message.content.text
        # Update the same message instead of sending a new one
    await message_obj.stream(content=f"Received: {accumulated_text}")
    await message_obj.update()"""
    pass
