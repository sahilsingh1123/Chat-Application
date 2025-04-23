import chainlit as cl
from bin.chat_providers import ChatFactory
from bin.chat_history_service import chat_history
from dotenv import load_dotenv
import os

load_dotenv()
CHAT_PROVIDER = os.getenv("CHAT_PROVIDER")


@cl.on_message
async def main(message: cl.Message):
    # message is the object which contains data from the user
    # message.content = text provided by the user in the chatbox
    # custom logic go here
    print(ChatFactory.available_chats())
    chat = ChatFactory.get_chat(CHAT_PROVIDER)
    response = chat.chat(message.content)
    chat_history.add_user_message(message.content)
    chat_history.add_system_message(response)
    # Start with an empty message and store a reference to it:
    print(chat_history.get_formatted_history())
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
