# This module holds a single shared instance and exposes it


class ChatHistory:
    def __init__(self):
        self.history = []

    def add_message(self, role: str, message: str):
        self.history.append({"role": role, "message": message})

    def add_user_message(self, message: str):
        self.add_message("user", message)

    def add_system_message(self, message: str):
        self.add_message("assistant", message)

    def get_history(self):
        return self.history

    def get_formatted_history(self) -> str:
        return "\n".join(
            f"{entry['role']}: {entry['message']}" for entry in self.history
        )


# Initialize it once — this becomes our singleton instance
chat_history = ChatHistory()
