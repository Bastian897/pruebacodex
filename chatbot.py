import os
from typing import Dict

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory


class UserManager:
    """Manage conversation chains for each user."""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self._chains: Dict[str, ConversationChain] = {}

    def _get_chain(self, user_id: str) -> ConversationChain:
        if user_id not in self._chains:
            self._chains[user_id] = ConversationChain(
                llm=self.llm,
                memory=ConversationBufferMemory(return_messages=True),
            )
        return self._chains[user_id]

    def chat(self, user_id: str, message: str) -> str:
        chain = self._get_chain(user_id)
        return chain.predict(input=message)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Chatbot with per-user memory")
    parser.add_argument("user_id", help="Identifier for the user")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Please set the OPENAI_API_KEY environment variable")

    llm = ChatOpenAI(openai_api_key=api_key)
    manager = UserManager(llm)

    print("Type 'exit' to quit")
    while True:
        message = input("You: ")
        if message.lower() in {"exit", "quit"}:
            break
        response = manager.chat(args.user_id, message)
        print("Bot:", response)


if __name__ == "__main__":
    main()
