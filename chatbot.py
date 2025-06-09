import os
from typing import Dict, List

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import MessageGraph, END


class UserAgent:
    """Agent that keeps per-user chat history and uses Tavily search."""

    def __init__(self, llm: ChatOpenAI) -> None:
        self.llm = llm
        self.search = TavilySearchResults()
        self._histories: Dict[str, List[HumanMessage]] = {}
        self.graph = self._build_graph()

    def _build_graph(self) -> MessageGraph:
        graph = MessageGraph()

        def run_search(state: Dict) -> Dict:
            message: HumanMessage = state["messages"][-1]
            results = self.search.run(message.content)
            state["messages"].append(
                HumanMessage(content=f"Search results: {results}")
            )
            return state

        def run_llm(state: Dict) -> Dict:
            response = self.llm(state["messages"])
            state["messages"].append(response)
            return state

        graph.add_node("search", run_search)
        graph.add_node("llm", run_llm)
        graph.add_edge("search", "llm")
        graph.add_edge("llm", END)
        graph.set_entry_point("search")
        return graph.compile()

    def chat(self, user_id: str, message: str) -> str:
        history = self._histories.get(user_id, [])
        state = {"messages": history + [HumanMessage(content=message)]}
        result = self.graph.invoke(state)
        self._histories[user_id] = result["messages"]
        return result["messages"][-1].content


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Chatbot using LangGraph and Tavily search"
    )
    parser.add_argument("user_id", help="Identifier for the user")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Please set the OPENAI_API_KEY environment variable")

    llm = ChatOpenAI(openai_api_key=api_key)
    agent = UserAgent(llm)

    print("Type 'exit' to quit")
    while True:
        message = input("You: ")
        if message.lower() in {"exit", "quit"}:
            break
        response = agent.chat(args.user_id, message)
        print("Bot:", response)


if __name__ == "__main__":
    main()
