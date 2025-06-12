import os
from typing import List
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain_tavily import TavilySearch
from langgraph.graph import MessageGraph, END

_ = load_dotenv(find_dotenv())


class UserAgent:
    """Agente que mantiene el historial por usuario y usa Tavily para buscar."""

    def __init__(self, llm: ChatOpenAI) -> None:
        self.llm = llm
        self.search = TavilySearch(api_key=os.environ["TAVILY_API_KEY"])
        self._histories: dict[str, List[HumanMessage]] = {}
        self.graph = self._build_graph()

    def _build_graph(self) -> MessageGraph:
        graph = MessageGraph()

        # Nodo de búsqueda
        def run_search(messages: List[HumanMessage]) -> List[HumanMessage]:
            query = messages[-1].content
            results = self.search.run(query)
            messages.append(HumanMessage(content=f"Search results: {results}"))
            return messages

        # Nodo del LLM
        def run_llm(messages: List[HumanMessage]) -> List[HumanMessage]:
            response = self.llm.invoke(messages)
            return messages + [response]

        graph.add_node("search", run_search)
        graph.add_node("llm", run_llm)
        graph.add_edge("search", "llm")
        graph.add_edge("llm", END)
        graph.set_entry_point("search")

        return graph.compile()

    def chat(self, user_id: str, message: str) -> str:
        messages = self._histories.get(user_id, []) + [HumanMessage(content=message)]
        result = self.graph.invoke(messages)
        self._histories[user_id] = result
        return result[-1].content


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Chatbot usando LangGraph y Tavily")
    parser.add_argument("user_id", help="Identificador del usuario")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Debes definir la variable de entorno OPENAI_API_KEY")

    llm = ChatOpenAI(openai_api_key=api_key)
    agent = UserAgent(llm)

    print("Escribe 'exit' para salir")
    while True:
        message = input("Tú: ")
        if message.lower() in {"exit", "quit"}:
            break
        response = agent.chat(args.user_id, message)
        print("Bot:", response)


if __name__ == "__main__":
    main()
