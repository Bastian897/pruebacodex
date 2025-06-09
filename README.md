# Chatbot Example with Langchain

This repository contains a minimal example of a chatbot built with
[LangChain](https://github.com/langchain-ai/langchain). The bot keeps memory
for each user so conversations are preserved between turns.

## Requirements

- Python 3.11+
- `langchain`
- `openai`
- `langgraph`
- `tavily-python`

Install the dependencies with:

```bash
pip install -r requirements.txt
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY=your-key-here
```

To enable web search, set your Tavily API key:

```bash
export TAVILY_API_KEY=your-tavily-key
```

## Usage

Start chatting by running:

```bash
python chatbot.py USER_ID
```

The application uses LangGraph with a Tavily web search tool to enrich
responses. Each user maintains their own conversation history.

Replace `USER_ID` with a unique identifier for the user. Each user has its own
conversation memory. Type `exit` to quit the program.
