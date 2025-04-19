from langchain_ollama.chat_models import ChatOllama

llm = ChatOllama(model="llama3.2")
print(f"Successfully initialized local Ollama model: {llm.model}")