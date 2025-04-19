import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings # Use langchain_community
from langchain_chroma import Chroma
from langchain_ollama.chat_models import ChatOllama
from typing import Sequence, List, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict, Annotated

print("Loading data...")
loader = TextLoader('./data.txt') # Assumes data.txt is in the same directory
raw_documents = loader.load()

print("Splitting documents...")
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)  # Smaller chunks for demo
documents = text_splitter.split_documents(raw_documents)

print(f"Loaded and split {len(documents)} document chunks.")



print("Initializing Ollama embeddings...")
# Make sure 'nomic-embed-text' model is available via `ollama pull nomic-embed-text`
embeddings = OllamaEmbeddings(model="nomic-embed-text")

print("Embeddings model initialized.")

print("Creating ChromaDB vector store (in-memory)...")
# This creates the vector store from the document chunks and uses Ollama embeddings
db = Chroma.from_documents(documents, embeddings)
print("ChromaDB vector store created.")

# Create the retriever interface
retriever = db.as_retriever(
    search_type="similarity", # Use similarity search
    search_kwargs={'k': 3}    # Retrieve top 3 relevant chunks
)
print("Retriever created.")


print("Initializing Ollama LLM...")
# Make sure 'llama3' model is available via `ollama pull llama3`
# Ensure Ollama service is running in the background
llm = ChatOllama(model="llama3", temperature=0) # Use the desired model, temp=0 for more factual

print("Ollama LLM initialized.")

# --- 1. Define State ---
class ChatState(TypedDict):
    # 'add_messages' ensures new messages are appended to the list
    messages: Annotated[Sequence[BaseMessage], add_messages]

# --- 2. Define RAG Logic and Node ---

# System prompt to guide the LLM, incorporating context
RAG_PROMPT_TEMPLATE = """
SYSTEM: You are a helpful assistant. Use the following context pieces retrieved from a knowledge base to answer the user's question. If you don't know the answer based on the context, just say that you don't know. Keep the answer concise and relevant to the question.

CONTEXT:
{context}

CONVERSATION HISTORY:
{chat_history}

USER QUESTION: {question}

ANSWER:
"""
rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

def format_docs(docs: List[Any]) -> str:
    """Helper function to format retrieved documents into a string."""
    return "\n\n".join(doc.page_content for doc in docs)

def format_chat_history(messages: Sequence[BaseMessage]) -> str:
    """Helper function to format chat history for the prompt."""
    history = ""
    if len(messages) > 1: # Exclude the latest user question
        for msg in messages[:-1]:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            history += f"{role}: {msg.content}\n"
    return history

def rag_chain_node(state: ChatState) -> Dict[str, Any]:
    """
    Node that performs RAG:
    1. Retrieves documents based on the latest user message.
    2. Formats context and chat history.
    3. Creates the prompt.
    4. Calls the LLM.
    5. Returns the AI message.
    """
    print("--- Executing RAG Node ---")
    messages = state['messages']
    last_user_message = messages[-1].content
    print(f"Retrieving documents for: {last_user_message}")

    # 1. Retrieve documents
    retrieved_docs = retriever.invoke(last_user_message)
    context_str = format_docs(retrieved_docs)
    print(f"Retrieved context:\n{context_str[:500]}...") # Print snippet of context

    # 2. Format chat history
    chat_history_str = format_chat_history(messages)

    # 3. Prepare input for the RAG chain using LCEL
    rag_chain = (
        RunnablePassthrough.assign(
            context=lambda x: context_str, # Assign formatted context
            chat_history=lambda x: chat_history_str, # Assign formatted history
            question=lambda x: x['question'] # Pass the user question through
        )
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    print("Invoking LLM with RAG prompt...")
    # 4. Invoke the chain
    ai_response_text = rag_chain.invoke({"question": last_user_message})
    print(f"LLM response: {ai_response_text}")

    # 5. Return the message to be added to the state
    return {"messages": [AIMessage(content=ai_response_text)]}

# --- 3. Build Graph ---
print("Building the graph...")
workflow = StateGraph(ChatState)

# Add the single RAG node
workflow.add_node("rag_assistant", rag_chain_node)

# Set the entry point and the edges
workflow.set_entry_point("rag_assistant") # Start with the RAG node
workflow.add_edge("rag_assistant", END) # After RAG node, the turn ends

# --- 4. Add Memory ---
memory = MemorySaver() # Simple in-memory checkpointing

# Compile the graph
app = workflow.compile(checkpointer=memory)
print("Graph compiled with memory.")

if __name__ == "__main__":
    print("\nChatbot initialized. Type 'quit' or 'exit' to end.")

    # Use a unique ID for each conversation thread
    thread_id_counter = 0
    while True:
        thread_id_counter += 1
        config = {"configurable": {"thread_id": f"user_session_{thread_id_counter}"}}
        print(f"\n--- Starting New Conversation (Thread ID: {config['configurable']['thread_id']}) ---")

        while True:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit"]:
                print("Ending conversation thread.")
                break # Break inner loop to start a new thread or exit outer

            # Invoke the app. LangGraph handles state loading/saving via checkpointer
            try:
                # Input to the graph is the current state, with the new user message added
                response = app.invoke({"messages": [HumanMessage(content=user_input)]}, config) # REPLACE THIS WITH FLASK SERVER INPUT *******

                # The response contains the full state; the last message is the AI's reply
                ai_response = response['messages'][-1]
                print(f"AI: {ai_response.content}")
            except Exception as e:
                print(f"An error occurred: {e}")
                # Potentially add more robust error handling

        # Ask if the user wants to start a new conversation or exit completely
        new_convo = input("Start a new conversation? (yes/no): ").lower()
        if new_convo != 'yes':
            print("Exiting chatbot.")
            break # Break outer loop