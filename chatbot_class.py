# chatbot_class.py

import os
from typing import Sequence, List, Dict, Any
from typing_extensions import TypedDict, Annotated

# Langchain Imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings # Corrected import
from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Langgraph Imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# Define the state structure for the graph
class ChatState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

class Chatbot:
    """
    A chatbot class that uses Ollama, ChromaDB, and LangGraph for
    Retrieval-Augmented Generation with conversation memory.
    """
    def __init__(self,
                 data_file_path: str,
                 embedding_model: str = "nomic-embed-text",
                 chat_model: str = "llama3",
                 chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 search_k: int = 3,
                 temperature: float = 0.0):
        """
        Initializes the Chatbot instance.

        Args:
            data_file_path (str): Path to the text file containing the knowledge base.
            embedding_model (str): Name of the Ollama embedding model to use.
            chat_model (str): Name of the Ollama chat model to use.
            chunk_size (int): Size of chunks for splitting documents.
            chunk_overlap (int): Overlap between document chunks.
            search_k (int): Number of relevant documents to retrieve.
            temperature (float): Temperature setting for the LLM.
        """
        # print("Initializing Chatbot...")
        self.data_file_path = data_file_path
        self.embedding_model_name = embedding_model
        self.chat_model_name = chat_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.search_k = search_k
        self.temperature = temperature

        # Initialize components
        self._initialize_embeddings()
        self._initialize_llm()
        self._load_and_process_data()
        self._build_graph()

        # print("Chatbot initialization complete.")

    def _initialize_embeddings(self):
        """Initializes the Ollama embedding model."""
        # print(f"Initializing Ollama embeddings model: {self.embedding_model_name}...")
        # Ensure the model is available locally via Ollama
        self.embeddings = OllamaEmbeddings(model=self.embedding_model_name)
        # print("Embeddings model initialized.")

    def _initialize_llm(self):
        """Initializes the Ollama chat model."""
        # print(f"Initializing Ollama LLM: {self.chat_model_name}...")
        # Ensure the model is available locally via Ollama
        self.llm = ChatOllama(model=self.chat_model_name, temperature=self.temperature)
        # print("Ollama LLM initialized.")

    def _load_and_process_data(self):
        """Loads data, splits it, creates ChromaDB store, and retriever."""
        if not os.path.exists(self.data_file_path):
            raise FileNotFoundError(f"Data file not found at: {self.data_file_path}")

        # print(f"Loading data from {self.data_file_path}...")
        loader = TextLoader(self.data_file_path)
        raw_documents = loader.load()

        if not raw_documents:
            raise ValueError(f"No documents loaded from {self.data_file_path}. Is the file empty?")

        # print("Splitting documents...")
        text_splitter = CharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        documents = text_splitter.split_documents(raw_documents)

        if not documents:
            raise ValueError("Splitting documents resulted in zero chunks. Check chunk size/overlap and file content.")

        # print(f"Loaded and split {len(documents)} document chunks.")

        # print("Creating ChromaDB vector store (in-memory)...")
        self.db = Chroma.from_documents(documents, self.embeddings)
        # print("ChromaDB vector store created.")

        # print(f"Creating retriever (k={self.search_k})...")
        self.retriever = self.db.as_retriever(
            search_type="similarity",
            search_kwargs={'k': self.search_k}
        )
        # print("Retriever created.")

    def _format_docs(self, docs: List[Any]) -> str:
        """Helper function to format retrieved documents into a string."""
        return "\n\n".join(doc.page_content for doc in docs)

    def _format_chat_history(self, messages: Sequence[BaseMessage]) -> str:
        """Helper function to format chat history for the prompt."""
        history = ""
        # Format all messages except the latest user message
        if len(messages) > 1:
            for msg in messages[:-1]:
                role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                history += f"{role}: {msg.content}\n"
        return history.strip()

    def _rag_chain_node(self, state: ChatState) -> Dict[str, Any]:
        """
        LangGraph node that performs RAG: retrieves, formats prompt, calls LLM.
        """
        # # print("--- Executing RAG Node ---") # Optional: for debugging
        messages = state['messages']
        last_user_message = messages[-1].content
        # # print(f"Retrieving documents for: {last_user_message}") # Optional: for debugging

        # Retrieve documents
        retrieved_docs = self.retriever.invoke(last_user_message)
        context_str = self._format_docs(retrieved_docs)
        # # print(f"Retrieved context snippet:\n{context_str[:200]}...") # Optional: for debugging

        # Format chat history
        chat_history_str = self._format_chat_history(messages)

        # Define the RAG prompt template (moved here for clarity)
        RAG_PROMPT_TEMPLATE = """SYSTEM: You are a helpful assistant. Use the following context pieces retrieved from a knowledge base to answer the user's question. If you don't know the answer based on the context, just say that you don't know. Keep the answer concise and relevant to the question.

CONTEXT:
{context}

CONVERSATION HISTORY:
{chat_history}

USER QUESTION: {question}

ANSWER:"""
        rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)


        # Prepare input for the RAG chain using LCEL
        rag_chain = (
            RunnablePassthrough.assign(
                # Pass formatted context, history, and the question to the prompt
                context=lambda x: context_str,
                chat_history=lambda x: chat_history_str,
                question=lambda x: x['question'] # The user's latest question
            )
            | rag_prompt
            | self.llm # Use the initialized Ollama LLM
            | StrOutputParser()
        )

        # # print("Invoking LLM with RAG prompt...") # Optional: for debugging
        # Invoke the chain with the last user message as the question
        ai_response_text = rag_chain.invoke({"question": last_user_message})
        # # print(f"LLM response: {ai_response_text}") # Optional: for debugging

        # Return the AI message to be added to the state
        return {"messages": [AIMessage(content=ai_response_text)]}

    def _build_graph(self):
        """Builds the LangGraph application."""
        # print("Building the graph...")
        workflow = StateGraph(ChatState)

        # Add the single RAG node
        workflow.add_node("rag_assistant", self._rag_chain_node)

        # Set the entry point and the only edge
        workflow.set_entry_point("rag_assistant")
        workflow.add_edge("rag_assistant", END) # End the graph after the RAG node runs

        # Add memory (checkpointer)
        self.memory = MemorySaver()

        # Compile the graph
        self.app = workflow.compile(checkpointer=self.memory)
        # print("Graph compiled with memory.")

    def chat(self, user_input: str, thread_id: str) -> str:
        """
        Sends user input to the chatbot for a specific conversation thread
        and returns the AI's response.

        Args:
            user_input (str): The message from the user.
            thread_id (str): A unique identifier for the conversation session.

        Returns:
            str: The chatbot's response message content.

        Raises:
            Exception: Propagates exceptions from the LangGraph invocation.
        """
        if not user_input:
            return "Please provide some input."
        if not thread_id:
            raise ValueError("A thread_id must be provided for conversation memory.")

        # Configuration for LangGraph memory
        config = {"configurable": {"thread_id": thread_id}}

        # Input message for the graph
        messages_input = {"messages": [HumanMessage(content=user_input)]}

        # Invoke the LangGraph app
        # LangGraph handles loading previous messages for the thread_id via the checkpointer
        response_state = self.app.invoke(messages_input, config)

        # The response state contains the full message history; the last one is the AI's reply
        ai_response_message = response_state['messages'][-1]

        # Return only the content of the AI's response
        return ai_response_message.content

# --- End of chatbot_class.py ---
