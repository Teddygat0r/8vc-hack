# chatbot_class.py

import os
from typing import Sequence, List, Dict, Any # Removed Iterator
from typing_extensions import TypedDict, Annotated
import logging

# Langchain Imports
# MODIFIED: Use DirectoryLoader/UnstructuredLoader, removed specific loaders
from langchain_community.document_loaders import DirectoryLoader # DirectoryLoader stays here
from langchain_unstructured import UnstructuredLoader             # UnstructuredLoader moves here
from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# Langgraph Imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the state structure for the graph
class ChatState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

class Chatbot:
    """
    A chatbot class that uses Ollama, ChromaDB, and LangGraph for
    Retrieval-Augmented Generation with conversation memory.
    Loads data using DirectoryLoader/UnstructuredLoader for broad file support.
    """
    def __init__(self,
                 base_directory: str,
                 embedding_model: str = "nomic-embed-text",
                 chat_model: str = "llama3",
                 chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 search_k: int = 3,
                 temperature: float = 0.0):
        """
        Initializes the Chatbot instance.

        Args:
            base_directory (str): Path to the base folder containing documents.
            embedding_model (str): Name of the Ollama embedding model to use.
            chat_model (str): Name of the Ollama chat model to use.
            chunk_size (int): Size of chunks for splitting documents.
            chunk_overlap (int): Overlap between document chunks.
            search_k (int): Number of relevant documents to retrieve.
            temperature (float): Temperature setting for the LLM.
        """
        logger.info("Initializing Chatbot...")
        # Ensure base_directory is absolute for consistency
        self.base_directory = os.path.abspath(base_directory)
        self.embedding_model_name = embedding_model
        self.chat_model_name = chat_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.search_k = search_k
        self.temperature = temperature

        # Initialize components
        self._initialize_embeddings()
        self._initialize_llm()
        # Call the NEW _load_and_process_data using DirectoryLoader
        self._load_and_process_data()
        self._build_graph()
        logger.info("Chatbot initialization complete.")

    def _initialize_embeddings(self):
        """Initializes the Ollama embedding model."""
        logger.info(f"Initializing Ollama embeddings model: {self.embedding_model_name}...")
        self.embeddings = OllamaEmbeddings(model=self.embedding_model_name)
        logger.info("Embeddings model initialized.")

    def _initialize_llm(self):
        """Initializes the Ollama chat model."""
        logger.info(f"Initializing Ollama LLM: {self.chat_model_name}...")
        self.llm = ChatOllama(model=self.chat_model_name, temperature=self.temperature)
        logger.info("Ollama LLM initialized.")

    # MODIFIED: Replaced method to use DirectoryLoader
    def _load_and_process_data(self):
        """
        Loads documents from the base directory using DirectoryLoader with UnstructuredLoader,
        ensures absolute paths in metadata, splits them, creates the ChromaDB store,
        and initializes the retriever.
        """
        if not os.path.isdir(self.base_directory):
            raise FileNotFoundError(f"Base directory not found or is not a directory: {self.base_directory}")

        logger.info(f"Starting document loading using DirectoryLoader from: {self.base_directory}")

        # Configure DirectoryLoader for broad file support
        # Note: Consider adding specific loader_kwargs for UnstructuredLoader if needed
        # e.g., loader_kwargs={'mode': 'single'} for some unstructured strategies
        loader = DirectoryLoader(
            path=self.base_directory,
            glob="**/*.*",  # Load all files with extensions in all subdirs
            # Example to exclude hidden files/dirs: glob="**/[!.]*.*"
            loader_cls=UnstructuredLoader, # Use UnstructuredLoader for parsing various types
            show_progress=True,      # Show progress bar (needs tqdm installed)
            use_multithreading=True, # Speed up loading
            silent_errors=True,      # Log errors for individual files but don't stop
            recursive=True           # Search subdirectories
        )

        try:
            logger.info("Loading documents via DirectoryLoader...")
            all_raw_documents = loader.load() # Load documents
        except Exception as e:
            logger.error(f"DirectoryLoader failed catastrophically during load: {e}", exc_info=True)
            raise ValueError("Failed during initial document loading phase.") from e

        # Check if any documents were successfully loaded
        if not all_raw_documents:
            logger.warning(f"No processable documents found or loaded by DirectoryLoader in: {self.base_directory}. Ensure files exist, have supported types for Unstructured, and are not empty/corrupted.")
            # Depending on requirements, you might allow proceeding with an empty DB or raise an error.
            raise ValueError(f"No processable documents found or loaded in directory: {self.base_directory}.")

        logger.info(f"Total raw documents loaded by DirectoryLoader: {len(all_raw_documents)}")

        # --- Ensure Absolute Path Metadata (Post-Loading) ---
        logger.info("Ensuring absolute paths in document metadata...")
        for doc in all_raw_documents:
            source_path = None
            if doc.metadata and 'source' in doc.metadata:
                source_path = doc.metadata['source']
            elif hasattr(doc, 'source'): # Some loaders might put it directly on the object
                 source_path = doc.source

            if source_path:
                 try:
                     # Convert potentially relative source path from loader to absolute
                     absolute_path = os.path.abspath(source_path)
                     # Ensure metadata dict exists and update/add the source key
                     if not hasattr(doc, 'metadata') or doc.metadata is None:
                         doc.metadata = {}
                     doc.metadata['source'] = absolute_path
                 except Exception as e:
                     logger.warning(f"Could not make path absolute for source '{source_path}': {e}")
                     # Ensure metadata exists even if path conversion failed
                     if not hasattr(doc, 'metadata') or doc.metadata is None:
                          doc.metadata = {}
                     if 'source' not in doc.metadata:
                          doc.metadata['source'] = source_path # Keep original if abspath failed
            else: # No source found, add a placeholder
                 if not hasattr(doc, 'metadata') or doc.metadata is None:
                      doc.metadata = {}
                 if 'source' not in doc.metadata:
                      doc.metadata['source'] = 'Unknown Source (Post-Load)'
        # --- End Metadata Fix ---

        logger.info("Splitting all loaded documents...")
        text_splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator="\n\n", # Adjust separator if needed for specific content types
            is_separator_regex=False,
        )
        try:
            # Split documents after metadata has been processed
            split_documents = text_splitter.split_documents(all_raw_documents)
        except Exception as e:
             logger.error(f"Error during document splitting: {e}", exc_info=True)
             raise ValueError("Failed to split documents. Check content or splitter settings.") from e

        if not split_documents:
            raise ValueError("Splitting documents resulted in zero chunks. Check chunk size/overlap and document content.")

        logger.info(f"Split into {len(split_documents)} document chunks.")

        logger.info("Creating ChromaDB vector store (in-memory)...")
        try:
            if not hasattr(self, 'embeddings') or self.embeddings is None:
                 raise ValueError("Embeddings model not initialized before creating ChromaDB store.")

            # ChromaDB automatically uses metadata from the split_documents
            self.db = Chroma.from_documents(
                 documents=split_documents,
                 embedding=self.embeddings
            )
            logger.info("ChromaDB vector store created successfully.")
        except Exception as e:
            logger.error(f"Failed to create ChromaDB store: {e}", exc_info=True)
            raise ValueError("Failed to create vector store. Check embedding model and ChromaDB setup.") from e

        logger.info(f"Creating retriever (k={self.search_k})...")
        self.retriever = self.db.as_retriever(
            search_type="similarity",
            search_kwargs={'k': self.search_k}
        )
        logger.info("Retriever created.")

    # MODIFIED: Replaced method to use DirectoryLoader
    def add_data(self, directory: str):
        """
        Adds documents from a specified directory to the existing vector store
        using DirectoryLoader/UnstructuredLoader, ensuring absolute paths in metadata.
        """
        abs_directory = os.path.abspath(directory) # Work with absolute path
        logger.info(f"Adding data using DirectoryLoader from: {abs_directory}")
        if not hasattr(self, 'db') or self.db is None:
            logger.error("Cannot add data: Database not initialized.")
            return
        if not os.path.isdir(abs_directory):
             logger.error(f"Cannot add data: Provided path is not a directory: {abs_directory}")
             return

        # Configure DirectoryLoader for adding data
        loader = DirectoryLoader(
            path=abs_directory, # Load from the specified absolute directory
            glob="**/*.*",
            loader_cls=UnstructuredLoader,
            show_progress=True,
            use_multithreading=True,
            silent_errors=True,
            recursive=True
        )

        try:
            logger.info("Loading documents to add...")
            new_raw_documents = loader.load()
        except Exception as e:
            logger.error(f"DirectoryLoader failed to load documents for adding: {e}", exc_info=True)
            return

        if not new_raw_documents:
            logger.warning(f"No new processable documents found by DirectoryLoader in: {abs_directory}")
            return

        logger.info(f"Loaded {len(new_raw_documents)} new raw documents.")

        # --- Ensure Absolute Path Metadata (Post-Loading for Added Data) ---
        logger.info("Ensuring absolute paths in new document metadata...")
        for doc in new_raw_documents:
            source_path = None
            if doc.metadata and 'source' in doc.metadata:
                source_path = doc.metadata['source']
            elif hasattr(doc, 'source'):
                 source_path = doc.source

            if source_path:
                 try:
                     absolute_path = os.path.abspath(source_path)
                     if not hasattr(doc, 'metadata') or doc.metadata is None:
                         doc.metadata = {}
                     doc.metadata['source'] = absolute_path
                 except Exception as e:
                     logger.warning(f"Could not make path absolute for source '{source_path}': {e}")
                     if not hasattr(doc, 'metadata') or doc.metadata is None:
                          doc.metadata = {}
                     if 'source' not in doc.metadata:
                          doc.metadata['source'] = source_path
            else:
                 if not hasattr(doc, 'metadata') or doc.metadata is None:
                      doc.metadata = {}
                 if 'source' not in doc.metadata:
                      doc.metadata['source'] = 'Unknown Source (Add Data)'
        # --- End Metadata Fix ---


        logger.info("Splitting new documents...")
        text_splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator="\n\n",
            is_separator_regex=False,
        )
        try:
            # Split after ensuring metadata is processed
            documents_to_add = text_splitter.split_documents(new_raw_documents)
        except Exception as e:
             logger.error(f"Error splitting new documents: {e}", exc_info=True)
             return

        if not documents_to_add:
            logger.warning("Splitting new documents resulted in zero chunks.")
            return

        logger.info(f"Split into {len(documents_to_add)} new chunks.")

        try:
            # Add the new split documents (with corrected metadata) to the existing Chroma DB
            self.db.add_documents(documents=documents_to_add)
            logger.info(f"Successfully added {len(documents_to_add)} new chunks to the vector store.")
        except Exception as e:
             logger.error(f"Failed to add new documents to vector store: {e}", exc_info=False)


    # --- REMOVED _recurse_filepaths ---
    # --- REMOVED _document_loader ---


    # --- _format_docs, _format_chat_history, _rag_chain_node, _build_graph, chat ---
    # --- These methods remain largely the same as in the last correct version ---
    # --- Ensure any stray '...' syntax errors are removed from these ---

    def _format_docs(self, docs: List[Document]) -> str:
        """
        Helper function to format retrieved documents into a string,
        including the absolute source path from metadata.
        """
        if not docs:
            return "No relevant context found."

        formatted_docs = []
        for i, doc in enumerate(docs):
            # Retrieve source (absolute path) from metadata
            source = doc.metadata.get("source", "Unknown Source")
            content = doc.page_content
            formatted_docs.append(f"--- Context Piece {i+1} (Source: {source}) ---\n{content}")

        return "\n\n".join(formatted_docs)

    def _format_chat_history(self, messages: Sequence[BaseMessage]) -> str:
        """Helper function to format chat history for the prompt."""
        history = ""
        if len(messages) > 1:
            for msg in messages[:-1]:
                role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                history += f"{role}: {msg.content}\n"
        return history.strip()

    def _rag_chain_node(self, state: ChatState) -> Dict[str, Any]:
        """
        LangGraph node that performs RAG: retrieves, formats prompt, calls LLM.
        """
        messages = state['messages']
        last_user_message = messages[-1].content

        retrieved_docs = self.retriever.invoke(last_user_message)
        context_str = self._format_docs(retrieved_docs) # Uses formatted context with absolute paths
        chat_history_str = self._format_chat_history(messages)

        RAG_PROMPT_TEMPLATE = """SYSTEM: You are a helpful assistant. Use the following context pieces retrieved from a knowledge base to answer the user's question. If you don't know the answer based on the context, just say that you don't know. Keep the answer concise and relevant to the question. When you want to cite specific context, link the full file path of the context in the end of the message.

CONTEXT:
{context}

CONVERSATION HISTORY:
{chat_history}

USER QUESTION: {question}

ANSWER:"""
        # FIX: Removed stray '...' from original paste.txt
        rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

        rag_chain = (
            RunnablePassthrough.assign(
                context=lambda x: context_str,
                chat_history=lambda x: chat_history_str,
                question=lambda x: x['question']
            )
            | rag_prompt
            | self.llm
            | StrOutputParser()
        )

        ai_response_text = rag_chain.invoke({"question": last_user_message})

        return {"messages": [AIMessage(content=ai_response_text)]}

    def _build_graph(self):
        """Builds the LangGraph application."""
        logger.info("Building the graph...")
        workflow = StateGraph(ChatState)
        workflow.add_node("rag_assistant", self._rag_chain_node)
        workflow.set_entry_point("rag_assistant")
        workflow.add_edge("rag_assistant", END)
        self.memory = MemorySaver()
        self.app = workflow.compile(checkpointer=self.memory)
        logger.info("Graph compiled with memory.")

    def chat(self, user_input: str, thread_id: str) -> str:
        """
        Sends user input to the chatbot for a specific conversation thread
        and returns the AI's response.
        """
        if not user_input:
            return "Please provide some input."
        if not thread_id:
            raise ValueError("A thread_id must be provided for conversation memory.")

        config = {"configurable": {"thread_id": thread_id}}
        messages_input = {"messages": [HumanMessage(content=user_input)]}

        try:
            response_state = self.app.invoke(messages_input, config)
            ai_response_message = response_state['messages'][-1]
            return ai_response_message.content
        except Exception as e:
             logger.error(f"Error during graph invocation for thread '{thread_id}': {e}", exc_info=True)
             # Depending on the desired behavior, return an error message or re-raise
             return "Sorry, an error occurred while processing your request."

# --- End of chatbot_class.py ---
