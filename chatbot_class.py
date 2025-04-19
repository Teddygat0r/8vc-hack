# chatbot_class.py

import os
from typing import Sequence, List, Dict, Any, Iterator # Added Iterator
from typing_extensions import TypedDict, Annotated
import logging # Added for logging errors

# Langchain Imports
from langchain_community.document_loaders import ( # Grouped imports
    TextLoader,
    PyMuPDFLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredFileLoader
)
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document # Explicitly import Document

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
            base_directory (str): Path to base folder
            embedding_model (str): Name of the Ollama embedding model to use.
            chat_model (str): Name of the Ollama chat model to use.
            chunk_size (int): Size of chunks for splitting documents.
            chunk_overlap (int): Overlap between document chunks.
            search_k (int): Number of relevant documents to retrieve.
            temperature (float): Temperature setting for the LLM.
        """
        # print("Initializing Chatbot...")
        self.base_directory = base_directory
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
        """
        Finds all supported files in the base directory, loads them with metadata,
        splits the combined content, creates the ChromaDB store, and initializes the retriever.
        """
        if not os.path.isdir(self.base_directory):
            raise FileNotFoundError(f"Base directory not found or is not a directory: {self.base_directory}")

        all_raw_documents: List[Document] = []
        logger.info(f"Starting document loading process from base directory: {self.base_directory}")

        file_count = 0
        for file_path in self._recurse_filepaths(self.base_directory):
            file_count += 1
            documents_from_file = self._document_loader(file_path)
            all_raw_documents.extend(documents_from_file)

        logger.info(f"Finished scanning directory. Processed {file_count} potential files.")

        # Check AFTER attempting to load all files
        if not all_raw_documents:
            raise ValueError(f"No processable documents found or loaded in directory: {self.base_directory}. Check file types, content, and permissions.")
        # Log count only if documents were found
        logger.info(f"Total raw documents loaded from all files: {len(all_raw_documents)}") # FIX 1 applied implicitly by correct flow

        logger.info("Splitting all loaded documents...")
        text_splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator="\n\n",
            is_separator_regex=False,
        )
        try:
            split_documents = text_splitter.split_documents(all_raw_documents)
        except Exception as e:
             logger.error(f"Error during document splitting: {e}", exc_info=True)
             raise ValueError("Failed to split documents.") from e

        if not split_documents:
            raise ValueError("Splitting documents resulted in zero chunks. Check chunk size/overlap and document content.")

        logger.info(f"Split into {len(split_documents)} document chunks.")

        logger.info("Creating ChromaDB vector store (in-memory)...")
        try:
            if not hasattr(self, 'embeddings') or self.embeddings is None:
                 raise ValueError("Embeddings model not initialized.")

            self.db = Chroma.from_documents(
                 documents=split_documents,
                 embedding=self.embeddings
            )
            logger.info("ChromaDB vector store created successfully.")
        except Exception as e:
            logger.error(f"Failed to create ChromaDB store: {e}", exc_info=True)
            raise ValueError("Failed to create vector store.") from e

        logger.info(f"Creating retriever (k={self.search_k})...")
        self.retriever = self.db.as_retriever(
            search_type="similarity",
            search_kwargs={'k': self.search_k}
        )
        logger.info("Retriever created.")
    
    def add_data(self, directory):
        """
        Adds documents from a specified directory to the existing vector store,
        including source metadata.
        """
        logger.info(f"Adding data from directory: {directory}")
        if not hasattr(self, 'db') or self.db is None:
            logger.error("Cannot add data: Database not initialized.")
            return
        if not os.path.isdir(directory):
             logger.error(f"Cannot add data: Provided path is not a directory: {directory}")
             return

        added_chunks_count = 0
        for file_path in self._recurse_filepaths(directory):
            # _document_loader now adds metadata
            raw_documents = self._document_loader(file_path)

            # FIX 3: Check if documents were actually loaded before splitting/adding
            if not raw_documents:
                continue # Skip this file if loading failed or it was empty

            text_splitter = CharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separator="\n\n",
                is_separator_regex=False,
            )
            documents_to_add = text_splitter.split_documents(raw_documents)

            if documents_to_add:
                try:
                    # Assuming CharacterTextSplitter preserves metadata
                    self.db.add_documents(documents=documents_to_add)
                    added_chunks_count += len(documents_to_add)
                    logger.info(f"Added {len(documents_to_add)} chunks from {os.path.basename(file_path)} to the vector store.")
                except Exception as e:
                     logger.error(f"Failed to add documents from {file_path} to vector store: {e}", exc_info=False)

        if added_chunks_count > 0:
            logger.info(f"Finished adding data. Total new chunks added: {added_chunks_count}")
        else:
            logger.warning(f"No new documents were added from directory: {directory}")
    
    def _recurse_filepaths(self, directory: str) -> Iterator[str]:
        """
        Recursively finds all files with supported extensions within a directory using yield.

        Args:
            directory (str): The base directory to search within.

        Yields:
            str: The full path to each supported file found.
        """
        supported_extensions = {".txt", ".pdf", ".md", ".html"}
        logger.info(f"Scanning directory '{directory}' for supported files ({', '.join(supported_extensions)})...")
        found_files = False
        if not os.path.isdir(directory):
             raise FileNotFoundError(f"The directory {directory} does not exist or is not a directory.")

        for root, _, files in os.walk(directory):
            for file in files:
                try:
                    # Skip hidden files/folders
                    if file.startswith('.'):
                        continue
                    file_path = os.path.join(root, file)
                    _, file_ext = os.path.splitext(file_path)
                    if os.path.isdir(file_path):
                        # If the file path is a directory, recurse into it
                        yield from self._recurse_filepaths(file_path)
                        found_files = True
                    else:
                        yield file_path  # Yield the file path regardless of extension
                        found_files = True
                except Exception as e:
                    logger.warning(f"Could not process file entry '{file}' in '{root}': {e}")
        if not found_files:
             logger.warning(f"No files with supported extensions found in directory: {directory}")

    # based on the type of file choose what Loader to use, and then return the raw_documents:
    # loader = TextLoader(self.data_file_path)
    # raw_documents = loader.load()
    def _document_loader(self, file_path: str) -> List[Document]:
        """
        Loads documents from a single file using the appropriate Langchain loader
        and adds the ABSOLUTE filepath to metadata['source'].
        """
        try:
            _, file_ext = os.path.splitext(file_path)
            file_ext_lower = file_ext.lower()
            loader = None

            supported_extensions = {".txt", ".pdf", ".md", ".html", ".docx"}
            if file_ext_lower not in supported_extensions:
                 logger.warning(f"Unsupported file type encountered (should have been filtered): {file_path}")
                 return []

            logger.info(f"Attempting to load document: {file_path}")

            # Determine loader
            if file_ext_lower == ".txt":
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_ext_lower == ".pdf":
                loader = PyMuPDFLoader(file_path)
            elif file_ext_lower == ".md":
                loader = UnstructuredMarkdownLoader(file_path, mode="single")
            elif file_ext_lower == ".html":
                loader = UnstructuredHTMLLoader(file_path)
            elif file_ext_lower == ".docx":
                loader = UnstructuredWordDocumentLoader(file_path)
            elif os.path.getsize(file_path) > 500 * 1024 * 1024:
                logger.warning(f"File {file_path} Too Big")
                return []
            else:
                loader = UnstructuredFileLoader(file_path)
            # No else needed

            loaded_docs = loader.load()

            if not loaded_docs:
                 logger.warning(f"No documents extracted from file: {file_path}")
                 return []

            # Calculate relative path for logging purposes (optional)
            try:
                log_relative_path = os.path.relpath(file_path, self.base_directory)
            except ValueError:
                log_relative_path = os.path.basename(file_path)

            # Get the absolute path for storage
            absolute_file_path = os.path.abspath(file_path) # <-- Get absolute path

            for doc in loaded_docs:
                if not hasattr(doc, 'metadata') or doc.metadata is None:
                    doc.metadata = {}
                # --- Store the ABSOLUTE path in metadata ---
                doc.metadata["source"] = absolute_file_path
                # --- End of Fix ---

            # Log uses the absolute path now for clarity, or keep log_relative_path if preferred
            logger.info(f"Successfully loaded {len(loaded_docs)} doc(s) from {absolute_file_path}")
            return loaded_docs

        except Exception as e:
            logger.error(f"Failed to load document {file_path}: {e}", exc_info=False)
            return []


    def _format_docs(self, docs: List[Document]) -> str:
        """
        Helper function to format retrieved documents into a string,
        including the source metadata.
        """
        if not docs:
            return "No relevant context found."

        formatted_docs = []
        for i, doc in enumerate(docs):
            # Retrieve source from metadata, providing a default if missing
            source = doc.metadata.get("source", "Unknown Source")
            content = doc.page_content
            # Format the string to include both source and content
            formatted_docs.append(f"--- Context Piece {i+1} (Source: {source}) ---\n{content}")

        return "\n\n".join(formatted_docs)

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
        RAG_PROMPT_TEMPLATE = """SYSTEM: You are a helpful assistant. Use the following context pieces retrieved from a knowledge base to answer the user's question. If you don't know the answer based on the context, just say that you don't know. Keep the answer concise and relevant to the question. For the context you receive and reference, cite the source by printing the full file path at the end of your response.

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
