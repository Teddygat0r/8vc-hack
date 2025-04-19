# mainfortesting.py

from chatbot_class import Chatbot # Import the class
import uuid # To generate unique thread IDs easily

# --- Configuration ---
DATA_DIR = './alldata'

if __name__ == "__main__":
    print("Creating Chatbot instance...")
    try:
        # Initialize the chatbot with the path to your data file
        chatbot = Chatbot(base_directory=DATA_DIR)
    except FileNotFoundError as e:
        print(f"Error initializing chatbot: {e}")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during initialization: {e}")
        exit(1)

    print("\nChatbot ready. Type 'quit' or 'exit' to end.")
    print("Starting a new conversation.")

    # Generate a unique ID for this conversation session
    current_thread_id = str(uuid.uuid4())
    print(f"Conversation Thread ID: {current_thread_id}")

    while True:
        user_message = input("You: ")
        if user_message.lower() in ["quit", "exit"]:
            print("Exiting chatbot.")
            break

        try:
            # Call the chat method with the user's message and the thread ID
            ai_message = chatbot.chat(user_message, current_thread_id)
            print(f"AI: {ai_message}")
        except Exception as e:
            # Catch errors during the chat invocation
            print(f"An error occurred during chat: {e}")
            # Depending on the error, you might want to break or continue

