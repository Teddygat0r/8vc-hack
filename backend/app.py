from flask import Flask
from . import server

from chatbot_class import Chatbot

def create_app():
    app = Flask(__name__)

    DATA_FILE = '../alldata/data.txt'
    try:
        app.config['LLM'] = Chatbot(base_directory=DATA_FILE)
    except FileNotFoundError as e:
        print(f"Error initializing chatbot: {e}")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during initialization: {e}")
        exit(1)

    

    app.register_blueprint(server.api)

    return app