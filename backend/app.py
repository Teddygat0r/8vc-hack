from flask import Flask
from . import server
import os

from chatbot_class import Chatbot

def create_app():
    app = Flask(__name__)

    DATA_DIR = os.path.join(app.root_path, '../alldata/')
    try:
        app.config['LLM'] = Chatbot(base_directory=DATA_DIR)
    except FileNotFoundError as e:
        print(f"Error initializing chatbot: {e}")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during initialization: {e}")
        exit(1)

    app.register_blueprint(server.api)

    return app