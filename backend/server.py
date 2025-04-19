from flask import Flask, Blueprint, request, jsonify, current_app
from flask_cors import CORS, cross_origin

# Create a Flask instance
api = Blueprint('api', __name__)
CORS(api, resources={r"/*": {"origins": "*"}})

@api.route('/api/add-hash', methods=['GET', 'POST'])
def add_hash():
    if request.method == 'POST':
        data = request.get_json()
        text = data.get('text')
        return jsonify({'result': '##' + text})
    else:
        # Read and return the contents of your JSON file
        return jsonify("hello")

@api.route('/api/process-prompt', methods=['POST'])
def process_prompt():
    if request.method == 'POST':
        data = request.get_json()
        prompt = data.get('prompt')

        model = current_app.config['LLM']
        response = model.chat(prompt)

        return jsonify({'response' : response})
    
    
# website prompts flask server, flask server prompts model for data (lane change idk wtf that is),
# lane chang shit i got lost, then query model for information, then smthing flask then frontend 
