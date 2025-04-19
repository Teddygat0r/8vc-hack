from flask import Flask, request, jsonify

# Create a Flask instance
app = Flask(__name__)

# Define a route
@app.route('/', methods=['GET'])
def hello():
    return "Hello, World!"

# JSON endpoint
@app.route('/api/echo', methods=['GET', 'POST'])
def echo():
    #data = request.get_json()
    # data = {
    # "name": "John Doe",
    # "age": 30,
    # "city": "New York"
    # }
    # return jsonify(data)
    if request.method == 'POST':
        data = request.get_json()
        return jsonify({"you_sent": data})
    else:
        # Read and return the contents of your JSON file
        
        return jsonify("hello")
    #return jsonify(markdown": "##content",)

@app.route('/api/add-hash', methods=['GET', 'POST'])
def add_hash():
    if request.method == 'POST':
        data = request.get_json()
        text = data.get('text')
        return jsonify({'result': '##' + text})
    else:
        # Read and return the contents of your JSON file
        
        return jsonify("hello")
    


    
# website prompts flask server, flask server prompts model for data (lane change idk wtf that is),
# lane chang shit i got lost, then query model for information, then smthing flask then frontend 

# Run the server
if __name__ == '__main__':
    app.run(debug=True, port=5000)
