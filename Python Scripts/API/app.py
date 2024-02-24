import os
from flask import Flask, request, jsonify
from model import SA_Model

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Access'})

@app.route("/emotion", methods=["POST", "GET"])
def result():
    if request.method == "POST":
        output = request.get_json() or request.form
        msg = output['msg'] 
        emotion = model.get_emotion(msg)
        return jsonify({'emotion': emotion}), 200

@app.errorhandler(500)
def server_error(error):
    return jsonify({
        "success": False,
        "error": 500,
        "message": "server error"
    }), 500


if __name__ == '__main__':
    model = SA_Model()
    port = int(os.environ.get("PORT", 2000))
    app.run(port=port, debug=True)
